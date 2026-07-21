//
//  ReferenceIndexer.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-20
//  Denotes a class which builds ReferenceEntries and corresponding LibraryRoot objects

import Foundation
import CoreGraphics

public class ReferenceIndexer {
    private let MTLContext: MetalComputeContext;
    // Reference images are only ever squashed to a 32x32 pHash, so they don't need the full analysis canvas
    private static let decodePixelSize : Int = 512;
    // How many images are decoded + hashed per pass. Bounds peak memory: an entire directory's worth of
    // decoded bitmaps would otherwise sit resident at once (a big library is many GB). Decode -> hash ->
    // release one chunk before moving to the next
    private static let hashChunkSize : Int = 200;

    init(MTLContext: MetalComputeContext) {
        self.MTLContext = MTLContext;
    }

    /// Turns a set of directories into the pieces a ReferenceLibrary is built from: one LibraryRoot per
    /// directory (with a security-scoped bookmark so the folder can be re-reached on later launches) and a
    /// ReferenceEntry (pHash + file metadata) per image found beneath it. Does not persist anything!
    /// Hand the result to ReferenceManager.createReferenceLibrary
    /// - Parameter from: the directories the user chose to make up the library
    /// - Returns: the roots and their entries, index-independent (entries carry their owning root's id)
    public func build(_ from : [URL]) async throws -> (roots: [LibraryRoot], references: [ReferenceEntry]) {
        var roots : [LibraryRoot] = [];
        var references : [ReferenceEntry] = [];

        // Drop any directory nested inside another chosen directory, so its images aren't hashed twice
        // (once under each root). Also collapses exact duplicates
        let rootDirs : [URL] = Self.nonOverlappingRoots(from);

        for dir in rootDirs {
            // A user-picked folder is security-scoped thus hold access for the whole walk/decode and mint the
            // bookmark while access is open (bookmarking a folder unable to be reached would fail)
            let didAccess : Bool = dir.startAccessingSecurityScopedResource();
            defer { if didAccess { dir.stopAccessingSecurityScopedResource(); } }

            // .withSecurityScope persists the grant across launches. Throws until that entitlement is present
            let bookmark : Data = try dir.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil);
            let root : LibraryRoot = LibraryRoot(id: UUID(), bookmark: bookmark, displayPath: dir.path(percentEncoded: false));
            roots.append(root);

            // Every image beneath this root, hashed into entries (memory-bounded chunks)
            let allImageURLs : [URL] = ImageModel.imageFileURLs(from: [dir]);
            references.append(contentsOf: try await self.makeEntries(for: allImageURLs, rootID: root.id, rootDir: dir));
        }

        return (roots: roots, references: references);
    }

    /// Re-scans an existing library against what is currently on disk, re-hashing only what changed.
    /// For each root (resolved from its bookmark): unchanged files (same relative path, byte size and
    /// modified date) keep their existing entry untouched. New or modified files are re-hashed and files that
    /// no longer exist are dropped. A root whose folder can't be reached keeps its entries as-is and is skipped.
    /// A stale-but-resolvable bookmark is refreshed
    /// - Parameter refLib: the library to re-scan
    /// - Returns: a new ReferenceLibrary (same id) reflecting the changes, or nil when nothing changed. Persist it via ReferenceManager.commitReferenceLibrary
    public func rescan(_ refLib: ReferenceLibrary) async throws -> ReferenceLibrary? {
        var newRoots : [LibraryRoot] = [];
        var newEntries : [ReferenceEntry] = [];
        var changed : Bool = false;

        for root in refLib.roots {
            // This root's existing entries, keyed by relative path (unique per root -- it's a file path)
            let existingForRoot : [ReferenceEntry] = refLib.entries.filter { $0.libraryRootID == root.id };
            var existingByPath : [String: ReferenceEntry] = [:];
            for entry in existingForRoot { existingByPath[entry.relativePath] = entry; }

            // Resolve the folder from its bookmark. If it can't be reached (e.g. volume offline), leave this
            // root and its entries exactly as they were and move on
            guard let resolved = Self.resolveRoot(root) else {
                newRoots.append(root);
                newEntries.append(contentsOf: existingForRoot);
                continue;
            }
            let dir : URL = resolved.url;
            let didAccess : Bool = dir.startAccessingSecurityScopedResource();
            defer { if didAccess { dir.stopAccessingSecurityScopedResource(); } }

            // A stale bookmark still resolved, but should be refreshed so it keeps working next launch
            if resolved.isStale, didAccess, let refreshed = try? dir.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil) {
                newRoots.append(LibraryRoot(id: root.id, bookmark: refreshed, displayPath: dir.path(percentEncoded: false)));
                changed = true;
            } else {
                newRoots.append(root);
            }

            // Diff current files against the existing entries
            let currentURLs : [URL] = ImageModel.imageFileURLs(from: [dir]);
            var reusable : [ReferenceEntry] = [];  // unchanged -> keep existing entry, no re-hash
            var toHashURLs : [URL] = [];           // new or modified -> re-hash
            var seenPaths : Set<String> = [];
            for url in currentURLs {
                let relativePath : String = Self.relativePath(of: url, under: dir);
                seenPaths.insert(relativePath);
                let metadata = Self.fileMetadata(for: url);
                if let existing = existingByPath[relativePath],
                   existing.byteSize == metadata.byteSize,
                   existing.modified == metadata.modified {
                    reusable.append(existing);
                } else {
                    toHashURLs.append(url);
                }
            }

            // Any existing path not seen on disk was deleted (it simply won't be carried into reusable)
            let removedPaths : Set<String> = Set(existingByPath.keys).subtracting(seenPaths);
            if !toHashURLs.isEmpty || !removedPaths.isEmpty { changed = true; }

            newEntries.append(contentsOf: reusable);
            newEntries.append(contentsOf: try await self.makeEntries(for: toHashURLs, rootID: root.id, rootDir: dir));
        }

        guard changed else { return nil; } // nothing to persist
        return ReferenceLibrary(id: refLib.id, name: refLib.name, roots: newRoots, entries: newEntries, lastScanned: Date(), formatVersion: refLib.formatVersion);
    }

    // MARK: - Entry building

    /// Decodes + pHashes the given image URLs into ReferenceEntry values, in memory-bounded chunks
    /// (decode -> hash -> release, so a large set never sits fully decoded in RAM). Undecodable files are
    /// skipped. Shared by `build` (whole root) and `rescan` (only the new/modified files)
    private func makeEntries(for urls: [URL], rootID: UUID, rootDir: URL) async throws -> [ReferenceEntry] {
        var entries : [ReferenceEntry] = [];
        var start : Int = 0;
        while start < urls.count {
            let end : Int = min(start + Self.hashChunkSize, urls.count);
            let chunkURLs : [URL] = Array(urls[start..<end]);

            // Decode this chunk concurrently (nil where a file couldn't be read)
            let decoded : [CGImage?] = await Self.decodeImages(at: chunkURLs, maxPixelSize: Self.decodePixelSize);

            // Build an ImageData for each decodable image, remembering its URL so metadata lines up
            var imageDatas : [ImageData] = [];
            var urlsForData : [URL] = [];
            for (offset, cgImage) in decoded.enumerated() {
                guard let cgImage else { continue; } // undecodable file -> skip, don't abort the scan
                imageDatas.append(ImageData(img: cgImage, targetWidth: Self.decodePixelSize, targetHeight: Self.decodePixelSize, filePath: chunkURLs[offset]));
                urlsForData.append(chunkURLs[offset]);
            }

            if !imageDatas.isEmpty {
                // toHash keeps input order, so hash i corresponds to urlsForData[i]
                let hashes : [ImageHash] = try await ImageHash.toHash(images: imageDatas, MTLContext: self.MTLContext);
                for (offset, hashObject) in hashes.enumerated() {
                    let url : URL = urlsForData[offset];
                    let metadata = Self.fileMetadata(for: url);
                    entries.append(ReferenceEntry(
                        hash: await hashObject.value(),
                        libraryRootID: rootID,
                        relativePath: Self.relativePath(of: url, under: rootDir),
                        fileName: url.lastPathComponent,
                        byteSize: metadata.byteSize,
                        modified: metadata.modified
                    ));
                }
            }

            start = end;
        }
        return entries;
    }

    // MARK: - Metadata / path helpers

    /// Resolves a root's directory URL from its security-scoped bookmark, reporting whether the bookmark is
    /// stale (still usable, but should be re-created). Returns nil if it can't be resolved at all
    private nonisolated static func resolveRoot(_ root: LibraryRoot) -> (url: URL, isStale: Bool)? {
        var isStale : Bool = false;
        guard let url = try? URL(resolvingBookmarkData: root.bookmark, options: [.withSecurityScope], relativeTo: nil, bookmarkDataIsStale: &isStale) else {
            return nil;
        }
        return (url, isStale);
    }

    /// The file's byte size and last-modified date (defaults when the values can't be read). Together they
    /// form the "has this file changed?" key that lets a later re-scan skip unchanged images
    private nonisolated static func fileMetadata(for url: URL) -> (byteSize: Int64, modified: Date) {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]);
        let byteSize : Int64 = Int64(values?.fileSize ?? 0);
        let modified : Date = values?.contentModificationDate ?? Date(timeIntervalSince1970: 0);
        return (byteSize, modified);
    }

    /// The path of `fileURL` relative to `root` (e.g. "2024/IMG_2201.jpg"), so an entry survives the root
    /// moving (resolved back to an absolute path via the root's bookmark). Falls back to the bare filename
    private nonisolated static func relativePath(of fileURL: URL, under root: URL) -> String {
        let rootComponents : [String] = root.standardizedFileURL.pathComponents;
        let fileComponents : [String] = fileURL.standardizedFileURL.pathComponents;
        guard fileComponents.count > rootComponents.count,
              Array(fileComponents.prefix(rootComponents.count)) == rootComponents else {
            return fileURL.lastPathComponent;
        }
        return fileComponents.dropFirst(rootComponents.count).joined(separator: "/");
    }

    /// Removes any directory that lives inside another directory in the same set (and exact duplicates), so
    /// overlapping picks don't produce duplicate entries under different roots
    private nonisolated static func nonOverlappingRoots(_ urls: [URL]) -> [URL] {
        let standardized : [URL] = urls.map { $0.standardizedFileURL };
        var result : [URL] = [];
        for url in standardized {
            let nestedInAnother : Bool = standardized.contains { other in Self.isDescendant(url, of: other); };
            if !nestedInAnother && !result.contains(url) { result.append(url); }
        }
        return result;
    }

    /// Whether `url` sits strictly beneath `ancestor` in the file tree (equal paths are not descendants)
    private nonisolated static func isDescendant(_ url: URL, of ancestor: URL) -> Bool {
        let urlComponents : [String] = url.pathComponents;
        let ancestorComponents : [String] = ancestor.pathComponents;
        return urlComponents.count > ancestorComponents.count
            && Array(urlComponents.prefix(ancestorComponents.count)) == ancestorComponents;
    }

    /// Decodes every image URL concurrently, keeping the result index-aligned to `urls`
    /// (a slot is nil when its file couldn't be decoded, so one bad file doesn't abort the batch).
    /// At most `activeProcessorCount` decodes run at once: each CGImageSource decode spikes memory,
    /// so an unbounded fan-out over a large directory would oversubscribe both CPU and RAM
    private nonisolated static func decodeImages(at urls: [URL], maxPixelSize: Int) async -> [CGImage?] {
        var results : [CGImage?] = Array<CGImage?>(repeating: nil, count: urls.count);
        guard !urls.isEmpty else { return results; }
        let maxInFlight : Int = max(1, ProcessInfo.processInfo.activeProcessorCount);

        await withTaskGroup(of: (Int, CGImage?).self) { group in
            var next : Int = 0;
            // Prime the group with up to maxInFlight decodes
            while next < urls.count && next < maxInFlight {
                let index : Int = next;
                let url : URL = urls[index];
                group.addTask { (index, ImageModel.decodedImage(at: url, maxPixelSize: maxPixelSize)); }
                next += 1;
            }
            // As each decode finishes, record it and enqueue the next, keeping ~maxInFlight running
            for await (index, image) in group {
                results[index] = image;
                if next < urls.count {
                    let nextIndex : Int = next;
                    let url : URL = urls[nextIndex];
                    group.addTask { (nextIndex, ImageModel.decodedImage(at: url, maxPixelSize: maxPixelSize)); }
                    next += 1;
                }
            }
        }
        return results;
    }

}

