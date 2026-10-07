//
//  MediaExporter.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-05.
//  Off-main helpers for getting images out of / removed from their source: copying/writing keepers to a
//  folder (NonDuplicateExporter) and deleting duplicates to the Trash / Recently Deleted (MediaRemover)

import Foundation
import Photos // For exporting/deleting photo-library assets (PHAssetResourceManager, PHAssetChangeRequest)

class NonDuplicateExporter {

    /// The outcome of an export: how many files copied, and which sources failed (with a reason string)
    /// Sendable (URL/String/Int are all Sendable) so it can be returned across actor boundaries.
    public struct ExportResult : Sendable {
        public let copied : Int;
        public let failures : [(source: URL, reason: String)];
    }

    /// Copies each of the given source files into the destination folder.
    /// `nonisolated` so it runs off the main actor (the project defaults isolation to MainActor) and takes
    /// only Sendable inputs, so it can be safely called from a background Task without data-race warnings
    /// - Parameters:
    ///     - sourceURLs: the files to copy (e.g. the keepers + unique images to keep)
    ///     - folder: the destination directory
    /// - Returns: A summary of how many files were copied and any per-file failures
    @discardableResult
    nonisolated public static func export(sourceURLs: [URL], to folder: URL) -> ExportResult {
        let fileManager = FileManager.default;

        // If the folder is security-scoped (sandbox), open access for the whole batch and close on exit
        let folderScoped = folder.startAccessingSecurityScopedResource();
        defer { if folderScoped { folder.stopAccessingSecurityScopedResource(); } }

        var copied = 0;
        var failures : [(source: URL, reason: String)] = [];

        for source in sourceURLs {
            // The source may also be security-scoped; open it just for this iteration (defer runs at the
            // end of each loop-body scope, so access is released before the next file)
            let sourceScoped = source.startAccessingSecurityScopedResource();
            defer { if sourceScoped { source.stopAccessingSecurityScopedResource(); } }

            // copyItem throws if the destination already exists, so pick a non-colliding name first
            let destination = self.uniqueDestination(for: source, in: folder, fileManager: fileManager);
            do {
                try fileManager.copyItem(at: source, to: destination);
                copied += 1;
            } catch {
                // Record and continue so one bad file doesn't abort the rest of the export
                failures.append((source, error.localizedDescription));
            }
        }

        return ExportResult(copied: copied, failures: failures);
    }

    /// Exports photo-library images by writing each asset's original resource bytes into the destination
    /// folder. Library assets have no on-disk file to copy, so PHAssetResourceManager streams the original
    /// data out instead. Runs off the main actor and takes only Sendable inputs
    /// - Parameters:
    ///     - identifiers: PhotoKit local identifiers of the assets to export
    ///     - folder: the destination directory
    /// - Returns: A summary of how many resources were written and any per-asset failures
    @discardableResult
    nonisolated static func exportLibraryAssets(identifiers: [String], to folder: URL) async -> ExportResult {
        guard !identifiers.isEmpty else { return ExportResult(copied: 0, failures: []); }

        // If the folder is security-scoped (sandbox), open access for the whole batch and close on exit
        let folderScoped = folder.startAccessingSecurityScopedResource();
        defer { if folderScoped { folder.stopAccessingSecurityScopedResource(); } }

        let fileManager = FileManager.default;
        var copied = 0;
        var failures : [(source: URL, reason: String)] = [];

        // Resolve the identifiers to PHAssets, then collect them into a plain array to iterate
        let fetched = PHAsset.fetchAssets(withLocalIdentifiers: identifiers, options: nil);
        var assets : [PHAsset] = [];
        fetched.enumerateObjects { asset, _, _ in assets.append(asset); }

        for asset in assets {
            // Prefer the full-size photo resource, fall back to whatever the asset exposes first
            let resources = PHAssetResource.assetResources(for: asset);
            guard let resource = resources.first(where: { $0.type == .photo }) ?? resources.first else {
                failures.append((folder, "No exportable resource for \(asset.localIdentifier)"));
                continue;
            }

            // Name the output after the asset's original filename, avoiding collisions in the folder
            let named = URL(fileURLWithPath: resource.originalFilename);
            let destination = self.uniqueDestination(for: named, in: folder, fileManager: fileManager);

            let options = PHAssetResourceRequestOptions();
            options.isNetworkAccessAllowed = true; // fetch from iCloud if the original isn't downloaded locally

            do {
                // writeData streams the resource bytes to the file, bridge its completion handler to async
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    PHAssetResourceManager.default().writeData(for: resource, toFile: destination, options: options) { error in
                        if let error { continuation.resume(throwing: error); } else { continuation.resume(); }
                    }
                }
                copied += 1;
            } catch {
                // Record and continue so one bad asset doesn't abort the rest of the export
                failures.append((destination, error.localizedDescription));
            }
        }

        return ExportResult(copied: copied, failures: failures);
    }

    /// Returns a URL inside `folder` that doesn't already exist, appending "-1", "-2", ... to the file name
    /// on collision (e.g. photo.jpg -> photo-1.jpg) so an existing file is never overwritten.
    nonisolated private static func uniqueDestination(for source: URL, in folder: URL, fileManager: FileManager) -> URL {
        var candidate = folder.appendingPathComponent(source.lastPathComponent);
        let fileExtension = source.pathExtension;                       // e.g. "jpg" (may be empty)
        let baseName = source.deletingPathExtension().lastPathComponent; // e.g. "photo"

        var counter = 1;
        while fileManager.fileExists(atPath: candidate.path) {
            let suffixedName = fileExtension.isEmpty ? "\(baseName)-\(counter)" : "\(baseName)-\(counter).\(fileExtension)";
            candidate = folder.appendingPathComponent(suffixedName);
            counter += 1;
        }
        return candidate;
    }
}

/// Removes images from their source. Both paths are recoverable: photo-library assets go to Recently Deleted,
/// on-disk files go to the Trash. Static + nonisolated so the blocking file work runs off the main actor
enum MediaRemover {

    /// Deletes the identified library assets. PhotoKit shows its own confirmation and routes to Recently
    /// Deleted rather than erasing immediately
    /// - Parameter identifiers: PhotoKit local identifiers of the assets to delete
    /// - Returns: true if the deletion committed, false if the user declined, it failed, or nothing resolved
    nonisolated static func deleteLibraryAssets(identifiers: [String]) async -> Bool {
        guard !identifiers.isEmpty else { return false; }
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: identifiers, options: nil);
        guard assets.count > 0 else { return false; }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.deleteAssets(assets);
            }
            return true;
        } catch {
            return false;
        }
    }

    /// Moves the given files to the Trash (recoverable). Skips any that fail rather than aborting the batch
    /// - Parameter urls: the source files to trash
    /// - Returns: the set of URLs that were successfully trashed
    nonisolated static func trashFiles(_ urls: [URL]) -> Set<URL> {
        let fileManager = FileManager.default;
        var trashed = Set<URL>();
        for url in urls {
            // User-selected files may be security-scoped; open access just for this operation
            let scoped = url.startAccessingSecurityScopedResource();
            defer { if scoped { url.stopAccessingSecurityScopedResource(); } }
            if (try? fileManager.trashItem(at: url, resultingItemURL: nil)) != nil {
                trashed.insert(url);
            }
        }
        return trashed;
    }
}
