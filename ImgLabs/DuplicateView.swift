//
//  DuplicateView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-05.
//  Displays the duplicate clusters found in a ZNCC similarity matrix and which image to keep in each

import SwiftUI
import AppKit // For NSOpenPanel (choosing the export folder)

class NonDuplicateExporter {

    /// The outcome of an export: how many files copied, and which sources failed (with a reason string).
    /// Sendable (URL/String/Int are all Sendable) so it can be returned across actor boundaries.
    public struct ExportResult : Sendable {
        public let copied : Int;
        public let failures : [(source: URL, reason: String)];
    }

    /// Copies each of the given source files into the destination folder.
    /// `nonisolated` so it runs off the main actor (the project defaults isolation to MainActor) and takes
    /// only Sendable inputs, so it can be safely called from a background Task without data-race warnings.
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

struct DuplicateView : View {
    // Index i in `matrix` corresponds to `images[i]` -- the same order similarityMatrix received
    let images : [ImageData];
    let matrix : [[Float]];
    // Per-image quality signals, aligned to `images`/`matrix`. Drives keeper selection. May be empty
    // (e.g. if quality couldn't be computed), in which case keeper selection falls back to the medoid
    let quality : [ImageQuality];
    // Shared status so the export can report progress and disable controls (via .copying) while it runs
    let status : AppStatusModel;

    // @State is view-local, mutable storage that SwiftUI watches: when it changes, `body` re-runs
    // The slider writes to this, and each change re-evaluates `groups` below and redraws the list
    @State private var threshold : Double = 0.95;

    // A computed property: it isn't stored, it's recalculated every time it's read (i.e. every `body`
    // evaluation). Because reading it happens during `body`, changing `threshold` re-runs body, which
    // re-reads this and re-clusters with the new threshold
    private var groups : [DuplicateGroup] {
        // Use the quality-aware strategy when quality signals are available and aligned, otherwise fall back
        // to the medoid so results are still sensible without them
        let strategy : KeeperStrategy = self.quality.count == self.matrix.count ? WeightedQualityStrategy() : MedoidStrategy();
        return duplicateGroups(matrix: self.matrix, threshold: Float(self.threshold), quality: self.quality, strategy: strategy);
    }

    // How many images are flagged for removal across all clusters.
    // reduce transforms the array into one number: $0 is the running total, $1 is the current group
    private var removalCount : Int {
        self.groups.reduce(0) { $0 + $1.duplicates.count; }
    }

    // The images to keep: every image EXCEPT the duplicates flagged for removal. This includes each
    // cluster's keeper AND all unique images (which belong to no cluster)
    private var imagesToKeep : [ImageData] {
        // flatMap concatenates every group's `duplicates` arrays into one flat [Int]; Set makes lookups O(1).
        let removeIndices = Set(self.groups.flatMap { $0.duplicates });
        // enumerated() pairs each element with its index ($0.offset); keep those not flagged for removal
        return self.images.enumerated()
            .filter { !removeIndices.contains($0.offset) }
            .map { $0.element };
    }

    // MARK: - Export

    /// Asks the user for a destination folder, then copies every kept image into it off the main thread.
    private func exportKeepers() {
        let panel = NSOpenPanel();
        panel.title = "Choose Export Folder";
        panel.prompt = "Export";
        panel.canChooseFiles = false;        // folders only
        panel.canChooseDirectories = true;
        panel.allowsMultipleSelection = false;
        panel.canCreateDirectories = true;   // let the user make a new folder in the panel

        panel.begin { response in
            // begin's completion runs on the main thread; bail unless the user picked a folder
            guard response == .OK, let folder = panel.url else { return; }

            // Resolve the source file URLs on the main thread. URLs are Sendable, so only Sendable values
            // (these URLs, the folder, and the status object) cross into the background task below
            let urlsToExport = self.imagesToKeep.map { $0.getURL() };
            let status = self.status;

            status.setPhase(to: .copying);
            status.setStatusMessage("Exporting images...");
            status.setProgress(0);

            // Task.detached runs OFF the main actor so the (synchronous, blocking) file copies don't freeze
            // the UI. Status updates are hopped back onto the main actor via MainActor.run
            Task.detached {
                let result = NonDuplicateExporter.export(sourceURLs: urlsToExport, to: folder);
                await MainActor.run {
                    status.setProgress(PROGRESS_END);
                    if result.failures.isEmpty {
                        status.setPhase(to: .finished);
                        status.setStatusMessage("Exported \(result.copied) images");
                    } else {
                        // Surface the first failure's reason so the cause is visible (e.g. a permissions error)
                        status.setPhase(to: .error);
                        let firstReason = result.failures.first?.reason ?? "unknown error";
                        status.setStatusMessage("Exported \(result.copied), \(result.failures.count) failed: \(firstReason)");
                    }
                }
            }
        }
    }

    var body : some View {
        VStack(alignment: .leading, spacing: 16) {
            self.header

            if self.groups.isEmpty {
                // A built-in placeholder view for "nothing to show" states
                ContentUnavailableView(
                    "No Duplicates",
                    systemImage: "checkmark.circle",
                    description: Text("No image pairs scored at or above \(self.threshold, format: .percent.precision(.fractionLength(0))).")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        // ForEach builds one view per element. It needs each element to be uniquely
                        // identifiable so SwiftUI can track them across updates -- DuplicateGroup is
                        // Identifiable (its UUID id), so can pass the array directly
                        ForEach(self.groups) { group in
                            self.groupSection(group);
                        }
                    }
                    .padding(.vertical, 8);
                }
                Button(action: {
                    self.exportKeepers();
                }) {
                    // How the button looks
                    Text("Export \(self.images.count - self.removalCount) Images")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .cornerRadius(10)
                }
                .disabled(self.status.isBusy) // Prevent re-triggering while another operation runs
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Subviews

    // Breaking `body` into smaller computed views keeps it readable; each returns `some View`
    private var header : some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Duplicates")
                .font(.system(size: 32, weight: .black))
                .frame(maxWidth: .infinity, alignment: .leading);

            // A summary of the current clustering
            Text("\(self.images.count) images \u{2192} \(self.images.count - self.removalCount) to keep, \(self.removalCount) to remove")
                .font(.headline)
                .foregroundStyle(.secondary);

            // Slider binds to $threshold. The `$` makes a two-way Binding: the slider both reads the
            // current value and writes new ones back into @State, which re-runs body
            HStack {
                Text("Sensitivity");
                Slider(value: $threshold, in: 0.5...1.0);
                // Show the threshold as a percentage (e.g. 95%)
                Text(self.threshold, format: .percent.precision(.fractionLength(0)))
                    .monospacedDigit() // keeps the number from shifting width as it changes
                    .frame(width: 44, alignment: .trailing);
            }
        }
    }

    // Constructs a group of thumbnails based off of a DuplicateGroup
    // Will display the keeper first, followed by the other images
    private func groupSection(_ group: DuplicateGroup) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Duplicate group")
                .font(.subheadline.bold())
                .foregroundStyle(.secondary);

            // A grid that flows thumbnails and wraps automatically. `adaptive` fits as many
            // columns of ~120pt as the width allows.
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 12)], spacing: 12) {
                // The keeper first, then the duplicates
                self.thumbnail(for: group.keep, isKeeper: true, keeperIndex: group.keep);
                // ForEach over plain Ints needs an explicit id since Int isn't Identifiable on its own;
                // \.self uses the value itself as the identity
                ForEach(group.duplicates, id: \.self) { index in
                    self.thumbnail(for: index, isKeeper: false, keeperIndex: group.keep);
                }
            }
        }
    }

    // Constructs a view for the given image, labeled if it's a keeper or not
    private func thumbnail(for index: Int, isKeeper: Bool, keeperIndex: Int) -> some View {
        VStack(spacing: 6) {
            // Build a SwiftUI Image from the ImageData's CGImage. `decorative` means it carries no
            // accessibility label (it's purely visual). getCGImage() is optional, fall back to
            // a placeholder if it's somehow missing
            Group {
                if let cgImage = self.images[index].getCGImage() {
                    Image(decorative: cgImage, scale: 1.0)
                        .resizable()
                        .scaledToFill();
                } else {
                    Image(systemName: "photo")
                        .resizable()
                        .scaledToFit()
                        .foregroundStyle(.secondary);
                }
            }
            .frame(width: 120, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                // Green outline for the image kept, red for ones suggested for removal
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isKeeper ? Color.green : Color.red, lineWidth: 3)
            );

            if isKeeper {
                Label("Keep", systemImage: "checkmark.circle.fill")
                    .font(.caption.bold())
                    .foregroundStyle(.green);
            } else {
                // Show how similar this duplicate is to the keeper (its ZNCC score against the keeper),
                // formatted as a percentage so the user understands why it was flagged
                Label {
                    Text(Double(self.matrix[keeperIndex][index]), format: .percent.precision(.fractionLength(0)));
                } icon: {
                    Image(systemName: "xmark.circle.fill");
                }
                .font(.caption)
                .foregroundStyle(.red);
            }
        }
    }
}
