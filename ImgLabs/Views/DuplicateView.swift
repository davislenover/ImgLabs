//
//  DuplicateView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-05.
//  Displays the duplicate clusters found in a ZNCC similarity matrix and which image to keep in each

import SwiftUI
import AppKit // For NSOpenPanel (choosing the export folder)

struct DuplicateView : View {
    // Strided (row-major) NxN similarity matrix; element (i, j) at i * images.count + j corresponds to
    // images[i] vs images[j] -- the same order similarityMatrix received
    let images : [ImageData];
    let matrix : [Float];
    // Per-image quality signals, aligned to `images`. Drives keeper selection. May be empty
    // (e.g. if quality couldn't be computed), in which case keeper selection falls back to the medoid
    let quality : [ImageQuality];
    // Strided (row-major) NxN Hamming-distance matrix of the images' perceptual hashes, aligned to `matrix`.
    // Adds near-duplicate edges to the clustering. May be empty, in which case grouping stays purely ZNCC-driven
    let hammingMatrix : [UInt8];
    // Shared status so the export can report progress and disable controls (via .copying) while it runs
    let status : AppStatusModel;
    // Owns the imported images, used to delete flagged duplicates (library -> Recently Deleted, files -> Trash)
    let model : ImageModel;
    // Called after a delete succeeds so the parent can drop the now-stale analysis results
    let onImagesDeleted : () -> Void;

    // @State is view-local, mutable storage that SwiftUI watches: when it changes, `body` re-runs
    // The slider writes to this, and each change re-evaluates `groups` below and redraws the list
    @State private var threshold : Double = 0.95;

    // Max Hamming distance (in bits, out of 64) at which two images' perceptual hashes are treated as a
    // near-duplicate. Higher = more tolerant = more matches. Its slider re-runs body the same way as threshold
    @State private var hashThreshold : Double = 8;

    // Drives the confirmation alert shown before deleting duplicates (file trashing has no OS prompt of its own)
    @State private var showDeleteConfirm : Bool = false;

    // Side length of each duplicate thumbnail. Bumped up so the image groups get plenty of room to breathe
    private let thumbSize : CGFloat = 172;

    // A computed property: it isn't stored, it's recalculated every time it's read (i.e. every `body`
    // evaluation). Because reading it happens during `body`, changing `threshold` re-runs body, which
    // re-reads this and re-clusters with the new threshold
    private var groups : [DuplicateGroup] {
        // Use the quality-aware strategy when quality signals are available and aligned, otherwise fall back
        // to the medoid so results are still sensible without them
        let strategy : KeeperStrategy = self.quality.count == self.images.count ? WeightedQualityStrategy() : MedoidStrategy();
        return duplicateGroups(matrix: self.matrix, count: self.images.count, threshold: Float(self.threshold), hammingMatrix: self.hammingMatrix, hashThreshold: UInt8(self.hashThreshold), quality: self.quality, strategy: strategy);
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

    // Every flagged (red) duplicate across all clusters, the images suggested for removal
    private var flaggedDuplicates : [ImageData] {
        self.groups.flatMap { $0.duplicates }.map { self.images[$0] };
    }

    // MARK: - Delete

    /// Deletes every flagged duplicate through the model (library assets -> Recently Deleted, files -> Trash),
    /// then asks the parent to drop the now-stale results so the user re-analyzes the reduced set
    private func deleteDuplicates() {
        let toDelete = self.flaggedDuplicates;
        guard !toDelete.isEmpty else { return; }
        let model = self.model;
        let status = self.status;
        let onDeleted = self.onImagesDeleted;
        Task {
            let deleted = await model.delete(toDelete, status);
            // Only invalidate results if something was actually removed (user may cancel the system prompt)
            if deleted > 0 { onDeleted(); }
        }
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

            // Split the keepers by origin: file imports have a real on-disk URL to copy; library imports have
            // no file, so they're written out from their PHAsset via PHAssetResourceManager. Both use only
            // Sendable values (URLs, identifiers, folder, status), which is what crosses into the task below
            let keepers = self.imagesToKeep;
            let fileURLs = keepers.filter { $0.getAssetIdentifier() == nil }.map { $0.getURL() };
            let libraryIDs = keepers.compactMap { $0.getAssetIdentifier() };
            let status = self.status;

            status.setPhase(to: .copying);
            status.setStatusMessage("Exporting images...");
            status.setProgress(0);

            // Task.detached runs OFF the main actor so the (synchronous, blocking) file copies don't freeze
            // the UI. Status updates are hopped back onto the main actor via MainActor.run
            Task.detached {
                // Copy file-origin keepers, then stream library-origin keepers out; combine the tallies
                let fileResult = NonDuplicateExporter.export(sourceURLs: fileURLs, to: folder);
                let libResult = await NonDuplicateExporter.exportLibraryAssets(identifiers: libraryIDs, to: folder);
                let copied = fileResult.copied + libResult.copied;
                let failures = fileResult.failures + libResult.failures;
                await MainActor.run {
                    status.setProgress(PROGRESS_END);
                    if failures.isEmpty {
                        status.setPhase(to: .finished);
                        status.setStatusMessage("Exported \(copied) images");
                    } else {
                        // Surface the first failure's reason so the cause is visible (e.g. a permissions error)
                        status.setPhase(to: .error);
                        let firstReason = failures.first?.reason ?? "unknown error";
                        status.setStatusMessage("Exported \(copied), \(failures.count) failed: \(firstReason)");
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
                        // identifiable so SwiftUI can track them across updates. DuplicateGroup is
                        // Identifiable (its UUID id), so can pass the array directly
                        ForEach(self.groups) { group in
                            self.groupSection(group);
                        }
                    }
                    .padding(.vertical, 8);
                }
                self.actionBar
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Subviews

    // The Export + Delete buttons shown beneath the cluster list
    private var actionBar : some View {
        HStack {
            // Copy the images being kept (cluster keepers + uniques) to a folder
            Button {
                self.exportKeepers();
            } label: {
                Label("Export \(self.images.count - self.removalCount) Keepers", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(FilledActionButtonStyle(fill: .brandPrimary))
            .help("Copy every image you're keeping (cluster keepers + uniques) into a folder you choose")
            .disabled(self.status.isBusy) // Prevent re-triggering while another operation runs

            // Delete every flagged duplicate. Shown whenever there are flagged duplicates
            if !self.flaggedDuplicates.isEmpty {
                Button {
                    self.showDeleteConfirm = true;
                } label: {
                    Label("Delete \(self.flaggedDuplicates.count) Duplicates", systemImage: "trash")
                }
                .buttonStyle(FilledActionButtonStyle(fill: .red))
                .help("Move the flagged duplicates to the Trash / Recently Deleted (both recoverable)")
                .disabled(self.status.isBusy)
                .alert("Delete \(self.flaggedDuplicates.count) duplicates?", isPresented: $showDeleteConfirm) {
                    Button("Cancel", role: .cancel) { }
                    Button("Delete", role: .destructive) { self.deleteDuplicates(); }
                } message: {
                    Text("Photos are moved to Recently Deleted and files are moved to the Trash. Both are recoverable.");
                };
            }
        }
    }

    // Breaking `body` into smaller computed views keeps it readable; each returns `some View`
    private var header : some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Duplicates")
                .font(.system(size: 32, weight: .black))
                .foregroundStyle(.brandPrimary)
                .frame(maxWidth: .infinity, alignment: .leading);

            // A summary of the current clustering
            Text("\(self.images.count) images \u{2192} \(self.images.count - self.removalCount) to keep, \(self.removalCount) to remove")
                .font(.headline)
                .foregroundStyle(.secondary);

            // Pixel-similarity threshold
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Sensitivity")
                        .font(.subheadline.bold());
                    Spacer();
                    // Show the threshold as a percentage (e.g. 95%)
                    Text(self.threshold, format: .percent.precision(.fractionLength(0)))
                        .font(.caption).monospacedDigit().foregroundStyle(.brandPrimary); // steady width as it changes
                }
                Slider(value: $threshold, in: 0.5...1.0)
                    .tint(.brandSecondary)
                    .help("How alike two images must be (by pixel correlation) to be grouped as duplicates");
                Text("Higher groups only very-close matches; lower catches looser look-alikes.")
                    .font(.caption2).foregroundStyle(.secondary);
            }

            // Perceptual-hash tolerance
            if self.hammingMatrix.count == self.matrix.count {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Hash tolerance")
                            .font(.subheadline.bold());
                        Spacer();
                        // Show the current cutoff as "<= N bits" (out of 64)
                        Text("\u{2264} \(Int(self.hashThreshold)) bits")
                            .font(.caption).monospacedDigit().foregroundStyle(.brandPrimary);
                    }
                    Slider(value: $hashThreshold, in: 0...16, step: 1)
                        .tint(.brandSecondary)
                        .help("How many of the 64 perceptual-hash bits may differ and still count as a near-duplicate");
                    Text("Catches crops, recompression and colour shifts that pixel matching can miss.")
                        .font(.caption2).foregroundStyle(.secondary);
                }
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
            // columns of ~thumbSize as the (now edge-to-edge) width allows.
            LazyVGrid(columns: [GridItem(.adaptive(minimum: thumbSize), spacing: 16)], spacing: 16) {
                // The keeper first, then the duplicates
                self.thumbnail(for: group.keep, isKeeper: true, keeperIndex: group.keep);
                // ForEach over plain Ints needs an explicit id since Int isn't Identifiable on its own
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
            .frame(width: thumbSize, height: thumbSize)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                // Green outline for the image kept, red for ones suggested for removal
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isKeeper ? Color.green : Color.red, lineWidth: 3)
            );

            // The image's file name (the real name for both file and library imports). Middle-truncated so the
            // extension stays visible, full name on hover
            Text(self.images[index].getURL().lastPathComponent)
                .font(.caption2)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(width: thumbSize)
                .help(self.images[index].getURL().lastPathComponent);

            if isKeeper {
                Label("Keep", systemImage: "checkmark.circle.fill")
                    .font(.caption.bold())
                    .foregroundStyle(.green);
            } else {
                // Show how similar this duplicate is to the keeper (its ZNCC score against the keeper),
                // formatted as a percentage so the user understands why it was flagged
                Label {
                    // Strided index into the row-major matrix: (keeperIndex, index) at keeperIndex * N + index
                    Text(Double(self.matrix[keeperIndex * self.images.count + index]), format: .percent.precision(.fractionLength(0)));
                } icon: {
                    Image(systemName: "xmark.circle.fill");
                }
                .font(.caption)
                .foregroundStyle(.red);
            }
        }
    }
}
