//
//  DuplicateView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-05.
//  Displays the duplicate clusters found in a ZNCC similarity matrix and which image to keep in each

import SwiftUI

struct DuplicateView : View {
    // Index i in `matrix` corresponds to `images[i]` -- the same order similarityMatrix received
    let images : [ImageData];
    let matrix : [[Float]];

    // @State is view-local, mutable storage that SwiftUI watches: when it changes, `body` re-runs
    // The slider writes to this, and each change re-evaluates `groups` below and redraws the list
    @State private var threshold : Double = 0.95;

    // A computed property: it isn't stored, it's recalculated every time it's read (i.e. every `body`
    // evaluation). Because reading it happens during `body`, changing `threshold` re-runs body, which
    // re-reads this and re-clusters with the new threshold
    private var groups : [DuplicateGroup] {
        duplicateGroups(matrix: self.matrix, threshold: Float(self.threshold));
    }

    // How many images are flagged for removal across all clusters.
    // reduce folds the array into one number: $0 is the running total, $1 is the current group.
    private var removalCount : Int {
        self.groups.reduce(0) { $0 + $1.duplicates.count; }
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
