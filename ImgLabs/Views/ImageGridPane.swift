//
//  ImageGridPane.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  The left-hand results pane: shows the analyzing/empty placeholders or the duplicate clusters

import SwiftUI

struct ImageGridPane : View {
    // Shared state from the parent. Reading these observable properties makes this pane re-render
    // automatically when the analysis phase changes or a matrix is produced
    let model : ImageModel;
    let status : AppStatusModel;
    let znccObj : ZNCCModel;

    // Results are only safe to show when the matrix matches the current image set. After clearing or
    // importing more images without re-analyzing, the old matrix is stale (a different size), so we treat
    // it as "no results" rather than indexing past the end of the images array (which would crash)
    private var hasValidResults : Bool {
        // The matrix is strided N*N; it matches the current image set only when its length is imageCount squared
        let imageCount = model.images.count;
        return !znccObj.results.isEmpty && znccObj.results.count == imageCount * imageCount;
    }

    var body : some View {
        VStack {
            if status.curPhase == .analyzing {
                // Something is being calculated
                self.stateIndicator(icon: "wand.and.rays", message: "Analyzing images...");
            } else if hasValidResults {
                DuplicateView(images: model.images, matrix: znccObj.results, quality: znccObj.quality, hammingMatrix: znccObj.hammingMatrix, status: status, model: model, onImagesDeleted: { znccObj.clear(); });
            } else {
                // Nothing to show yet
                self.stateIndicator(icon: "photo.on.rectangle.angled", message: "Import photos and press Analyze to find duplicates");
            }
        }
        // Fill the available space, then sit on a full-bleed material surface: this is the
        // edge-to-edge main content area, with only the sidebar floating as a glass inspector. The material
        // keeps text/thumbnails readable instead of placing them directly on the window gradient
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
        .background(.regularMaterial)
    }

    // A centered, animated placeholder used for the "calculating" and "empty" states
    // .symbolEffect(.pulse, ...) animates the SF Symbol; .repeat(.continuous) keeps it looping
    private func stateIndicator(icon: String, message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 60))
                .foregroundStyle(.secondary)
                .symbolEffect(.pulse, options: .repeat(.continuous), isActive: true);
            Text(message)
                .font(.headline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center);
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
