//
//  ControlSideBar.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  The right-hand control panel: import (file + Photos), clear, analyze, canvas-size, and the benchmark tool.

import SwiftUI
import PhotosUI // For PHPickerFilter.images passed to the Photo Library import

struct ControlSideBar : View {
    // Received from the parent (ContentView) so the grid pane can share the same state
    // These are @Observable classes, so reading their properties here observes them for changes
    let model: ImageModel;
    let status : AppStatusModel;
    let znccObj : ZNCCModel;
    @Binding var maxCanvasDim : Double;
    // Drives the confirmation alert shown when importing over existing images
    @State private var showImportConfirm : Bool = false;
    // Same, but for the photo-library (PhotoKit) import path
    @State private var showPhotoConfirm : Bool = false;
    

    /// Opens the Finder panel to import images (wrapped so both the button and alert can call it)
    private func startImport() {
        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) { // Animate in the Clear button
            model.browseForImages(status,Int(maxCanvasDim));
        }
    }

    /// Opens the PhotoKit picker to import from the photo library (wrapped so both the button and alert can call it)
    private func startPhotoImport() {
        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) { // Animate in the Clear button
            model.browseForImages(status, _maxCanvasDim: Int(maxCanvasDim), .images);
        }
    }

    var body: some View {
        VStack {
            Text("ImgLabs")
                .font(.system(size: 40, weight: .black))
                .foregroundStyle(.brandPrimary)
            // A single column of full-width buttons
            VStack(spacing: 10) {
                // Import from disk (files or a folder)
                Button {
                    // If images already exist, confirm first; otherwise import straight away
                    if model.containsImages() {
                        showImportConfirm = true;
                    } else {
                        startImport();
                    }
                } label: {
                    Label("Import Files", systemImage: "folder.badge.plus")
                }
                .buttonStyle(FilledActionButtonStyle(fill: .brandPrimary))
                .help("Import image files or a folder from your Mac (you can also drag & drop here)")
                .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                .alert("Import more images?", isPresented: $showImportConfirm) {
                    Button("Cancel", role: .cancel) { }
                    Button("Import") { startImport(); }
                } message: {
                    Text("You already have images imported. Newly selected images will be added to your current imports.");
                }
                // Import from the Photos library via PhotoKit (assets can later be deleted to Recently Deleted)
                Button {
                    // If images already exist, confirm first; otherwise import straight away
                    if model.containsImages() {
                        showPhotoConfirm = true;
                    } else {
                        startPhotoImport();
                    }
                } label: {
                    Label("Photo Library", systemImage: "photo.stack")
                }
                .buttonStyle(FilledActionButtonStyle(fill: .brandSecondary))
                .help("Import from your Photos library. These can later be deleted to Recently Deleted")
                .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                .alert("Import more images?", isPresented: $showPhotoConfirm) {
                    Button("Cancel", role: .cancel) { }
                    Button("Import") { startPhotoImport(); }
                } message: {
                    Text("You already have images imported. Newly selected images will be added to your current imports.");
                }
                // Clear + Analyze slide in once images are imported. They render OVER a glass BACKGROUND layer
                if model.containsImages() {
                    // Empty the in-app set (does not touch the user's files)
                    Button {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            model.clearImages(status);
                            znccObj.clear(); // Old results reference the now-removed images, so drop them
                        }
                    } label: {
                        Label("Clear", systemImage: "xmark.circle")
                    }
                    .buttonStyle(FilledActionButtonStyle(fill: .brandSecondary))
                    .help("Remove all imported images from the app. Your original files are untouched")
                    .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                    .transition(.move(edge: .top).combined(with: .opacity));
                    // The hero action: run the duplicate analysis
                    Button {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            model.runZNCC(with: status, alg: znccObj);
                        }
                    } label: {
                        Label("Analyze", systemImage: "wand.and.rays")
                    }
                    .buttonStyle(FilledActionButtonStyle(fill: .brandAccent, foreground: .brandPrimary))
                    .help("Find duplicates: score every pair of images and group the near-identical ones")
                    .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                    .transition(.move(edge: .top).combined(with: .opacity));
                }
            }
            .padding(.horizontal, 20);
            AppStatus(model: self.status).padding(.vertical, 20);

            // Comparison-detail control. Before importing, the slider sets the canvas cap for the NEXT import;
            // once images exist it locks and shows the size the set is actually held at (a smaller image
            // joining can still shrink it, since every image must match size to compare). Clearing frees it
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Comparison Detail")
                        .font(.headline);
                    Spacer();
                    // The active size: the locked canvas once images exist, otherwise the pending cap
                    if model.containsImages(), let size = model.images.first?.currentSize() {
                        Text("\(size.width)×\(size.height) px")
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary);
                    } else {
                        Text("\(Int(maxCanvasDim)) px")
                            .font(.caption).monospacedDigit().foregroundStyle(.brandPrimary);
                    }
                }
                // Endpoint icons make the trade-off legible at a glance: speed/memory vs. detail
                Slider(value: $maxCanvasDim, in: 512...2048, step: 128) {
                    Text("Comparison detail");
                } minimumValueLabel: {
                    Image(systemName: "bolt.fill").foregroundStyle(.brandSecondary);
                } maximumValueLabel: {
                    Image(systemName: "sparkle.magnifyingglass").foregroundStyle(.brandSecondary);
                }
                .tint(.brandSecondary)
                .disabled(status.isBusy || model.containsImages())
                .help("Pixels per side each image is resampled to before comparing. Higher = finer matches but more memory");

                // A one-line explanation on a subtle amber background so it reads as a hint, not a control
                Text(model.containsImages()
                     ? "Locked while images are imported — Clear to change."
                     : "Lower is faster and lighter on memory; higher catches finer differences.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(Color.brandAccentLight.opacity(0.5), in: RoundedRectangle(cornerRadius: 8));
            }
            .padding(.horizontal, 20);
            // Developer/profiling tool, only shown when PerformanceBenchmark.isEnabled is flipped on
            // Runs the GPU-vs-CPU ZNCC benchmark and prints a report to the Xcode console, using the
            // imported photos (or synthetic images sized to the current Max Canvas Size if fewer than two)
            if PerformanceBenchmark.isEnabled {
                Button("Run GPU vs CPU Benchmark") {
                    let canvas = Int(self.maxCanvasDim);
                    // Prefer the user's imported photos; fall back to synthetic images only if fewer than two
                    // are imported, so the button always does something
                    let imported = model.images;
                    status.setPhase(to: .benchmarking);
                    status.setStatusMessage("Benchmarking...");
                    Task.detached {
                        let images = await imported.count >= 2 ? imported
                                                         : PerformanceBenchmark.syntheticImages(count: 20, canvas: canvas);
                        guard let ctx = MetalComputeContext(),
                              let result = await PerformanceBenchmark.run(images: images, context: ctx) else {
                            await MainActor.run {
                                status.setPhase(to: .error);
                                status.setStatusMessage("Benchmark failed");
                            }
                            return;
                        }
                        await print(result.report);
                        await MainActor.run {
                            status.setPhase(to: .finished);
                            status.setStatusMessage(result.summary);
                        }
                    }
                }
                .padding(.top, 8)
                .disabled(status.isBusy);
            }
            Spacer();
        }
        .padding()
        .frame(maxHeight: .infinity)
        // Glass as a BACKGROUND layer (not a parent of the content). This is the key to the slide-in fix: the
        // buttons above are siblings drawn over the glass, so inserting/removing them no longer animates
        // subviews *inside* the visual-effect view -- which is what caused the AppKit constraint loop
        .background {
            Color.clear.glassEffect(.regular, in: .rect(cornerRadius: 10));
        }
    }
}
