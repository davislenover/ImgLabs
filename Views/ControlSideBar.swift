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
    // Drives the confirmation alert shown when importing over existing images
    @State private var showImportConfirm : Bool = false;
    // Same, but for the photo-library (PhotoKit) import path
    @State private var showPhotoConfirm : Bool = false;

    // The comparison canvas is capped to this many pixels per side. Duplicate detection doesn't need full
    // resolution, and every image is decoded/resampled/compared at the canvas size so this bounds the
    // memory per image (canvas^2 * 4 bytes) regardless of how large the originals (e.g. RAW) are. Tune for
    // the quality/memory trade-off: larger = more detail, more memory
    @State private var maxCanvasDim : Double = 512;

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
            HStack {
                Button(action: {
                    // If images already exist, confirm first; otherwise import straight away
                    if model.containsImages() {
                        showImportConfirm = true;
                    } else {
                        startImport();
                    }
                }) {
                    // How the button looks
                    Text("Import Photos")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .cornerRadius(10)
                }
                .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                .alert("Import more images?", isPresented: $showImportConfirm) {
                    Button("Cancel", role: .cancel) { }
                    Button("Import") { startImport(); }
                } message: {
                    Text("You already have images imported. Newly selected images will be added to your current imports.");
                };
                // Import from the Photos library via PhotoKit (assets can later be deleted to Recently Deleted)
                Button(action: {
                    // If images already exist, confirm first; otherwise import straight away
                    if model.containsImages() {
                        showPhotoConfirm = true;
                    } else {
                        startPhotoImport();
                    }
                }) {
                    // How the button looks
                    Text("Photo Library")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .cornerRadius(10)
                }
                .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                .alert("Import more images?", isPresented: $showPhotoConfirm) {
                    Button("Cancel", role: .cancel) { }
                    Button("Import") { startPhotoImport(); }
                } message: {
                    Text("You already have images imported. Newly selected images will be added to your current imports.");
                };
                if model.containsImages() {
                    Button(action: {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            model.clearImages(status);
                            znccObj.clear(); // Old results reference the now-removed images, so drop them
                        }
                    }) {
                        // How the button looks
                        Text("Clear Imports")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .cornerRadius(10)
                    }
                    .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                    .transition(.move(edge: .trailing).combined(with: .opacity)); // Clear button moves in from right
                    // Analyze button
                    Button(action: {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            model.runZNCC(with: status, alg: znccObj);
                        }
                    }) {
                        // How the button looks
                        Text("Analyze")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .cornerRadius(10)
                    }
                    .disabled(status.isBusy) // Prevent re-triggering while an operation runs
                    .transition(.move(edge: .trailing).combined(with: .opacity)); // Clear button moves in from right
                }
            }.padding(.horizontal, 20);
            AppStatus(model: self.status).padding(.vertical, 20);
            Text("Options");
            // Before anything is imported the slider sets the canvas cap for the NEXT import. Once images
            // exist the canvas is fixed for the set: the slider locks, show the actual size the images
            // are held at. This keeps later imports consistent. Note a smaller
            // image joining can still shrink the canvas, since every image must stay the same size to compare
            // Clearing imports frees the slider again
            if model.containsImages() {
                if let size = model.images.first?.currentSize() {
                    Text("Canvas locked at \(size.width)×\(size.height) px — Clear imports to change");
                } else {
                    Text("Canvas locked — Clear imports to change");
                }
            } else {
                Text("Max Canvas Size (Higher -> Higher Memory Usage / More Accurate Results): \(Int(self.maxCanvasDim)) px");
            }
            Slider(value: $maxCanvasDim, in: 512...2048, step: 128)
                .disabled(status.isBusy || model.containsImages());
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
        .glassEffect(.regular, in: .rect(cornerRadius: 10)) // Use liquid glass material on the VStack behind the content
        .frame(maxHeight: .infinity)
    }
}
