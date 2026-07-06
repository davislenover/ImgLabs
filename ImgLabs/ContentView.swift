//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI
import PhotosUI
import ImageIO // For reading image dimensions from file headers without decoding pixels

public let PROGRESS_END : CGFloat = 1.0;
public let PROGRESS_START : CGFloat = 0.0;

@Observable
class ZNCCModel {
    private var znccResults : [[Float]] = [];
    private let metalContext : MetalComputeContext? = MetalComputeContext();

    /// The most recent similarity matrix (empty until an analysis has run). Read-only to callers
    public var results : [[Float]] { self.znccResults; }

    /// Discards the current results (e.g. when the underlying images change so the matrix is stale)
    public func clear() {
        self.znccResults = [];
    }
    public func getZNCC(_ imgList : [ImageData]) async throws {
        guard let context = metalContext else {
            return;
        }
        let imgCorrelation = ImageCorrelation(MetalContext: context);
        znccResults = try await imgCorrelation.similarityMatrix(images: imgList);
    }
}

// Holds the state for the app's status indicator
// This is a reference type (@Observable class), NOT a View, so that mutations-
// made from anywhere (e.g. ImageModel) are seen by the View that renders it
@Observable
class AppStatusModel {
    enum Phase {
        case idle
        case importing
        case analyzing
        case clearing
        case browsing
        case copying
        case finished
        case error
    }
    var curPhase : Phase = .idle;
    var statusMessage : String = "Idle...";
    var progress : Double? = nil;
    var isProgressVisible : Bool = false;
    // Task to reset status to idle
    @ObservationIgnored private var resetTask : Task<Void,Never>?; // Task returns nothing and never throws an error

    /// True while an operation is in progress - used to disable controls & drive the spinner
    var isBusy : Bool {
        return self.curPhase == .importing || self.curPhase == .analyzing || self.curPhase == .clearing || self.curPhase == .browsing || self.curPhase == .copying;
    }

    func setPhase(to: Phase) {
        if to == .finished || to == .error { // If finished, then want to return status to idle after sometime
            self.resetTask?.cancel();
            // Start a new async Task
            self.resetTask = Task {
                do {
                    // Sleep for 3 seconds (3,000,000,000 nanoseconds)
                    try await Task.sleep(nanoseconds: 3 * 1_000_000_000);

                    // This runs on the Main Actor automatically to safely update UI
                    await MainActor.run {
                        withAnimation {
                            self.curPhase = .idle; // Reset back to idle
                            self.isProgressVisible = false;
                            self.progress = nil;
                        }
                    };
                } catch {
                    // The task was canceled before the 3 seconds finished (ignored)
                }
            }
        } else {
            self.resetTask?.cancel();
        }
        withAnimation {
            self.curPhase = to;
        }
    }

    func setStatusMessage(_ msg: String) {
        withAnimation {
            self.statusMessage = msg;
        }
    }

    func setProgress(_ progress: Double) {
        withAnimation {
            if (!self.isProgressVisible) {
                self.isProgressVisible = true;
            }
            self.progress = progress;
        }
    }
}

// Renders the status indicator described by an AppStatusModel
struct AppStatus : View {
    let model : AppStatusModel;
    private let iconSize : CGFloat = 20;

    /// Bouncing moon symbol - to indicate no operations are currently ongoing
    private var idleIndicator : some View {
        Image(systemName: "moon.zzz.fill")
            .font(.system(size: self.iconSize))
            .foregroundStyle(.blue)
            .symbolEffect(.bounce.up.byLayer, options: .nonRepeating)
    }
    
    /// Rotating loading symbol - to indicate operation is ongoing
    private var loadingIndicator : some View {
        Image(systemName: "clock.arrow.trianglehead.2.counterclockwise.rotate.90")
            .font(.system(size: self.iconSize))
            .symbolRenderingMode(.hierarchical) // Gives depth to the clock face vs arrows
            .foregroundStyle(.indigo)
            .symbolEffect(
                .rotate.byLayer,
                options: .repeat(.continuous), // Spin smoothly & continuously (not a single discrete turn)
                isActive: self.model.isBusy
            )
    }
    
    /// Bouncing checkmark - to indicate an operation has completed
    private var finishedIndicator : some View {
        Image(systemName: "checkmark.circle.fill")
            .font(.system(size: self.iconSize))
            .foregroundStyle(.green)
            .symbolEffect(.bounce.up.byLayer, options: .nonRepeating)
    }
    
    /// Bouncing error octagon - to indicate an operation failed
    private var errorIndicator : some View {
        Image(systemName: "exclamationmark.octagon.fill")
            .font(.system(size: self.iconSize))
            .foregroundStyle(.red)
            .symbolEffect(.bounce.up.byLayer, options: .nonRepeating)
    }
    
    @ViewBuilder
    public func getStatusView() -> some View {
        if self.model.isBusy {
            self.loadingIndicator;
        } else if self.model.curPhase == .finished {
            self.finishedIndicator;
        } else if self.model.curPhase == .error {
            self.errorIndicator;
        } else {
            self.idleIndicator;
        }
    }

    var body: some View {
        VStack {
            // Status - Icon and Text
            HStack {
                self.getStatusView();
                Text(self.model.statusMessage)
                    .font(.headline)
                    .fontWeight(.medium)
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .overlay(
                        Capsule().stroke(.white.opacity(0.2), lineWidth: 1)
                    )
            }
            // Add progress bar below status
            // Always a determinate bar: when there's no progress (idle / after finished) it sits at 0 as a
            // static, empty track instead of an animated indeterminate sweep. progress ?? 0 supplies the
            // fallback value when model.progress is nil
            ProgressView(value: self.model.progress ?? 0, total: PROGRESS_END)
            .progressViewStyle(.linear) // Forces the horizontal bar layout
            .tint(self.model.isProgressVisible ? .white : .gray)
            .animation(.default, value: self.model.progress)
            .scaleEffect(x: 1, y: 2, anchor: .center) // Make the bar slightly thicker
            .padding(.horizontal, 40)
        }
    }
}

@Observable
class ImageModel {
    private var imageList: [ImageData] = [];

    /// The imported images, in the same order the similarity matrix was built from. Read-only to callers
    public var images : [ImageData] { self.imageList; }

    func browseForImages(_ status : AppStatusModel) {
        status.setPhase(to: .browsing);
        status.setStatusMessage("Browsing for images...");
        let openPanel = NSOpenPanel()
        openPanel.title = "Choose an Image"
        openPanel.showsHiddenFiles = false
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        openPanel.allowsMultipleSelection = true
        openPanel.allowedContentTypes = [.image] // Filters for images

        // Open the native Finder sheet
        openPanel.begin { response in
            if response == .OK {
                let selectedURLs = openPanel.urls;
                let selectedCount = selectedURLs.count;
                // How many images were already imported before this run
                let previousCount = self.imageList.count;
                if (selectedCount == 0) {
                    status.setPhase(to: .idle);
                    status.setStatusMessage("Idle...");
                    return;
                }
                Task {
                    await MainActor.run {
                        status.setPhase(to: .importing);
                        status.setProgress(PROGRESS_START);
                        status.setStatusMessage("Analyzing image sizes...");
                    }

                    // Determine the common canvas size (min width/height across the whole set)
                    // Read the new selections' dimensions straight from their file headers - no pixel decode, so
                    // this is low-memory. Add in the already-imported images too (via their retained
                    // source size) so the canvas is consistent across accumulated imports
                    let existingImages = await MainActor.run { self.imageList };
                    var minWidth = Int.max;
                    var minHeight = Int.max;
                    for image in existingImages {
                        if let size = image.originalSize() {
                            minWidth = min(minWidth, size.width);
                            minHeight = min(minHeight, size.height);
                        }
                    }
                    for selectedURL in selectedURLs {
                        if let size = Self.imageDimensions(at: selectedURL) {
                            minWidth = min(minWidth, size.width);
                            minHeight = min(minHeight, size.height);
                        }
                    }

                    // Nothing yielded a readable size - finish without importing
                    guard minWidth != Int.max, minHeight != Int.max else {
                        await MainActor.run {
                            status.setPhase(to: .finished);
                            status.setProgress(PROGRESS_END);
                            status.setStatusMessage("No valid images to import");
                        }
                        return;
                    }
                    let canvasWidth = minWidth;
                    let canvasHeight = minHeight;

                    // If a smaller image just joined the set, re-resample the already-imported images down to the
                    // new common canvas so every image in the list stays the same length for the correlation
                    for image in existingImages {
                        if let current = image.currentSize(),
                           current.width != canvasWidth || current.height != canvasHeight {
                            image.resample(targetWidth: canvasWidth, targetHeight: canvasHeight);
                        }
                    }

                    await MainActor.run {
                        status.setStatusMessage("Importing images...");
                    }

                    // Decode + build each image ON THE MAIN ACTOR. CGImage and ImageData are not
                    // Sendable, so creating them on a background thread and then handing them to the main
                    // actor races their reference counts (an objc_release crash)
                    var importedCount = 0; // How many were successfully decoded & added this run
                    for (index, selectedURL) in selectedURLs.enumerated() {
                        let processed = index + 1;
                        await MainActor.run {
                            // CGImageSource (not NSImage) handles RAW (.ARW etc.) robustly and downsamples
                            // during decode, so we never fully decode a huge original just to shrink it.
                            if let cgImage = Self.decodedImage(at: selectedURL, maxPixelSize: max(canvasWidth, canvasHeight)) {
                                let decoded = ImageData(img: cgImage, targetWidth: canvasWidth, targetHeight: canvasHeight, filePath: selectedURL);
                                // Animate the append so the "Clear Imports" button's
                                // transition plays when the first image arrives
                                withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                                    self.imageList.append(decoded);
                                }
                                importedCount += 1;
                            }
                            status.setProgress(Double(processed) / Double(selectedCount));
                            status.setStatusMessage("Importing image \(processed) of \(selectedCount)");
                        }
                    }
                    await MainActor.run {
                        let totalCount = self.imageList.count;
                        let noun = importedCount == 1 ? "image" : "images";
                        status.setPhase(to: .finished);
                        status.setProgress(PROGRESS_END);
                        if previousCount > 0 {
                            // Appended to an existing set - report the delta and the running total
                            status.setStatusMessage("Imported \(importedCount) more \(noun), \(totalCount) total");
                        } else {
                            status.setStatusMessage("Importing complete, \(importedCount) \(noun) imported");
                        }
                    }
                }
            } else {
                // User clicked Cancel in the Finder panel - return to idle
                status.setPhase(to: .idle);
                status.setStatusMessage("Idle...");
            }
        }
    }

    /// Reads an image's pixel dimensions from its file header without decoding the pixel data
    /// - Parameter url: The image file to inspect
    /// - Returns: The width/height in pixels, or nil if the header couldn't be read
    private static func imageDimensions(at url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            return nil;
        }
        return (width, height);
    }

    /// Decodes an image file to a CGImage, downsampled so its largest dimension is at most maxPixelSize.
    /// Uses CGImageSource rather than NSImage: it handles RAW formats (.ARW, etc.) more reliably and
    /// generates a scaled-down image directly, avoiding a full-resolution decode of large originals.
    /// - Parameters:
    ///     - url: the image file to decode
    ///     - maxPixelSize: cap for the largest output dimension (aspect ratio preserved)
    /// - Returns: a decoded CGImage, or nil if the file couldn't be read
    private static func decodedImage(at url: URL, maxPixelSize: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil;
        }
        let options : [CFString: Any] = [
            // Prefer the file's embedded preview; only fall back to decoding the full image if there isn't one.
            // This avoids forcing a full RAW demosaic (kCGImageSourceCreateThumbnailFromImageAlways), which is
            // slow and can crash inside ImageIO's RAW decoder for some formats (e.g. .ARW). RAW files ship a
            // baked-in JPEG preview, which is more than enough for a downsampled similarity comparison
            kCGImageSourceCreateThumbnailFromImageIfAbsent: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,  // downsample so the largest side is at most this
            kCGImageSourceCreateThumbnailWithTransform: true    // apply EXIF orientation so pixels match how it displays
        ];
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary);
    }

    func clearImages(_ status : AppStatusModel) {
        // Show the "clearing" state, pause briefly so it's visible, then finish
        status.setPhase(to: .clearing);
        status.setStatusMessage("Clearing images...");
        status.setProgress(0);
        Task {
            // Pause ~0.5s so the clearing state is perceptible before completing
            try? await Task.sleep(nanoseconds: 1000 * 1_000_000);
            await MainActor.run {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                    self.imageList.removeAll();
                }
                status.setProgress(PROGRESS_END);
                status.setPhase(to: .finished);
                status.setStatusMessage("Cleared Images!");
            }
        }
    }
    
    func containsImages() -> Bool {
        return !self.imageList.isEmpty;
    }
    
    func runZNCC(with status: AppStatusModel, alg zncc: ZNCCModel) {
        status.setPhase(to: .analyzing);
        status.setStatusMessage("Checking for Image Similarity...");
        status.setProgress(0);
        Task {
            do {
                try await zncc.getZNCC(self.imageList);
            } catch {
                await MainActor.run {
                    status.setProgress(PROGRESS_END);
                    status.setPhase(to: .error);
                    status.setStatusMessage("Similarity Check Failed!");
                }
                return;
            }
            await MainActor.run {
                status.setProgress(PROGRESS_END);
                status.setPhase(to: .finished);
                status.setStatusMessage("Similarity Check Complete!");
            }
        }
    }
    
}

struct ControlSideBar : View {
    // Received from the parent (ContentView) so the grid pane can share the same state
    // These are @Observable classes, so reading their properties here observes them for changes
    let model: ImageModel;
    let status : AppStatusModel;
    let znccObj : ZNCCModel;
    // Drives the confirmation alert shown when importing over existing images
    @State private var showImportConfirm : Bool = false;

    /// Opens the Finder panel to import images (wrapped so both the button and alert can call it)
    private func startImport() {
        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) { // Animate in the Clear button
            model.browseForImages(status);
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
            Spacer();
        }
        .padding()
        .glassEffect(.regular, in: .rect(cornerRadius: 10)) // Use liquid glass material on the VStack behind the content
        .frame(maxHeight: .infinity)
    }
}

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
        !znccObj.results.isEmpty && znccObj.results.count == model.images.count;
    }

    var body : some View {
        VStack {
            if status.curPhase == .analyzing {
                // Something is being calculated
                self.stateIndicator(icon: "wand.and.rays", message: "Analyzing images...");
            } else if hasValidResults {
                DuplicateView(images: model.images, matrix: znccObj.results, status: status);
            } else {
                // Nothing to show yet
                self.stateIndicator(icon: "photo.on.rectangle.angled", message: "Import photos and press Analyze to find duplicates");
            }
        }
        // Fill the available space BEFORE the glass background, so the glass keeps the full pane size
        // regardless of how little content is inside it (otherwise it shrinks to fit the content)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
        .glassEffect(.regular, in: .rect(cornerRadius: 10))
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

// View is the fundamental building block of SwiftUI
// It is declarative, i.e., the expected result but supports imperative code
struct ContentView: View { // This a custom view, it conatains a body
    // The shared app state lives here (the common parent) so both panes can read it: the sidebar drives
    // imports/analysis, and the grid pane displays the results. @State keeps the instances alive across
    // re-renders; being @Observable classes, passing them to child views lets those views observe changes
    @State private var model : ImageModel = ImageModel();
    @State private var status : AppStatusModel = AppStatusModel();
    @State private var znccObj : ZNCCModel = ZNCCModel();

    var body: some View { // some means that body returns an object which conforms to the View type (i.e., don't need to specify exactly what is returned)
        HStack {
            ImageGridPane(model: self.model, status: self.status, znccObj: self.znccObj)
            ControlSideBar(model: self.model, status: self.status, znccObj: self.znccObj)
        }.glassEffect(.regular, in: RoundedRectangle(cornerRadius: 0, style: .continuous));
    }
}

#Preview {
    ContentView()
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
