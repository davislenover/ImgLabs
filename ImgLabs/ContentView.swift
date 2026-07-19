//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI
import PhotosUI
import Photos
import ImageIO // For reading image dimensions from file headers without decoding pixels
import UniformTypeIdentifiers // For UTType.image when loading picked photo data

public let PROGRESS_END : CGFloat = 1.0;
public let PROGRESS_START : CGFloat = 0.0;

@Observable
class ZNCCModel {
    private var znccResults : [Float] = [];
    private var qualityResults : [ImageQuality] = [];
    private var hammingResults : [UInt8] = [];
    private let metalContext : MetalComputeContext? = MetalComputeContext();

    /// The most recent similarity matrix, strided (row-major, element (i, j) at i * imageCount + j). Empty
    /// until an analysis has run. Read-only to callers
    public var results : [Float] { self.znccResults; }

    /// Per-image quality signals (sharpness, resolution, file size, format), aligned to the analyzed image
    /// order. Empty until an analysis has run. Consumed by the keeper-selection strategy
    public var quality : [ImageQuality] { self.qualityResults; }

    /// All-pairs Hamming-distance matrix of the images' perceptual hashes, strided (row-major, element
    /// (i, j) at i * imageCount + j = differing bits between image i and j). Empty until an analysis has
    /// run. Consumed by DuplicateView to add near-duplicate edges to the ZNCC clustering
    public var hammingMatrix : [UInt8] { self.hammingResults; }

    /// Discards the current results (e.g. when the underlying images change so the results are stale)
    public func clear() {
        self.znccResults = [];
        self.qualityResults = [];
        self.hammingResults = [];
    }
    
    /// Runs the main analysis algorithm (generates all scores)
    public func getZNCC(_ imgList : [ImageData]) async throws {
        guard let context = metalContext else {
            return;
        }
        // One session over this image set. The grayscale batch is computed once and shared by the similarity
        // matrix and the sharpness scores; perceptual hashing runs its own 32x32 pipeline
        let session = ImageAnalysisSession(images: imgList, context: context);
        let matrix = try await session.similarityMatrix();
        let sharpness = try await session.sharpnessScores();
        let hashes = try await session.hashes();

        // Pull the plain values off the result actors, keeping input order, so the UI/model holds Sendable arrays
        var sharpnessValues : [Float] = [];
        for result in sharpness {
            await sharpnessValues.append(result.value());
        }
        // Build the all-pairs Hamming-distance matrix, off the hash actors, so the sync clustering can treat
        // it like the ZNCC matrix. Both are strided (row-major) [element (i, j) at i * count + j]
        let hammingMatrix : [UInt8] = await ImageHash.getHammingDistanceMtx(imagesHashes: hashes);
        // Combine the sharpness scores with each image's resolution/file metadata into the quality signals
        // the keeper strategy scores against
        let quality = await ImageQuality.build(images: imgList, sharpness: sharpnessValues);

        self.znccResults = matrix;
        self.qualityResults = quality;
        self.hammingResults = hammingMatrix;
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
        case benchmarking
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
        return self.curPhase == .importing || self.curPhase == .analyzing || self.curPhase == .clearing || self.curPhase == .browsing || self.curPhase == .copying || self.curPhase == .benchmarking;
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
class ImageModel: PHPickerViewControllerDelegate {
    private var configuration = PHPickerConfiguration(photoLibrary: .shared());
    private var imageList: [ImageData] = [];

    // The PhotoKit picker reports its results through the delegate callback, which receives no context of its
    // own. These are stashed when the picker is presented so the callback can resume the import with the same
    // status object and canvas cap the caller supplied. @ObservationIgnored: transient plumbing, not UI state
    @ObservationIgnored private var pendingStatus : AppStatusModel?;
    @ObservationIgnored private var pendingMaxCanvasDim : Int = 0;

    /// The imported images, in the same order the similarity matrix was built from. Read-only to callers
    public var images : [ImageData] { self.imageList; }
    
    
    /// Overloaded browseForImages, uses PhotoKit to allow for ingestion of photo library photos.
    /// Requests full read/write library access first: it's what lets picked assets be resolved and later
    /// deleted (moved to Recently Deleted). The privacy-preserving PHPicker itself needs no authorization,
    /// but resolving a pick's PHAsset for deletion does, so we require it up front
    func browseForImages(_ status : AppStatusModel, _maxCanvasDim: Int, _ photoFilter : PHPickerFilter) {
        status.setPhase(to: .browsing);
        status.setStatusMessage("Requesting photo access...");
        Task {
            // Prompts on first use (.notDetermined); returns the current decision otherwise
            let authStatus = await PHPhotoLibrary.requestAuthorization(for: .readWrite);
            await MainActor.run {
                guard authStatus == .authorized else {
                    // Limited access can't resolve arbitrary picks for deletion, so it's treated as insufficient here
                    status.setPhase(to: .error);
                    status.setStatusMessage(authStatus == .limited
                        ? "Full Photos access needed to delete. Enable it in System Settings > Privacy > Photos."
                        : "Photos access denied. Enable it in System Settings > Privacy > Photos.");
                    return;
                }
                self.presentPhotoPicker(status: status, maxCanvasDim: _maxCanvasDim, filter: photoFilter);
            }
        }
    }

    /// Configures and presents the PhotoKit picker as a sheet over the app window. Must run on the main actor
    /// (touches AppKit windows) and only after full library authorization has been granted
    @MainActor
    private func presentPhotoPicker(status: AppStatusModel, maxCanvasDim: Int, filter: PHPickerFilter) {
        // Stash the context the delegate callback needs, since it receives none of its own
        self.pendingStatus = status;
        self.pendingMaxCanvasDim = maxCanvasDim;
        status.setStatusMessage("Choosing photos...");

        // Set the filter type according to the user’s selection
        configuration.filter = filter;
        // Set the mode to avoid transcoding (hands back the asset's original encoding)
        configuration.preferredAssetRepresentationMode = .current;
        // Respect the user’s selection order
        configuration.selection = .ordered;
        // 0 == unlimited, enabling multiselection
        configuration.selectionLimit = 0;
        let picker = PHPickerViewController(configuration: configuration);
        // Assign the delegate to capture picked images
        picker.delegate = self;

        // Grab a window to attach the picker sheet to. Right after the permission prompt dismisses, the app's
        // window is briefly not key, so keyWindow is nil -- fall back to the main/any visible window instead of
        // failing (which is what forced a second button press the first time access was granted)
        guard let hostWindow = NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first(where: { $0.isVisible }),
              let contentViewController = hostWindow.contentViewController else {
            print("Error: Could not locate a valid macOS window container to attach the picker.");
            status.setPhase(to: .error);
            status.setStatusMessage("Couldn't present the photo picker");
            return;
        }

        // PHPickerViewController has no default content size on macOS, so the sheet opens tiny. Request a
        // generous size (~75% of the host window, clamped to a minimum)...
        let hostSize = hostWindow.frame.size;
        let targetSize = NSSize(width: max(720, hostSize.width * 0.75),
                                height: max(520, hostSize.height * 0.75));
        picker.preferredContentSize = targetSize;
        // Display the picker sheet smoothly over the application window
        contentViewController.presentAsSheet(picker);
        // ...and resize the sheet's window after presentation: the picker resets its size on present, so
        // preferredContentSize alone is ignored. Doing it on the next runloop tick, once the sheet window
        // exists, is what actually enlarges it
        DispatchQueue.main.async {
            picker.view.window?.setContentSize(targetSize);
        }
    }
    
    /// Delegate to capture resulting picked images. Mirrors the NSOpenPanel import pipeline: load the original
    /// bytes for every pick, resample the whole set onto a common canvas, then decode + build each ImageData
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        // Close the picker sheet now that the user has finished
        picker.dismiss(picker);

        // Recover the context stashed when the picker was presented
        let status = self.pendingStatus;
        let maxCanvasDim = self.pendingMaxCanvasDim;

        // User picked nothing (or cancelled) -- return to idle
        guard !results.isEmpty else {
            status?.setPhase(to: .idle);
            status?.setStatusMessage("Idle...");
            return;
        }

        // How many images were already imported before this run
        let previousCount = self.imageList.count;
        status?.setPhase(to: .importing);
        status?.setProgress(PROGRESS_START);
        status?.setStatusMessage("Loading photos...");

        // NSItemProvider vends the image bytes; assetIdentifier (present because the picker was configured with a
        // photoLibrary) resolves the backing PHAsset for deletion. Capture both so the Task never touches the
        // picker after it's been dismissed
        let picks = results.map { (provider: $0.itemProvider, assetID: $0.assetIdentifier) };

        Task {
            // First load the original encoded bytes for every pick. Data is Sendable and CGImage is not, so only the
            //    raw bytes are produced here (decoding into CGImage/ImageData happens on the main actor below)
            //    preferredAssetRepresentationMode = .current means these are the asset's original encodings
            //    (RAW/HEIC/JPEG as stored), matching what the file-picker path reads off disk
            var loaded : [(data: Data, name: String, assetID: String?)] = [];
            for (index, pick) in picks.enumerated() {
                await MainActor.run {
                    status?.setStatusMessage("Loading photo \(index + 1) of \(picks.count)...");
                }
                guard let data = try? await Self.loadImageData(from: pick.provider) else { continue; }
                // Prefer the asset's real original filename (correct extension -> accurate format rank);
                // fall back to the provider's suggested name, then a synthesized one
                let name = Self.originalFilename(forAssetIdentifier: pick.assetID)
                    ?? pick.provider.suggestedName
                    ?? "photo-\(UUID().uuidString)";
                loaded.append((data, name, pick.assetID));
            }

            // Nothing decodable came back -- finish without importing
            guard !loaded.isEmpty else {
                await MainActor.run {
                    status?.setPhase(to: .finished);
                    status?.setProgress(PROGRESS_END);
                    status?.setStatusMessage("No valid images to import");
                }
                return;
            }

            // Next, determine the common canvas size (min width/height across the whole set). Dimensions come from
            // the image headers. Include the already-imported images so the canvas is consistent across accumulated imports
            let existingImages = await MainActor.run { self.imageList };
            var minWidth = Int.max;
            var minHeight = Int.max;
            for image in existingImages {
                if let size = image.originalSize() {
                    minWidth = min(minWidth, size.width);
                    minHeight = min(minHeight, size.height);
                }
            }
            for item in loaded {
                if let size = Self.imageDimensions(from: item.data) {
                    minWidth = min(minWidth, size.width);
                    minHeight = min(minHeight, size.height);
                }
            }

            guard minWidth != Int.max, minHeight != Int.max else {
                await MainActor.run {
                    status?.setPhase(to: .finished);
                    status?.setProgress(PROGRESS_END);
                    status?.setStatusMessage("No valid images to import");
                }
                return;
            }
            // Cap the canvas
            let canvasWidth = min(minWidth, maxCanvasDim);
            let canvasHeight = min(minHeight, maxCanvasDim);

            // If a smaller image just joined the set, re-resample the already-imported images down to the new
            // common canvas so every image in the list stays the same length for the correlation
            for image in existingImages {
                if let current = image.currentSize(),
                   current.width != canvasWidth || current.height != canvasHeight {
                    image.resample(targetWidth: canvasWidth, targetHeight: canvasHeight);
                }
            }

            await MainActor.run { status?.setStatusMessage("Importing images..."); }

            // Finally, decode + build each image ON THE MAIN ACTOR. CGImage and ImageData are not Sendable so need to create on MainActor
            var importedCount = 0;
            for (index, item) in loaded.enumerated() {
                let processed = index + 1;
                await MainActor.run {
                    // CGImageSource downsamples during decode, so never fully decode a huge original just to shrink it
                    if let cgImage = Self.decodedImage(from: item.data, maxPixelSize: max(canvasWidth, canvasHeight)) {
                        // Library assets have no persistent file URL, synthesize one from the filename
                        let placeholderURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(item.name);
                        let decoded = ImageData(img: cgImage, targetWidth: canvasWidth, targetHeight: canvasHeight, filePath: placeholderURL, assetIdentifier: item.assetID);
                        // Animate the append so the "Clear Imports" button's transition plays when the first image arrives
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            self.imageList.append(decoded);
                        }
                        importedCount += 1;
                    }
                    status?.setProgress(Double(processed) / Double(loaded.count));
                    status?.setStatusMessage("Importing image \(processed) of \(loaded.count)");
                }
            }

            await MainActor.run {
                let totalCount = self.imageList.count;
                let noun = importedCount == 1 ? "image" : "images";
                status?.setPhase(to: .finished);
                status?.setProgress(PROGRESS_END);
                if previousCount > 0 {
                    // Appended to an existing set, report the delta and the running total
                    status?.setStatusMessage("Imported \(importedCount) more \(noun), \(totalCount) total");
                } else {
                    status?.setStatusMessage("Importing complete, \(importedCount) \(noun) imported");
                }
            }
        }
    }

    func browseForImages(_ status : AppStatusModel, _ maxCanvasDim : Int) {
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
                    // Cap the canvas so huge originals (e.g. RAW) don't blow up memory -- every image is
                    // decoded and stored at this size, so this is the main lever on memory usage.
                    let canvasWidth = min(minWidth, maxCanvasDim);
                    let canvasHeight = min(minHeight, maxCanvasDim);

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

    /// Async wrapper around NSItemProvider's data load. The completion-handler API returns a Progress (not
    /// Void), so Swift doesn't synthesize an async version; bridge it with a continuation. Yields the item's
    /// original encoded bytes for a type conforming to public.image
    /// - Parameter provider: the picked photo's item provider
    /// - Returns: the encoded image bytes
    private static func loadImageData(from provider: NSItemProvider) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                if let data {
                    continuation.resume(returning: data);
                } else {
                    continuation.resume(throwing: error ?? ImageError.noPixelData);
                }
            }
        }
    }

    /// The original filename of a library asset (e.g. "IMG_1234.HEIC"), resolved from its local identifier
    /// Requires full library authorization and will return nil if the asset can't be resolved (e.g. not accessible)
    /// - Parameter identifier: the PhotoKit local identifier from a PHPickerResult, or nil for file imports
    /// - Returns: the original filename, or nil
    private static func originalFilename(forAssetIdentifier identifier: String?) -> String? {
        guard let identifier else { return nil; }
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil);
        guard let asset = assets.firstObject else { return nil; }
        return PHAssetResource.assetResources(for: asset).first?.originalFilename;
    }

    /// Reads an image's pixel dimensions from in-memory encoded data without decoding the pixel data.
    /// The Data counterpart of `imageDimensions(at:)`, used for photos loaded via PhotoKit (no file URL)
    /// - Parameter data: the encoded image bytes to inspect
    /// - Returns: The width/height in pixels, or nil if the header couldn't be read
    private static func imageDimensions(from data: Data) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            return nil;
        }
        return (width, height);
    }

    /// Decodes in-memory encoded image data to a CGImage, downsampled so its largest dimension is at most
    /// maxPixelSize. The Data counterpart of `decodedImage(at:maxPixelSize:)` for PhotoKit picks. Same
    /// thumbnail options: prefer an embedded preview so RAW originals aren't fully demosaiced
    /// - Parameters:
    ///     - data: the encoded image bytes to decode
    ///     - maxPixelSize: cap for the largest output dimension (aspect ratio preserved)
    /// - Returns: a decoded CGImage, or nil if the data couldn't be read
    private static func decodedImage(from data: Data, maxPixelSize: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            return nil;
        }
        let options : [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageIfAbsent: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,  // downsample so the largest side is at most this
            kCGImageSourceCreateThumbnailWithTransform: true    // apply EXIF orientation so pixels match how it displays
        ];
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary);
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
            // Prefer the file's embedded preview, only fall back to decoding the full image if there isn't one
            kCGImageSourceCreateThumbnailFromImageIfAbsent: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,  // downsample so the largest side is at most this
            kCGImageSourceCreateThumbnailWithTransform: true    // apply EXIF orientation so pixels match how it displays
        ];
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary);
    }

    /// Deletes the given images from the photo library, sending them to Recently Deleted (a recoverable,
    /// non-destructive removal). Only images imported from the library (those carrying an assetIdentifier) can
    /// be deleted; file imports are ignored. The system shows its own confirmation prompt before removing
    /// anything, and on success the deleted images are dropped from imageList so results stay consistent
    /// - Parameters:
    ///     - images: the images to delete (typically a duplicate group's flagged members)
    ///     - status: shared status for progress/messaging
    /// - Returns: the number of assets actually moved to Recently Deleted
    @discardableResult
    func deleteFromLibrary(_ images: [ImageData], _ status : AppStatusModel) async -> Int {
        // Only library-origin images can be deleted through PhotoKit thus keep the ImageData alongside its id
        let deletable = images.filter { $0.getAssetIdentifier() != nil };
        let identifiers = deletable.compactMap { $0.getAssetIdentifier() };
        guard !identifiers.isEmpty else { return 0; }

        // Resolve the picked identifiers back to PHAssets (requires the full authorization requested at import)
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: identifiers, options: nil);
        guard assets.count > 0 else { return 0; }

        await MainActor.run {
            status.setPhase(to: .clearing);
            status.setStatusMessage("Deleting \(assets.count) photos...");
        }

        do {
            // performChanges triggers the system's delete confirmation which it throws if the user declines or it fails
            // deleteAssets routes to Recently Deleted rather than erasing immediately
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.deleteAssets(assets);
            }
            // Drop the now-deleted images from the working set so the grid/results don't reference gone photos
            let deletedIDs = Set(deletable.map { $0.id });
            await MainActor.run {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                    self.imageList.removeAll { deletedIDs.contains($0.id); }
                }
                status.setProgress(PROGRESS_END);
                status.setPhase(to: .finished);
                status.setStatusMessage("Moved \(assets.count) photos to Recently Deleted");
            }
            return assets.count;
        } catch {
            await MainActor.run {
                status.setPhase(to: .error);
                status.setStatusMessage("Delete cancelled or failed");
            }
            return 0;
        }
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
            // Developer/profiling tool -- only shown when PerformanceBenchmark.isEnabled is flipped on.
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
                DuplicateView(images: model.images, matrix: znccObj.results, quality: znccObj.quality, hammingMatrix: znccObj.hammingMatrix, status: status);
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
