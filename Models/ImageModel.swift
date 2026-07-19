//
//  ImageModel.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  Owns the imported image set and the import (file + PhotoKit) and delete pipelines

import SwiftUI
import PhotosUI
import Photos
import ImageIO // For reading image dimensions from file headers without decoding pixels
import UniformTypeIdentifiers // For UTType.image when loading picked photo data

@Observable
class ImageModel: PHPickerViewControllerDelegate {
    private var configuration = PHPickerConfiguration(photoLibrary: .shared());
    private var imageList: [ImageData] = [];

    // The PhotoKit picker reports its results through the delegate callback, which receives no context of its
    // own. These are stashed when the picker is presented so the callback can resume the import with the same
    // status object and canvas cap the caller supplied. @ObservationIgnored is for indicating no UI state
    @ObservationIgnored private var pendingStatus : AppStatusModel?;
    @ObservationIgnored private var pendingMaxCanvasDim : Int = 0;

    /// The imported images, in the same order the similarity matrix was built from. Read-only to callers
    public var images : [ImageData] { self.imageList; }

    // MARK: - PhotoKit import

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
        // window is briefly not key, so keyWindow is nil, fall back to the main/any visible window instead of
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

        // User picked nothing (or cancelled), return to idle
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

    // MARK: - File import

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
                    // Read the new selections' dimensions straight from their file headers, no pixel decode, so
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

                    // Nothing yielded a readable size, finish without importing
                    guard minWidth != Int.max, minHeight != Int.max else {
                        await MainActor.run {
                            status.setPhase(to: .finished);
                            status.setProgress(PROGRESS_END);
                            status.setStatusMessage("No valid images to import");
                        }
                        return;
                    }
                    // Cap the canvas
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

                    // Decode + build each image ON THE MAIN ACTOR
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
                            // Appended to an existing set, report the delta and the running total
                            status.setStatusMessage("Imported \(importedCount) more \(noun), \(totalCount) total");
                        } else {
                            status.setStatusMessage("Importing complete, \(importedCount) \(noun) imported");
                        }
                    }
                }
            } else {
                // User clicked Cancel in the Finder panel, return to idle
                status.setPhase(to: .idle);
                status.setStatusMessage("Idle...");
            }
        }
    }

    // MARK: - Delete / clear

    /// Deletes the given images from their source: photo-library imports go to Recently Deleted (PhotoKit
    /// shows its own confirmation), file imports go to the Trash. Both are recoverable. On success the deleted
    /// images are dropped from imageList so results stay consistent
    /// - Parameters:
    ///     - images: the images to delete (typically a duplicate group's flagged members)
    ///     - status: shared status for progress/messaging
    /// - Returns: the number of images actually removed
    @discardableResult
    func delete(_ images: [ImageData], _ status : AppStatusModel) async -> Int {
        guard !images.isEmpty else { return 0; }

        // Partition by origin: library assets are deleted via PhotoKit, file imports are trashed
        let libraryImages = images.filter { $0.getAssetIdentifier() != nil };
        let fileImages = images.filter { $0.getAssetIdentifier() == nil };

        status.setPhase(to: .clearing);
        status.setStatusMessage("Deleting \(images.count) images...");

        var deletedIDs = Set<UUID>();

        // Photo-library assets -> Recently Deleted
        let identifiers = libraryImages.compactMap { $0.getAssetIdentifier() };
        if await MediaRemover.deleteLibraryAssets(identifiers: identifiers) {
            for image in libraryImages { deletedIDs.insert(image.id); }
        }

        // File imports -> Trash. Run the blocking file I/O off the main actor
        let fileURLs = fileImages.map { $0.getURL() };
        let trashed = await Task.detached { MediaRemover.trashFiles(fileURLs) }.value;
        for image in fileImages where trashed.contains(image.getURL()) { deletedIDs.insert(image.id); }

        // Drop everything actually removed from the working set so the grid/results don't reference gone images
        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
            self.imageList.removeAll { deletedIDs.contains($0.id); }
        }
        status.setProgress(PROGRESS_END);
        if deletedIDs.isEmpty {
            status.setPhase(to: .error);
            status.setStatusMessage("Delete cancelled or failed");
        } else {
            status.setPhase(to: .finished);
            status.setStatusMessage("Deleted \(deletedIDs.count) images");
        }
        return deletedIDs.count;
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

    // MARK: - Decoding helpers

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
    /// Void), so Swift doesn't synthesize an async version, bridge it with a continuation. Yields the item's
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
}
