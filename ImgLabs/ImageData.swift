//
//  ImageData.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-04.
//

import Foundation
import Metal
import CoreGraphics

public class ImageData : Identifiable, MTBufable { // Identifiable denotes to Swift that ImageData objects can be distinct from each other
    public let id = UUID(); // Meant to disern different ImageData instances
    
    private static let NUM_OF_VALUES_IN_PIXEL: Int = 4;
    
    private var imageContext: CGContext?; // ? indicates an optional, i.e., this could be nil
    private var cgImage: CGImage?;
    private var pixelData: UnsafeMutablePointer<UInt8>?;
    
    private var sourceURL: URL; // Stores the path of the original file
    
    // Constructor for class
    // targetWidth/targetHeight define the canvas the image is resampled onto, so every ImageData in a
    // set can be forced to a common size (required for the pixel-wise correlation to compare equal-length arrays)
    init(img : CGImage, targetWidth : Int, targetHeight : Int, filePath: URL) {
        // Extract raw pixel data
        self.cgImage = img;
        self.sourceURL = filePath;
        self.ingestImage(imgToIngest: img, targetWidth: targetWidth, targetHeight: targetHeight);
    }
    
    deinit {
        if let pixelData = self.pixelData { // Check for null
            pixelData.deallocate();
        }
    }
    
    /// Gets the file path of the image
    public func getURL() -> URL {return self.sourceURL;}
    
    /// Gets the CGImage representation of the ImageData object
    public func getCGImage() -> CGImage? {return self.cgImage;}

    /// The original pixel dimensions of the source image (before any resampling onto a canvas)
    public func originalSize() -> (width: Int, height: Int)? {
        guard let img = self.cgImage else { return nil; }
        return (img.width, img.height);
    }

    /// The canvas dimensions the image is currently resampled to (drives the buffer/array length)
    public func currentSize() -> (width: Int, height: Int)? {
        guard let ctx = self.imageContext else { return nil; }
        return (ctx.width, ctx.height);
    }

    /// Re-resamples the retained source image onto a new canvas size, replacing the current pixel data
    /// Used when a smaller image joins the set and every image must be brought to a common size
    public func resample(targetWidth : Int, targetHeight : Int) {
        guard let img = self.cgImage else { return; }
        // Free the previous buffer before ingestImage allocates a replacement
        self.pixelData?.deallocate();
        self.pixelData = nil;
        self.ingestImage(imgToIngest: img, targetWidth: targetWidth, targetHeight: targetHeight);
    }

    /// Extracts the raw pixel data from the image, resampling it onto a fixed canvas, stores it in pixelData and saves an imageContext with all other properties about the image
    /// - Parameters:
    ///     - imgToIngest: An image of type CGImage
    ///     - targetWidth: The canvas width to resample the image onto (source is scaled up or down to fit)
    ///     - targetHeight: The canvas height to resample the image onto (source is scaled up or down to fit)
    /// - Returns: Void
    private func ingestImage(imgToIngest : CGImage, targetWidth : Int, targetHeight : Int) -> () {
        // The canvas dimensions drive the buffer, so the image is scaled to these regardless of its own size
        // Always normalise to 8 bits per channel, NOT the source's bit depth. RAW files (e.g. Sony .ARW)
        // decode to 16-bit CGImages; a 16 bpc + premultipliedLast + DeviceRGB context is an unsupported
        // combination (CGContext creation fails), and the rest of the pipeline assumes 8-bit RGBA. Drawing
        // into an 8-bit context down-converts whatever the source format is into the buffer we expect.
        let bitsPerComponent: Int = 8;
        let bytesPerPixel: Int = bitsPerComponent / 8 * ImageData.NUM_OF_VALUES_IN_PIXEL; // = 4 (RGBA, one byte each)
        // Tightly pack the rows for the target width (no source row padding, since this is our own buffer)
        let bytesPerRow: Int = targetWidth * bytesPerPixel;

        // Allocate space for bitmap array (each pixel is 0 to 255, thus 8-bits * number of channels needed)
        // This is allocated on the heap, and a pointer is returned
        self.pixelData = .allocate(capacity: targetWidth * targetHeight * bytesPerPixel); // Swift knows it's type and thus can use shorthand .

        // Draw the image into a CGContext object (as the image is likely compressed which this will decompress)
        self.imageContext = CGContext(
            data:self.pixelData, // Add raw pixel data here when image is drawn into this CGContext
            width:targetWidth,
            height:targetHeight,
            bitsPerComponent:bitsPerComponent,
            bytesPerRow:bytesPerRow,
            space:CGColorSpaceCreateDeviceRGB(),
            bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue // How to store the alpha info (doesn't matter what the type of the import image is, it will figure it out)
            // premultiplied basically means the RGB values have been multiplied by the alpha, last meants Alpha is stored in the last byte of the entire pixel data
        );
        // Smooth interpolation when the source is scaled to the canvas (up or down)
        self.imageContext?.interpolationQuality = .high;
        // Drawing into a rect the size of the canvas resamples the source image to fit it exactly
        self.imageContext?.draw(imgToIngest, in: CGRect(x: 0, y: 0, width: CGFloat(targetWidth), height: CGFloat(targetHeight)));
        // bitMapData should now contain the raw RGBA values
    }
    
    /// Converts raw pixel data from the image ingested on creation to an MTLBuffer
    public func toMTLBuffer(_ device : MTLDevice) async throws -> MTLBuffer {
        guard let pixData = self.pixelData, let imgContext = self.imageContext else {
            throw ImageError.noPixelData;
        }
        // Multiply by the size of UInt8 in bytes (which should be 1 byte) as length specifies the number of bytes to copy
        guard let newBuf : MTLBuffer = device.makeBuffer(bytes: pixData, length: imgContext.width * imgContext.height * 4 * MemoryLayout<UInt8>.size, options: .storageModeShared) else {
            throw ImageError.failedToConvertDataToMTLBuffer;
        }
        return newBuf;
    }
    
    /// Returns the number of elements within the raw pixel data MTLBuffer array
    /// It's the number of pixels * 4 (4 channels per pixel) -- Each channel value s stored as a UInt8 type
    public func MTLBufferSize() throws -> UInt32 {
        guard let imgCtx = self.imageContext else {
            throw ImageError.noPixelData;
        }
        return UInt32(imgCtx.width * imgCtx.height * 4);
    }
    
    /// A copy of the raw premultiplied-RGBA bytes backing this image at its current canvas size
    /// (four bytes per pixel: R, G, B, A). Used by the CPU reference in PerformanceBenchmark to score the
    /// same pixels the GPU path uploads. Returns nil if the image has no pixel data
    public func rawRGBA() -> [UInt8]? {
        guard let pixData = self.pixelData, let ctx = self.imageContext else { return nil; }
        let count = ctx.width * ctx.height * ImageData.NUM_OF_VALUES_IN_PIXEL;
        return [UInt8](UnsafeBufferPointer(start: pixData, count: count));
    }

    public func printRawData(_ pixelX: UInt32, _ pixelY: UInt32) {
        // Safely unwrap the optional pointer
        guard let buffer = self.pixelData else { return; }
        // Multiply by 4 because 1 pixel = 4 bytes (RGBA)
        let bytesPerRow = self.imageContext!.width * 4;
        let pixelIndex = (Int(pixelY) * bytesPerRow) + (Int(pixelX) * 4);
        // Read directly from the raw pointer using array syntax
        let pixel: [UInt8] = [
            buffer[pixelIndex],     // Byte 0
            buffer[pixelIndex + 1], // Byte 1
            buffer[pixelIndex + 2], // Byte 2
            buffer[pixelIndex + 3]  // Byte 3
        ];
        print(pixel);
    }
    
}
