//
//  ImageData.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-04.
//

import Foundation
import CoreGraphics

class ImageData : Identifiable { // Identifiable denotes to Swift that ImageData objects can be distinct from each other
    let id = UUID(); // Meant to disern different ImageData instances
    
    private static let NUM_OF_VALUES_IN_PIXEL: Int = 4;
    
    private var imageContext: CGContext?; // ? indicates an optional, i.e., this could be nil
    private var cgImage: CGImage?;
    private var pixelData: UnsafeMutablePointer<UInt8>?;
    
    // Constructor for class
    init(img : CGImage) async { // async indicates this function may be ran asyncronously (on a separate thread)
        // Extract raw pixel data
        self.cgImage = img;
        self.ingestImage(imgToIngest: img);
    }
    
    deinit {
        if let pixelData = self.pixelData { // Check for null
            pixelData.deallocate();
        }
    }
    
    public func getCGImage() -> CGImage? {
        return self.cgImage;
    }

    /// Extracts the raw pixel data from the image, stores it in pixelData and saves an imageContext with all other properties about the image
    /// - Parameter imgToIngest: An image of type CGImage
    /// - Returns: Void
    private func ingestImage(imgToIngest : CGImage) -> () {
        // Extract info about the image
        let width: Int = imgToIngest.width;
        let height: Int = imgToIngest.height;
        let bytesPerRow: Int = imgToIngest.bytesPerRow;
        let bitsPerComponent: Int = imgToIngest.bitsPerComponent; // Indicates how many bytes are used per pixel (typically 4, one for R, G, B, A channels)
        let bytesPerPixel: Int = bitsPerComponent / 8 * ImageData.NUM_OF_VALUES_IN_PIXEL;
        
        // Allocate space for bitmap array (each pixel is 0 to 255, thus 8-bits * number of channels needed)
        // This is allocated on the heap, and a pointer is returned
        self.pixelData = .allocate(capacity: width * height * bytesPerPixel); // Swift knows it's type and thus can use shorthand .
        
        // Draw the image into a CGContext object (as the image is likely compressed which this will decompress)
        self.imageContext = CGContext(
            data:self.pixelData, // Add raw pixel data here when image is drawn into this CGContext
            width:width,
            height:height,
            bitsPerComponent:bitsPerComponent,
            bytesPerRow:bytesPerRow,
            space:CGColorSpaceCreateDeviceRGB(),
            bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue // How to store the alpha info (doesn't matter what the type of the import image is, it will figure it out)
            // premultiplied basically means the RGB values have been multiplied by the alpha, last meants Alpha is stored in the last byte of the entire pixel data
        );
        self.imageContext?.draw(imgToIngest, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)));
        // bitMapData should now contain the raw RGBA values
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
