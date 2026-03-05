//
//  ImageData.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-04.
//

import CoreGraphics

class ImageData {
    
    private static let NUM_OF_VALUES_IN_PIXEL: Int = 4;
    
    private var imageContext: CGContext?; // ? indicates an optional, i.e., this could be nil
    private var pixelData: UnsafeMutablePointer<UInt8>?;
    
    // Constructor for class
    init(img : CGImage) {
        // Extract raw pixel data
        self.ingestImage(imgToIngest: img);
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
}
