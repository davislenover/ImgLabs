//
//  GrayScaleConvert.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Kernel class definition which converts an image to it's grayscale representation

import Foundation
import Metal

/// A class which defines to other classes a kernel on how to convert an image of type ImageData to a 1D array of values representing the grayscale image
/// Is thread-safe, after init no values change
class GrayScaleConvert : ComputeKernel, Sendable {
    
    private let rgbaBuf : MTLBuffer; // UChar array holding RGBA values, single photo
    private let grayScaleBuf : MTLBuffer; // Resulting UChar array after conerting to grayscale
    private static let RED_WEIGHT : Float = 0.299;
    private static let GREEN_WEIGHT : Float = 0.587;
    private static let BLUE_WEIGHT : Float = 0.114;
    private static let rgbWeights : SIMD4<Float> = .init(RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT, 0.0); // 0.0 so ending doesn't affect calculation
    private static let backgroundColor : SIMD4<Float> = .init(0.0, 0.0, 0.0, 1.0);
    private var numOfPixels : UInt32;
    
    /// Creates an instance of GrayScaleConvert
    /// - Parameters:
    ///     - img: Contains a 1D array of values which can be allocated on GPU memory
    ///     - context: The MetalComputeContext object which houses the device object
    init(_ img : MTBufable, _ context : MetalComputeContext) async throws {
        // Get raw pixel values, convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();
        self.rgbaBuf = try await img.toMTLBuffer(devToAlloc);
        // Grayscale result should contain one element per pixel in rgbaBuf (which has 4 elements per pixel)
        self.numOfPixels = try img.MTLBufferSize()/4;
        // Note that length argument is specified in bytes, the metal argument is float which one float is 4 bytes (i.e., each resulting pixel grayscale value is 4 bytes)
        // Thus, multiply number of pixels by the size of a Float
        guard let resultAlloc = devToAlloc.makeBuffer(length: Int(self.numOfPixels) * MemoryLayout<Float>.size, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        self.grayScaleBuf = resultAlloc;
    }
    
    func getFunctionName() -> String {
        return "convertToGrayScale"; // Matches the Metal definition
    }
    
    /// Sets up conversion to grayscale kernel. Encodes internal weights, a black background color and the ingested image raw pixel values along with the out result
    /// One thread per pixel, will dispatch the maximum width allowed per thread group
    func encode() -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) throws -> () {
        return ({ (encoder, pipelineState) in
            // Setup memory
            var rgbWeights : SIMD4<Float> = Self.rgbWeights;
            var bkgColor : SIMD4<Float> = Self.backgroundColor;
            var numOfPixels : UInt32 = self.numOfPixels;
            // setBytes copies small memory directly to the encoded command (i.e., it's inlined thus being fast)
            encoder.setBytes(&rgbWeights, length: MemoryLayout.size(ofValue: rgbWeights), index: 2);
            encoder.setBytes(&bkgColor, length: MemoryLayout.size(ofValue: bkgColor), index: 1);
            encoder.setBytes(&numOfPixels, length: MemoryLayout.size(ofValue: numOfPixels), index: 4);
            encoder.setBuffer(self.rgbaBuf, offset: 0, index: 0);
            encoder.setBuffer(self.grayScaleBuf, offset: 0, index: 3);
            
            // Setup thread dispatch
            // Determine the total 1D grid size (total number of pixel threads to run)
            let totalPixelThreads = Int(self.numOfPixels);
            let gridExecutionSize = MTLSize(width: totalPixelThreads, height: 1, depth: 1);

            // Query the pipeline state to discover the absolute maximum allocation permitted by the GPU hardware
            // This safely maximizes the threads inside the group (commonly returns 512 or 1024)
            let maxThreadsAllowed = pipelineState.maxTotalThreadsPerThreadgroup;
            let threadsPerGroup = MTLSize(width: maxThreadsAllowed, height: 1, depth: 1);

            // Dispatch the grid setup
            // Metal automatically breaks the grid into chunks of 'maxThreadsAllowed'
            encoder.dispatchThreads(gridExecutionSize, threadsPerThreadgroup: threadsPerGroup);
        });
    }
    
}

