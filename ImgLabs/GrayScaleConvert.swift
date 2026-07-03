//
//  GrayScaleConvert.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Kernel class definition which converts an image to it's grayscale representation

import Foundation
import Metal

class GrayScaleConvert : ComputeKernel {
    
    private let rgbaBuf : MTLBuffer; // UChar array holding RGBA values, single photo
    private let grayScaleBuf : MTLBuffer; // Resulting UChar array after conerting to grayscale
    private static let RED_WEIGHT : Float = 0.299;
    private static let GREEN_WEIGHT : Float = 0.587;
    private static let BLUE_WEIGHT : Float = 0.114;
    private static let rgbWeights : SIMD4<Float> = .init(RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT, 0.0); // 0.0 so ending doesn't affect calculation
    private static let backgroundColor : SIMD4<Float> = .init(0.0, 0.0, 0.0, 1.0);
    
    /// Creates an instance of GrayScaleConvert
    /// - Parameters:
    ///     - img: Contains a 1D array of values which can be allocated on GPU memory
    ///     - context: The MetalComputeContext object which houses the device object
    init(_ img : MTBufable, _ context : MetalComputeContext) async throws {
        // Get raw pixel values, convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();
        self.rgbaBuf = try await img.toMTLBuffer(devToAlloc);
        // Grayscale result should contain one element per pixel in rgbaBuf (which has 4 elements per pixel)
        guard let resultAlloc = devToAlloc.makeBuffer(length: self.rgbaBuf.length/4, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        self.grayScaleBuf = resultAlloc;
    }
    
    func getFunctionName() -> String {
        return "convertToGrayScale"; // Matches the Metal definition
    }
    
    func encode() -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) throws -> () {
        return ({ (encoder, pipelineState) in
            // Setup memory
            var rgbWeights : SIMD4<Float> = Self.rgbWeights;
            var bkgColor : SIMD4<Float> = Self.backgroundColor;
            // setBytes copies small memory directly to the encoded command (i.e., it's inlined thus being fast)
            encoder.setBytes(&rgbWeights, length: MemoryLayout.size(ofValue: rgbWeights), index: 2);
            encoder.setBytes(&bkgColor, length: MemoryLayout.size(ofValue: bkgColor), index: 1);
            encoder.setBuffer(self.rgbaBuf, offset: 0, index: 0);
            encoder.setBuffer(self.grayScaleBuf, offset: 0, index: 3);
            
            // Setup thread dispatch
            // Determine the total 1D grid size (total number of pixel threads to run)
            let totalPixelThreads = self.rgbaBuf.length / MemoryLayout<UInt8>.stride;
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

