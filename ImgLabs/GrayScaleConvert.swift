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
            return;
        });
    }
    
    
}

