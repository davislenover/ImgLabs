//
//  MeanValue.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-03.
//  Houses logic to run a kernel which computes the mean of all values within a 1D array

import Metal

class MeanValue: ComputeKernel {
    
    private nonisolated static let name : String = "calculateArraySum";
    
    private var values: MTLBuffer;
    private var maxLength: UInt32;
    private var sumResult: MTLBuffer;
    
    init(arrayToSum: MTLBuffer, numOfElements: UInt32, resultBuf: MTLBuffer) {
        self.values = arrayToSum;
        self.maxLength = numOfElements;
        self.sumResult = resultBuf;
    }

    nonisolated static func getFunctionName() -> String {return Self.name;}
    
    func encode() -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) throws -> () {
        return ({ (encoder, pipelineState) in
            // Setup memory
            encoder.setBytes(&self.maxLength, length: MemoryLayout.size(ofValue: self.maxLength), index: 1);
            encoder.setBuffer(self.values, offset: 0, index: 0);
            encoder.setBuffer(self.sumResult, offset: 0, index: 2);
            // Thread group requires the length of bytes be specified (the object isin't created on the CPU)
            // Specified for threadgroup(0) -- want one float per thread in a thread group of maximum thread size for the given device
            encoder.setThreadgroupMemoryLength(MemoryLayout.size(ofValue: Float.self)*pipelineState.maxTotalThreadsPerThreadgroup, index: 0);
            
            // Setup thread dispatch
            // Determine the total 1D grid size
            let totalThreads = Int(self.maxLength);
            let gridExecutionSize = MTLSize(width: totalThreads, height: 1, depth: 1);

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

class MeanValueFactory: ComputeKernelCreatable {
    static func getFactoryName() -> String {
        return MeanValue.getFunctionName();
    }
    
    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        // Get array values, convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();
        guard let valuesToSumBuf : MTLBuffer = try? await bufable.toMTLBuffer(devToAlloc) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        guard let numOfValues : UInt32 = try? bufable.MTLBufferSize() else {
            fatalError("Failed to get number of values");
        }
        // resultAlloc will be wrapped in a metal atomic float in the kernel
        guard let resultAlloc = devToAlloc.makeBuffer(length: MemoryLayout<Float>.size, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        return MeanValue(arrayToSum: valuesToSumBuf, numOfElements: numOfValues, resultBuf: resultAlloc);
    }
    
    
}

