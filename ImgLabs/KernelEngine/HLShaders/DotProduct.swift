//
//  DotProduct.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-04.
//  Defines a ComputeKernel which calculates the dot product of two arrays

import Metal

public actor DotProduct: ComputeKernel {
    private let observerStore : ObserverStore = ObserverStore();
    
    /// Observe for a Float type
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }
    
    public func notifyObservers() async {
        let rawPtr : UnsafeMutableRawPointer = self.result.contents();
        let typedPointer : UnsafeMutablePointer<Float> = rawPtr.bindMemory(to: Float.self, capacity: 1); // Single float value, multiplied by stride automatically
        let result : Float = typedPointer.pointee;
        await self.observerStore.callAll(with: result);
    }
    
    private nonisolated static let name : String = "dotArr";
    
    private var arr1 : MTLBuffer;
    private var arr2 : MTLBuffer;
    private var arrLength : UInt32;
    private var result : MTLBuffer;
    
    init(inputArr1: MTLBuffer, inputArr2: MTLBuffer, numOfArrElements: UInt32, resultBuf: MTLBuffer) async {
        self.arr1 = inputArr1;
        self.arr2 = inputArr2;
        self.arrLength = numOfArrElements;
        self.result = resultBuf;
    }
    
    nonisolated public static func getFunctionName() -> String {return Self.name;}
    
    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        var len : UInt32 = await self.arrLength;
        let arr1Vals : MTLBuffer = await self.arr1;
        let arr2Vals : MTLBuffer = await self.arr2;
        let resultVal : MTLBuffer = await self.result;
        
        return ({ (encoder, pipelineState) in
            // Setup memory
            encoder.setBytes(&len, length: MemoryLayout<UInt32>.stride, index: 2);
            encoder.setBuffer(arr1Vals, offset: 0, index: 0);
            encoder.setBuffer(arr2Vals, offset: 0, index: 1);
            encoder.setBuffer(resultVal, offset: 0, index: 3);
            
            // Thread group requires the length of bytes be specified (the object isin't created on the CPU)
            // Specified for threadgroup(0) -- want one float per thread in a thread group of maximum thread size for the given device
            encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.stride*pipelineState.maxTotalThreadsPerThreadgroup, index: 0);
            
            // Setup thread dispatch
            // Determine the total 1D grid size
            let totalThreads = Int(len);
            let gridExecutionSize = MTLSize(width: totalThreads, height: 1, depth: 1);
            
            // Query the pipeline state to discover the absolute maximum allocation permitted by the GPU hardware
            let maxThreadsAllowed = pipelineState.maxTotalThreadsPerThreadgroup;
            let threadsPerGroup = MTLSize(width: maxThreadsAllowed, height: 1, depth: 1);
            
            // Dispatch the grid setup
            // Metal automatically breaks the grid into chunks of 'maxThreadsAllowed'
            encoder.dispatchThreads(gridExecutionSize, threadsPerThreadgroup: threadsPerGroup);
        });
    }
}

class DotProductFactory: ComputeKernelCreatable {
    private var arr2Bufable : (any MTBufable)?;

    // Shared cache of already-uploaded operand buffers, so an array reused across many pairs (as in an
    // all-pairs similarity matrix) is only uploaded to the device once instead of per kernel
    private let bufferCache : BufferCache;

    /// - Parameter bufferCache: The cache used to reuse operand buffers. Pass a shared instance to reuse
    ///   buffers across factories; the default gives this factory its own private cache.
    init(bufferCache: BufferCache = BufferCache()) {
        self.bufferCache = bufferCache;
    }

    static func getFactoryName() -> String {
        return DotProduct.getFunctionName();
    }

    public func setArr2(bufable: any MTBufable) {
        self.arr2Bufable = bufable;
    }

    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        // Get array values, convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();

        guard let arr2Bufable = self.arr2Bufable else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        // Reuse previously uploaded operand buffers where possible; both operands are read-only in the kernel
        let arr1Buf : MTLBuffer = try await self.bufferCache.buffer(for: bufable, device: devToAlloc);
        let arr2Buf : MTLBuffer = try await self.bufferCache.buffer(for: arr2Bufable, device: devToAlloc);

        guard let numOfValues : UInt32 = try? bufable.MTLBufferSize() else {
            fatalError("Failed to get number of values");
        }
        // resultAlloc will be wrapped in a metal atomic float in the kernel
        // The output is unique per kernel, so it is never cached/shared
        guard let resultAlloc = devToAlloc.makeBuffer(length: MemoryLayout<Float>.stride, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        return await DotProduct(inputArr1: arr1Buf, inputArr2: arr2Buf, numOfArrElements: numOfValues, resultBuf: resultAlloc);
    }
}
