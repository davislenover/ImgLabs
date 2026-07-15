//
//  Variance.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-14.
//  Denotes a ComputeKernel for computing the population variance of each 2D matrix within a 3D strided matrix

import Foundation
import Metal

/// Computes the population variance of every 2D matrix (set) within a 3D strided matrix, yielding one variance per set
public actor Variance: ComputeKernel {
    private nonisolated static let name : String = "calculateVariance";

    private let inputMtx : MTLBuffer;   // depth 2D matricies laid back-to-back (each elemsPerSet floats)
    private let elemsPerSet : UInt32;   // Number of elements in each 2D matrix (numRows * numColumns)
    private let depth : UInt32;         // Number of 2D matricies within inputMtx
    private let varianceResult : MTLBuffer; // One variance per set (depth floats)

    private let observerStore : ObserverStore = ObserverStore();

    public init(inputMtx: MTLBuffer, elemsPerSet: UInt32, depth: UInt32, resultBuf: MTLBuffer) async {
        self.inputMtx = inputMtx;
        self.elemsPerSet = elemsPerSet;
        self.depth = depth;
        self.varianceResult = resultBuf;
    }

    nonisolated public static func getFunctionName() -> String {return Self.name;}

    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        let input : MTLBuffer = await self.inputMtx;
        var elemsPerSet : UInt32 = self.elemsPerSet;
        var depth : UInt32 = self.depth;
        let result : MTLBuffer = await self.varianceResult;

        return ({ (encoder, pipelineState) in
            // Setup memory
            encoder.setBuffer(input, offset: 0, index: 0);
            encoder.setBytes(&elemsPerSet, length: MemoryLayout<UInt32>.stride, index: 1);
            encoder.setBytes(&depth, length: MemoryLayout<UInt32>.stride, index: 2);
            encoder.setBuffer(result, offset: 0, index: 3);
            // Thread group shared memory: one float2 (Σx, Σx²) per thread, sized to the maximum threadgroup for the device
            encoder.setThreadgroupMemoryLength(MemoryLayout<SIMD2<Float>>.stride*pipelineState.maxTotalThreadsPerThreadgroup, index: 0);

            // Setup thread dispatch
            // One threadgroup per set (each group reduces one 2D matrix into a single variance)
            let numThreadGroupsPerGrid = MTLSize(width: Int(depth), height: 1, depth: 1);
            // As many threads per group as the device allows (a power of two, keeps the tree reduction shallow)
            let maxThreadsAllowed = pipelineState.maxTotalThreadsPerThreadgroup;
            let threadsPerGroup = MTLSize(width: maxThreadsAllowed, height: 1, depth: 1);

            // Dispatch the grid setup
            encoder.dispatchThreadgroups(numThreadGroupsPerGrid, threadsPerThreadgroup: threadsPerGroup);
        });
    }

    /// Observe for a Float array (one variance per 2D matrix, indexed by set)
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }

    public func notifyObservers() async {
        let rawPtr : UnsafeMutableRawPointer = self.varianceResult.contents();
        let count : Int = self.varianceResult.length / MemoryLayout<Float>.stride;
        let typedPointer : UnsafeMutablePointer<Float> = rawPtr.bindMemory(to: Float.self, capacity: count);
        let result : [Float] = [Float](UnsafeBufferPointer(start: typedPointer, count: count));
        await self.observerStore.callAll(with: result);
    }
}

nonisolated class VarianceFactory: ComputeKernelCreatable {

    private let bufferCache : BufferCache;
    // The number of elements in each 2D matrix (numRows * numColumns)
    private let elemsPerSet : UInt32;

    /// - Parameters:
    ///     - elemsPerSet: The number of elements in each 2D matrix within the input (i.e., numRows * numColumns)
    ///     - bufferCache: Pass a shared instance to reuse buffers across factories; the default gives this factory its own private cache
    init(elemsPerSet: UInt32, bufferCache: BufferCache = BufferCache()) {
        self.elemsPerSet = elemsPerSet;
        self.bufferCache = bufferCache;
    }

    static func getFactoryName() -> String {
        return Variance.getFunctionName();
    }

    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        // Get the input matrix values, convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();
        let inputBuf : MTLBuffer = try await self.bufferCache.buffer(for: bufable, device: devToAlloc);

        guard let totalElements : UInt32 = try? bufable.MTLBufferSize() else {
            fatalError("Failed to get number of values");
        }
        // The input is a stack of equal-sized 2D matricies. The number of sets is the total divided by the size of each set
        let depth : UInt32 = totalElements / self.elemsPerSet;

        // The output is unique per kernel
        guard let resultAlloc = devToAlloc.makeBuffer(length: MemoryLayout<Float>.stride*Int(depth), options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        return await Variance(inputMtx: inputBuf, elemsPerSet: self.elemsPerSet, depth: depth, resultBuf: resultAlloc);
    }
}
