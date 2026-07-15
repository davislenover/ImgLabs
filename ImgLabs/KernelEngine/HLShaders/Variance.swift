//
//  Variance.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-14.
//  Denotes a ComputeKernel for computing the population variance of each 2D matrix within a 3D strided matrix

import Foundation
import Metal

/// Computes the population variance of every 2D matrix (set) within a 3D strided matrix, yielding one variance per set
/// The final variance is computed on the CPU from the accumulated totals
public actor Variance: ComputeKernel {
    private nonisolated static let name : String = "calculateVariance";

    // Upper bound on how many threadgroups cooperate on a single set
    private nonisolated static let maxGroupsPerSlice : Int = 1024;

    private let inputMtx : MTLBuffer;   // depth 2D matricies laid back-to-back (each elemsPerSet floats)
    private let elemsPerSet : UInt32;   // Number of elements in each 2D matrix (numRows * numColumns)
    private let depth : UInt32;         // Number of 2D matricies within inputMtx
    private let accumBuffer : MTLBuffer; // (Σx, Σx²) per set -- 2 floats per set, folded into by the kernel (starts zeroed)

    private let observerStore : ObserverStore = ObserverStore();

    public init(inputMtx: MTLBuffer, elemsPerSet: UInt32, depth: UInt32, accumBuf: MTLBuffer) async {
        self.inputMtx = inputMtx;
        self.elemsPerSet = elemsPerSet;
        self.depth = depth;
        self.accumBuffer = accumBuf;
    }

    nonisolated public static func getFunctionName() -> String {return Self.name;}

    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        let input : MTLBuffer = await self.inputMtx;
        var elemsPerSet : UInt32 = self.elemsPerSet;
        var depth : UInt32 = self.depth;
        let accum : MTLBuffer = await self.accumBuffer;
        let maxGroupsPerSlice : Int = Self.maxGroupsPerSlice;

        return ({ (encoder, pipelineState) in
            // As many threads per group as the device allows (a power of two, keeps the tree reduction shallow)
            let groupSize = pipelineState.maxTotalThreadsPerThreadgroup;
            // Enough cooperating threadgroups to cover the set (one thread per element), capped so the count stays reasonable
            // The kernel's grid-stride loop keeps this correct even when the cap forces each thread to handle several elements
            let groupsNeeded = (Int(elemsPerSet) + groupSize - 1) / groupSize;
            var groupsPerSlice : UInt32 = UInt32(max(1, min(maxGroupsPerSlice, groupsNeeded)));

            // Setup memory
            encoder.setBuffer(input, offset: 0, index: 0);
            encoder.setBytes(&elemsPerSet, length: MemoryLayout<UInt32>.stride, index: 1);
            encoder.setBytes(&depth, length: MemoryLayout<UInt32>.stride, index: 2);
            encoder.setBytes(&groupsPerSlice, length: MemoryLayout<UInt32>.stride, index: 3);
            encoder.setBuffer(accum, offset: 0, index: 4);
            // Thread group shared memory: one float2 (Σx, Σx²) per thread
            encoder.setThreadgroupMemoryLength(MemoryLayout<SIMD2<Float>>.stride*groupSize, index: 0);

            // Setup thread dispatch
            // 2D grid of threadgroups: X selects the cooperating group within a set, Y selects the set
            let numThreadGroupsPerGrid = MTLSize(width: Int(groupsPerSlice), height: Int(depth), depth: 1);
            let threadsPerGroup = MTLSize(width: groupSize, height: 1, depth: 1);

            // Dispatch the grid setup
            encoder.dispatchThreadgroups(numThreadGroupsPerGrid, threadsPerThreadgroup: threadsPerGroup);
        });
    }

    /// Observe for a Float array (one variance per 2D matrix, indexed by set)
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }

    public func notifyObservers() async {
        // The kernel leaves (Σx, Σx²) per set in the accumulator; finish the variance on the CPU (one cheap division per set)
        // Copy the accumulator into a Sendable array so the parallel closure captures a value type instead of a raw pointer
        let rawPtr : UnsafeMutableRawPointer = self.accumBuffer.contents();
        let typedPointer : UnsafeMutablePointer<SIMD2<Float>> = rawPtr.bindMemory(to: SIMD2<Float>.self, capacity: Int(self.depth));
        let accumPairs : [SIMD2<Float>] = Array(UnsafeBufferPointer(start: typedPointer, count: Int(self.depth)));
        let n : Float = Float(self.elemsPerSet);

        var variances : [Float] = Array(repeating: 0.0, count: Int(self.depth));
        // Run the loop in parallel across available CPU cores
        DispatchQueue.concurrentPerform(iterations: accumPairs.count) { index in
            let sumX : Float = accumPairs[index].x;        // Σx
            let sumOfSquares : Float = accumPairs[index].y; // Σx²
            let mean : Float = sumX / n;
            let meanOfSquares : Float = sumOfSquares / n;
            variances[index] = meanOfSquares - (mean * mean); // variance = E[x²] - (E[x])²
        }
        await self.observerStore.callAll(with: variances);
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

        // The accumulator holds (Σx, Σx²) per set (2 floats each). Newly allocated Metal buffers start zeroed, which the
        // kernel relies on since every cooperating threadgroup atomically adds into these slots
        guard let accumAlloc = devToAlloc.makeBuffer(length: MemoryLayout<Float>.stride*2*Int(depth), options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        return await Variance(inputMtx: inputBuf, elemsPerSet: self.elemsPerSet, depth: depth, accumBuf: accumAlloc);
    }
}
