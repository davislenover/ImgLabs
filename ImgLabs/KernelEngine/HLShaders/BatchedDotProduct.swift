//
//  BatchedDotProduct.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-06.
//  Defines a ComputeKernel which computes every pairwise dot product for a set of equal-length arrays in a
//  single dispatch (one threadgroup per pair), avoiding the per-pair kernel/buffer/observer overhead of
//  running DotProduct a number of times

import Metal

public actor BatchedDotProduct: ComputeKernel {
    private let observerStore : ObserverStore = ObserverStore();

    /// Observe for a [Float] type -- one dot product per pair, in the pair ordering the factory produced
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }

    public func notifyObservers() async {
        // Read every pair's result back as one flat array
        let rawPtr : UnsafeMutableRawPointer = self.results.contents();
        let count : Int = self.results.length / MemoryLayout<Float>.stride;
        let typedPointer : UnsafeMutablePointer<Float> = rawPtr.bindMemory(to: Float.self, capacity: count);
        let result : [Float] = [Float](UnsafeBufferPointer(start: typedPointer, count: count));
        await self.observerStore.callAll(with: result);
    }

    private nonisolated static let name : String = "batchedDotArr"; // Matches the Metal definition

    private let data : MTLBuffer;         // Every image's array concatenated back-to-back
    private let pairs : MTLBuffer;        // uint2 (i, j) image indices, one per pair/threadgroup
    private let results : MTLBuffer;      // One float per pair
    private let pairCount : Int;          // Number of pairs (== threadgroups to dispatch)
    private var elementsPerImage : UInt32;// Length of each image's array within the concatenated buffer

    init(data: MTLBuffer, pairs: MTLBuffer, elementsPerImage: UInt32, results: MTLBuffer, pairCount: Int) async {
        self.data = data;
        self.pairs = pairs;
        self.elementsPerImage = elementsPerImage;
        self.results = results;
        self.pairCount = pairCount;
    }

    nonisolated public static func getFunctionName() -> String {return Self.name;}

    /// Binds the concatenated data, the pair table and the result buffer, then dispatches one threadgroup per
    /// pair (each threadgroup reduces that pair's dot product)
    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        var elemsPerImg : UInt32 = await self.elementsPerImage;
        let dataBuf : MTLBuffer = await self.data;
        let pairsBuf : MTLBuffer = await self.pairs;
        let resultsBuf : MTLBuffer = await self.results;
        let groupCount : Int = self.pairCount;

        return ({ (encoder, pipelineState) in
            // Setup memory
            encoder.setBuffer(dataBuf, offset: 0, index: 0);
            encoder.setBuffer(pairsBuf, offset: 0, index: 1);
            encoder.setBytes(&elemsPerImg, length: MemoryLayout<UInt32>.stride, index: 2);
            encoder.setBuffer(resultsBuf, offset: 0, index: 3);

            // One float of shared memory per thread in the group for the reduction
            let threadsPerGroup = pipelineState.maxTotalThreadsPerThreadgroup;
            encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.stride * threadsPerGroup, index: 0);

            // Dispatch exactly one threadgroup per pair (NOT dispatchThreads) so each group maps to a pair via
            // threadgroup_position_in_grid
            let groupGrid = MTLSize(width: groupCount, height: 1, depth: 1);
            let groupThreads = MTLSize(width: threadsPerGroup, height: 1, depth: 1);
            encoder.dispatchThreadgroups(groupGrid, threadsPerThreadgroup: groupThreads);
        });
    }
}

class BatchedDotProductFactory: ComputeKernelCreatable {
    // The per-image arrays to correlate, in order -- image k occupies slot k of the strided data buffer
    // Every array must be the same length (the length is taken from the bufable passed to createKernel)
    private var imageArrays : [any MTBufable] = [];

    static func getFactoryName() -> String {
        return BatchedDotProduct.getFunctionName();
    }

    /// Sets the ordered per-image arrays to correlate. Must be called before createKernel; the factory
    /// assembles them into one strided device buffer (image k at offset k * elementsPerImage)
    public func setImageArrays(_ arrays: [any MTBufable]) {
        self.imageArrays = arrays;
    }

    /// The flat result index for the lower-triangle pair (row, col), where col <= row. The caller uses this to
    /// read a given pair's dot product out of the [Float] the kernel produces
    public static func pairIndex(row: Int, col: Int) -> Int {
        return row * (row + 1) / 2 + col;
    }

    /// The number of lower-triangle pairs (including the diagonal) for a given image count
    public static func pairCount(imageCount: Int) -> Int {
        return imageCount * (imageCount + 1) / 2;
    }

    /// Builds the strided data buffer (housing every array from setImageArrays back-to-back), the pair table
    /// and the result buffer, then returns the kernel. The kernel enumerates the lower triangle in row-major
    /// order, so pair (row, col) lands at pairIndex(row:col:) in the result
    /// - Parameter bufable: a representative image array -- its length sets each image's slot size in the
    ///   strided buffer (every array in setImageArrays must be this same length)
    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        let devToAlloc : MTLDevice = context.getDevice();
        let imageCount : Int = self.imageArrays.count;
        guard imageCount > 0, let elementsPerImage : UInt32 = try? bufable.MTLBufferSize() else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        // Allocate the strided buffer and fill it one image at a time, so at most a single per-image buffer is
        // resident alongside it (rather than uploading all N and concatenating afterwards)
        let stride : Int = MemoryLayout<Float>.stride;
        let perImageBytes : Int = Int(elementsPerImage) * stride;
        guard let dataBuf = devToAlloc.makeBuffer(length: imageCount * perImageBytes, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        let dataBase = dataBuf.contents();
        for (index, imageArray) in self.imageArrays.enumerated() {
            let imageBuf : MTLBuffer = try await imageArray.toMTLBuffer(devToAlloc);
            let copyBytes : Int = min(imageBuf.length, perImageBytes);
            dataBase.advanced(by: index * perImageBytes).copyMemory(from: imageBuf.contents(), byteCount: copyBytes);
        }

        // Enumerate the lower-triangle pairs (row-major, incl. diagonal) as the per-threadgroup work list
        var pairs : [SIMD2<UInt32>] = [];
        pairs.reserveCapacity(Self.pairCount(imageCount: imageCount));
        for row in 0..<imageCount {
            for col in 0...row {
                pairs.append(SIMD2<UInt32>(UInt32(row), UInt32(col)));
            }
        }

        // uint2 in Metal matches SIMD2<UInt32> (8 bytes) -- upload the pair table
        guard let pairsBuf = devToAlloc.makeBuffer(bytes: pairs, length: pairs.count * MemoryLayout<SIMD2<UInt32>>.stride, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        // One float of output per pair
        guard let resultsBuf = devToAlloc.makeBuffer(length: pairs.count * stride, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        return await BatchedDotProduct(data: dataBuf, pairs: pairsBuf, elementsPerImage: elementsPerImage, results: resultsBuf, pairCount: pairs.count);
    }
}
