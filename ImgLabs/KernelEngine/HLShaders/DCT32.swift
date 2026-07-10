//
//  DCT32.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-10.
//  Denotes a ComputeKernel for finding the DCT of 32x32 images


import Foundation
import Metal

public actor DCT32: ComputeKernel {
    private nonisolated static let name : String = "convertDCT";
    
    private let images : MTLBuffer; // 32x32xnumOfImages
    private let resultDCTs : MTLBuffer; // 8x8xnumOfImages -- 64 floats per z-axis (resulting DCT image)
    private let numOfImages : UInt32;
    private let preConstMtx : MTLBuffer; // 8x32 precomputed DCT basis (buffer 0 in the shader)

    private let observerStore : ObserverStore = ObserverStore();

    public init(valuesArr: MTLBuffer, numOfImgs: UInt32, resultBuf: MTLBuffer, preConst: MTLBuffer) async {
        self.images = valuesArr;
        self.numOfImages = numOfImgs;
        self.resultDCTs = resultBuf;
        self.preConstMtx = preConst;
    }
    
    nonisolated public static func getFunctionName() -> String {
        return Self.name;
    }
    
    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        let values : MTLBuffer = await self.images;
        let preConst : MTLBuffer = await self.preConstMtx;
        var num : UInt32 = self.numOfImages;
        var numRowsAndColumns : UInt32 = 32;
        var maxFreq : UInt32 = 8;
        let result : MTLBuffer = await self.resultDCTs;
        return { encoder, pipelineState in
            // Setup values
            /*
             kernel void convertDCT(constant float* preConstMtx [[buffer(0)]],
                                    device const float* inputMtx [[buffer(1)]],
                                    constant uint32_t& numRowsAndColumns [[buffer(2)]],
                                    constant uint32_t& depth [[buffer(3)]],
                                    constant uint32_t& maxFreq [[buffer(4)]],
                                    device float* result [[buffer(5)]],
                                    uint2 groupId [[threadgroup_position_in_grid]],
                                    uint2 localId [[thread_position_in_threadgroup]]) {
             */
            encoder.setBuffer(preConst, offset: 0, index: 0);
            encoder.setBuffer(values, offset: 0, index: 1);
            encoder.setBytes(&numRowsAndColumns, length: MemoryLayout<UInt32>.stride, index: 2);
            encoder.setBytes(&num, length: MemoryLayout<UInt32>.stride, index: 3);
            encoder.setBytes(&maxFreq, length: MemoryLayout<UInt32>.stride, index: 4);
            encoder.setBuffer(result, offset: 0, index: 5);
            
            // Setup thread dispatch
            // One threadgroup per image thus num threadgroups per grid
            let numThreadGroupsPerGrid = MTLSize(width: Int(num), height: 1, depth: 1);
            // 32x8 threads per threadgroup (1 image per threadgroup)
            let threadsPerGroup = MTLSize(width: Int(numRowsAndColumns), height: Int(maxFreq), depth: 1);

            // Dispatch the grid setup
            encoder.dispatchThreadgroups(numThreadGroupsPerGrid,threadsPerThreadgroup: threadsPerGroup);
        }
    }
    
    /// Observe for a DeviceBuffer of type [Float]
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }

    public func notifyObservers() async {
        await self.observerStore.callAll(with: DeviceBuffer(self.resultDCTs));
    }
}


class DCT32Factory: ComputeKernelCreatable {
    
    // preConstMtx[u][x] = α(u) · cos((2x+1)·u·π / 2N)
    // N = 32 (total number of pixels per row)
    // u is the current freqeuncy wave (from 0-7)
    // a(u) = sqrt(1/N) if a(0) / sqrt(2/N) if a(i) -> i>0
    // This is computed on creation of the factory (/on CPU because it's small and GPU overhead would be wasteful)
    
    // Shared cache of preConstMtx (computed once then reused across every createKernel call)
    private let bufferCache : BufferCache = BufferCache();
    private let preConstSource : PreConstMtx = PreConstMtx();

    static func getFactoryName() -> String {
        return DCT32.getFunctionName();
    }

    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        // Get 32x32 image values (assumed to be 32x32), convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();

        // The image data is unique per call, so it is converted directly (not cached)
        guard let imagesBuf : MTLBuffer = try? await bufable.toMTLBuffer(devToAlloc) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        guard let numOfImages : UInt32 = try? bufable.MTLBufferSize() else {
            fatalError("Failed to get number of Images");
        }

        // The output is unique per kernel, so it is never cached/shared -- 64 floats (8x8) per image
        guard let resultAlloc = devToAlloc.makeBuffer(length: MemoryLayout<Float>.stride*64*Int(numOfImages), options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        // Pre-calculate preConstMtx if needed (the cache computes it once, then returns the same buffer)
        let preConstMtxBuf : MTLBuffer = try await bufferCache.buffer(for: self.preConstSource, device: devToAlloc);

        return await DCT32(valuesArr: imagesBuf, numOfImgs: numOfImages, resultBuf: resultAlloc, preConst: preConstMtxBuf);
    }
    
    private class PreConstMtx : MTBufable {
        func toMTLBuffer(_ device: any MTLDevice) async throws -> any MTLBuffer {
            // Precompute preConstMtx (8x32 matrix)
            var preConstMtx : [Float] = .init(repeating: 0, count: 256);
            for u in 0...7 {
                for x in 0...31 {
                    let val : Float = Self.a(u) * cosf((Float.pi * (2*Float(x) + 1) * Float(u)) / 64);
                    preConstMtx[(u*32) + x] = val;
                }
            }

            guard let newBuf : MTLBuffer = device.makeBuffer(bytes: preConstMtx, length: preConstMtx.count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                throw ImageError.failedToConvertDataToMTLBuffer;
            }
            return newBuf;
        }
        
        func MTLBufferSize() throws -> UInt32 {
            return 256;
        }
        
        private static func a(_ u: Int) -> Float {
            // Standard DCT-II normalisation: α(0) = sqrt(1/N), α(u>0) = sqrt(2/N), N = 32.
            // Use Float division -- 1/32 and 2/32 as Ints truncate to 0 (so sqrt would return 0).
            let n : Float = 32;
            return u == 0 ? (1 / n).squareRoot() : (2 / n).squareRoot();
        }
        
    }
}


