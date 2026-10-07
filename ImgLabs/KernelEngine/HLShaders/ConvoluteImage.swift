//
//  ConvoluteImage.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-14.
//  Denotes a ComputeKernel which slides a 3x3 filter over every 2D matrix within a 3D strided matrix

import Foundation
import Metal

/// Convolves a 3x3 filter over every 2D matrix (set) within a 3D strided matrix, producing a same-sized 3D strided result
public actor ConvoluteImage: ComputeKernel {
    private nonisolated static let name : String = "convoluteImage";

    private let convolMtx : MTLBuffer;  // 3x3 filter (9 floats)
    private let inputMtx : MTLBuffer;   // depth 2D matricies laid back-to-back (each numRows*numColumns floats)
    private let numRows : UInt32;       // Row dimension of each 2D matrix
    private let numColumns : UInt32;    // Column dimension of each 2D matrix
    private let depth : UInt32;         // Number of 2D matricies within inputMtx
    private let convolResult : MTLBuffer; // Same size as inputMtx

    private let observerStore : ObserverStore = ObserverStore();

    public init(convolMtx: MTLBuffer, inputMtx: MTLBuffer, numRows: UInt32, numColumns: UInt32, depth: UInt32, resultBuf: MTLBuffer) async {
        self.convolMtx = convolMtx;
        self.inputMtx = inputMtx;
        self.numRows = numRows;
        self.numColumns = numColumns;
        self.depth = depth;
        self.convolResult = resultBuf;
    }

    nonisolated public static func getFunctionName() -> String {return Self.name;}

    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        let convol : MTLBuffer = await self.convolMtx;
        let input : MTLBuffer = await self.inputMtx;
        var numRows : UInt32 = self.numRows;
        var numColumns : UInt32 = self.numColumns;
        var depth : UInt32 = self.depth;
        let result : MTLBuffer = await self.convolResult;

        return ({ (encoder, pipelineState) in
            // Setup memory
            encoder.setBuffer(convol, offset: 0, index: 0);
            encoder.setBuffer(input, offset: 0, index: 1);
            encoder.setBytes(&numRows, length: MemoryLayout<UInt32>.stride, index: 2);
            encoder.setBytes(&numColumns, length: MemoryLayout<UInt32>.stride, index: 3);
            encoder.setBytes(&depth, length: MemoryLayout<UInt32>.stride, index: 4);
            encoder.setBuffer(result, offset: 0, index: 5);

            // Setup thread dispatch
            // One thread per pixel position in each image; the X axis packs all images (X*Z), Y axis is the image rows
            let gridExecutionSize = MTLSize(width: Int(numColumns) * Int(depth), height: Int(numRows), depth: 1);
            // Shape the 2D threadgroup around the SIMD width
            let groupWidth = pipelineState.threadExecutionWidth;
            let groupHeight = pipelineState.maxTotalThreadsPerThreadgroup / groupWidth;
            let threadsPerGroup = MTLSize(width: groupWidth, height: groupHeight, depth: 1);

            // Dispatch the grid setup (non-uniform threadgroups are handled by the kernel's bounds check)
            encoder.dispatchThreads(gridExecutionSize, threadsPerThreadgroup: threadsPerGroup);
        });
    }

    /// Observe for a DeviceBuffer -- the convolution output is published as a resident buffer
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }

    public func notifyObservers() async {
        // Publish the convolved output as a resident DeviceBuffer rather than downloading it to a [Float]. The
        // next stage reads straight from this buffer on the GPU, avoiding a GPU->CPU->GPU round-trip per stage
        await self.observerStore.callAll(with: DeviceBuffer(self.convolResult));
    }
}

nonisolated class ConvoluteImageFactory: ComputeKernelCreatable {
    // A 3x3 Laplacian filter -- pairs with Variance to produce a "variance of Laplacian" sharpness/blur score
    public static let laplacian3x3 : [Float] = [ 0,  1,  0,
                                                 1, -4,  1,
                                                 0,  1,  0 ];

    private let bufferCache : BufferCache = BufferCache();
    // The 3x3 filter to slide over each 2D matrix, wrapped so it can be cached like any other source buffer
    private let filterSource : ConvolutionMatrix;
    // The dimensions of each 2D matrix within the input
    private let numRows : UInt32;
    private let numColumns : UInt32;

    /// - Parameters:
    ///     - convolMtx: The 3x3 filter (9 floats, row-major) to slide over each 2D matrix
    ///     - numRows: The row dimension of each 2D matrix within the input
    ///     - numColumns: The column dimension of each 2D matrix within the input
    init(convolMtx: [Float] = ConvoluteImageFactory.laplacian3x3, numRows: UInt32, numColumns: UInt32) {
        self.filterSource = ConvolutionMatrix(convolMtx);
        self.numRows = numRows;
        self.numColumns = numColumns;
    }

    static func getFactoryName() -> String {
        return ConvoluteImage.getFunctionName();
    }

    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        let devToAlloc : MTLDevice = context.getDevice();

        // The image data is unique per call, so it is converted directly (not cached)
        guard let inputBuf : MTLBuffer = try? await bufable.toMTLBuffer(devToAlloc) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        guard let totalElements : UInt32 = try? bufable.MTLBufferSize() else {
            fatalError("Failed to get number of values");
        }
        // The input is a stack of equal-sized 2D matricies. The number of sets is the total divided by the size of each set
        let depth : UInt32 = totalElements / (self.numRows * self.numColumns);

        // Upload the filter once, then reuse the same buffer across every createKernel call
        let convolBuf : MTLBuffer = try await self.bufferCache.buffer(for: self.filterSource, device: devToAlloc);

        // The output mirrors the input dimensions, so it is never cached/shared
        guard let resultAlloc = devToAlloc.makeBuffer(length: Int(totalElements) * MemoryLayout<Float>.stride, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }

        return await ConvoluteImage(convolMtx: convolBuf, inputMtx: inputBuf, numRows: self.numRows, numColumns: self.numColumns, depth: depth, resultBuf: resultAlloc);
    }

    /// Wraps a 3x3 filter as an MTBufable so BufferCache can upload and reuse it like any other source buffer
    private nonisolated final class ConvolutionMatrix : MTBufable {
        private let matrix : [Float];

        init(_ matrix: [Float]) {
            self.matrix = matrix;
        }

        func toMTLBuffer(_ device: any MTLDevice) async throws -> any MTLBuffer {
            guard let newBuf : MTLBuffer = device.makeBuffer(bytes: self.matrix, length: self.matrix.count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                throw ImageError.failedToConvertDataToMTLBuffer;
            }
            return newBuf;
        }

        func MTLBufferSize() throws -> UInt32 {
            return UInt32(self.matrix.count);
        }
    }
}
