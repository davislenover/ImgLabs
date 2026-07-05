//
//  Subtraction.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-04.
//  A compute kernel which subtracts a value from all elements

import Foundation
import Metal

public actor Subtraction: ComputeKernel {
    private nonisolated static let name : String = "calculateSubtractionPow";
    
    private let valueBuf : MTLBuffer;
    private let valueLength : UInt32;
    private let floatToSubtract : Float;
    private let powVal: Float;
    private let result : MTLBuffer;
    
    private let observerStore : ObserverStore = ObserverStore();
    
    public init(valuesArr: MTLBuffer, numOfValues: UInt32, valueToSubtract: Float, powerVal: Float, resultBuf: MTLBuffer) async {
        self.valueBuf = valuesArr;
        self.valueLength = numOfValues;
        self.floatToSubtract = valueToSubtract;
        self.result = resultBuf;
        self.powVal = powerVal;
    }
    
    nonisolated public static func getFunctionName() -> String {
        return Self.name;
    }
    
    nonisolated public func encode() async -> (any MTLComputeCommandEncoder, any MTLComputePipelineState) -> () {
        let values : MTLBuffer = await self.valueBuf;
        var length : UInt32 = self.valueLength;
        var sub : Float = self.floatToSubtract;
        let result : MTLBuffer = await self.result;
        var pwVal : Float = self.powVal;
        return { encoder, pipelineState in
            // Setup values
            encoder.setBuffer(values, offset: 0, index: 0);
            encoder.setBytes(&length, length: MemoryLayout<UInt32>.stride, index: 1);
            encoder.setBytes(&sub, length: MemoryLayout<Float>.stride, index: 2);
            encoder.setBytes(&pwVal, length: MemoryLayout<Float>.stride, index: 3);
            encoder.setBuffer(result, offset: 0, index: 4);
            
            // Setup thread dispatch
            // Determine the total 1D grid size
            let totalThreads = Int(length);
            let gridExecutionSize = MTLSize(width: totalThreads, height: 1, depth: 1);

            let maxThreadsAllowed = pipelineState.maxTotalThreadsPerThreadgroup;
            let threadsPerGroup = MTLSize(width: maxThreadsAllowed, height: 1, depth: 1);

            // Dispatch the grid setup
            encoder.dispatchThreads(gridExecutionSize, threadsPerThreadgroup: threadsPerGroup);
        }
    }
    
    /// Observe for a [Float] value
    public func addObserver<O: ResultObserver>(_ observer: O) async {
        await self.observerStore.add(observer);
    }
    
    public func notifyObservers() async {
        // Get result
        let rawPtr : UnsafeMutableRawPointer = self.result.contents();
        let length : Int = self.result.length / MemoryLayout<Float>.stride;
        let typedPointer = rawPtr.bindMemory(to: Float.self, capacity: length);
        let result : [Float] = [Float](UnsafeBufferPointer(start: typedPointer, count: length));
        await self.observerStore.callAll(with: result);
    }
}

class SubtractionFactory: ComputeKernelCreatable {

    private var value: Float = 1.0;
    private var powVal : Float = 1.0;

    // Shared cache of already-uploaded source buffers
    private let bufferCache : BufferCache;

    /// - Parameter bufferCache: Pass a shared instance to reuse buffers across factories; the default
    ///   gives this factory its own private cache.
    init(bufferCache: BufferCache = BufferCache()) {
        self.bufferCache = bufferCache;
    }

    static func getFactoryName() -> String {
        return Subtraction.getFunctionName();
    }
    
    /// Sets the subtraction value the resulting ComputeKernel will use on all elements within the values MTLBuffer
    /// Call this function before createKernel()
    func setSubtractionValue(value: Float) {
        self.value = value;
    }
    
    /// Sets the power value used after the subtraction the resulting ComputeKernel will use on all elements within the values of MTLBuffer
    func setPowValue(value: Float) {
        self.powVal = value;
    }
    
    func createKernel(bufable: any MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel {
        // Get array values, convert to MTLBuffer (put in shared memory)
        let devToAlloc : MTLDevice = context.getDevice();
        let valuesToSub : MTLBuffer = try await self.bufferCache.buffer(for: bufable, device: devToAlloc);

        guard let numOfValues : UInt32 = try? bufable.MTLBufferSize() else {
            fatalError("Failed to get number of values");
        }
        // One result is written per element, thus allocate one Float per value
        guard let resultAlloc = devToAlloc.makeBuffer(length: Int(numOfValues) * MemoryLayout<Float>.stride, options: [.storageModeShared]) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        return await Subtraction(valuesArr: valuesToSub, numOfValues: numOfValues, valueToSubtract: self.value, powerVal: self.powVal, resultBuf: resultAlloc)
    }
}

