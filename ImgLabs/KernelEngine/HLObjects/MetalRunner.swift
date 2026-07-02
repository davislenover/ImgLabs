//
//  MetalRunner.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Denotes an actor which is responsible for executing compute kernels and watching their status

import Metal

public actor MetalRunner {
    /// Takes all kernels and executes them against a MetalComputeContext
    ///  - Parameters:
    ///     - from: The MetalComputeContext object to get the device, pipeline state (per kernel) and queue from
    ///     - kernels: The list of compute kernels to execute
    ///     - cb : The callback function to call once all kernels in the list finish execution (will outlive this function, is thread-safe)
    /// - Throws: KernelEngineError if setup fails
    public static func runCompute(from context : MetalComputeContext, for kernels : [ComputeKernel], callback cb : @escaping @Sendable (MTLCommandBuffer) -> ()) async throws -> () {
        // await may pick back up execution on a different thread...Metal objects arn't inherently thread safe
        // thus, create all objects (thread doesn't matter), then only when modifying those objects, use the same thread
        var pipelines: [MTLComputePipelineState] = [];
        var kernelEncodingLogics: [(MTLComputeCommandEncoder,MTLComputePipelineState) throws -> ()] = [];
        for kernel in kernels {
            let pipeline = try await context.getPipelineState(for: kernel); // need to await as we are calling another actor in which-
            // a lock on it may be held by another thread so waiting might be the only option
            let encodingLogic = await kernel.encode(); // Gets function kernel uses to bind arguments and set dispatch threads
            pipelines.append(pipeline);
            kernelEncodingLogics.append(encodingLogic);
        }
        guard let commandBuffer = context.getQueue().makeCommandBuffer() else {
            throw KernelEngineError.failedToCreateComputeCommandBuffer;
        }
        
        // Encoding and executing of Kernels now happens on the same thread
        // Zip encoding lambda's pre-fetched pipelines together
        for (pipeline,encoding) in zip(pipelines,kernelEncodingLogics) {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw KernelEngineError.failedToCreateComputeCommandEncoder;
            }
            encoder.setComputePipelineState(pipeline);
            try encoding(encoder, pipeline);
            encoder.endEncoding();
        }
        commandBuffer.addCompletedHandler(cb);
        commandBuffer.commit(); // Run all the work on the GPU
    }
}
