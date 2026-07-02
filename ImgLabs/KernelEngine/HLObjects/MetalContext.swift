//
//  MetalContext.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Denotes a MetalContext actor who's job is to cache and return Metal pipeline states

import Foundation
import Metal

/// A compute engine context actor which handles the creation and caching of Metal compute pipeline states
/// All concurrent/parallel access to this class is serialized, meaning only one thread may operate on the class at any given time
public actor MetalComputeContext {
    private let device: MTLDevice; // MTLDevice is thread-safe
    private let queue: MTLCommandQueue; // MTLCommandQueue is thread-safe
    private let library: MTLLibrary;
    
    // Hashmap with function name as key, value is the corresponding pipeline state
    private var pipelineCache: [String: Task<MTLComputePipelineState, Error>] = [:];
    
    /// Create a new MetalComputeContext
    /// - Returns: MetalComputeContext optional -- nil if creating default device, command queue or default library fails, otherwise returns an instance
    init?() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(), let library = device.makeDefaultLibrary() else {
            return nil;
        }
        self.device = device;
        self.library = library;
        self.queue = queue;
    }
    
    /// Gets the compute pipeline state object for a given function
    /// - Parameters: functionName
    public func getPipelineState(for computeObject: ComputeKernel) async throws -> MTLComputePipelineState {
        let functionName = await computeObject.getFunctionName();
        
        // If a task already exists (either running or finished), await it directly
        if let existingTask = self.pipelineCache[functionName] {
            return try await existingTask.value;
        }
        
        // Otherwise, spawn a new task to do the compilation (happens on another thread)
        let compilationTask = Task { // Registers a new Task but won't run right away given the thread which registered the task holds the actor lock
            () throws -> MTLComputePipelineState in
                guard let function = self.library.makeFunction(name: functionName) else {
                    throw KernelEngineError.kernelFunctionNotFound(name: functionName);
                }
                guard let pipelineState = try? self.device.makeComputePipelineState(function: function) else {
                    throw KernelEngineError.deviceFailedToCreateKernel(name: functionName);
                }
            return pipelineState;
        }
        
        // Save the TASK to the cache synchronously before hitting any awaits
        self.pipelineCache[functionName] = compilationTask;
        
        // Await the task completion
        return try await compilationTask.value;
    }
    
    /// Gets the Metal command queue, guarenteed to not be nil if the MetalComputeContext instance was created. Can be called without lock on actor object
    /// - Returns: Command queue of type MTLCommandQueue
    public nonisolated func getQueue() -> MTLCommandQueue {return self.queue;}
    
    /// Gets the Metal device this context is associated with, guarenteed to not be nil if the MetalComputeContext instance was created. Can be called without lock on actor object
    /// - Returns: Device of type MTLDevice
    public nonisolated func getDevice() -> MTLDevice {return self.device;}
    
}

