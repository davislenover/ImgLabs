//
//  KernelEngineError.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Created for specifying custom errors related to Metal kernel functions

public enum KernelEngineError: Error {
    case kernelFunctionNotFound(name: String)
    case deviceFailedToCreateKernel(name: String)
    case failedToBindArguments(argumentsNotBound: [String])
    case failedToAllocateMTLBufferMemory
    case failedToCreateComputeCommandBuffer
    case failedToCreateComputeCommandEncoder
    case failedToDispatchThreads
    case failedToFindKernelCreateFunction(name: String)
}

