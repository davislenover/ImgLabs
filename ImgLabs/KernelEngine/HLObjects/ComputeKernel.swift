//
//  ComputeKernel.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Defines the protocol for classes which setup compute kernels

import Foundation
import Metal

/// Denotes a  protocol in which a class is responsible for owning a kernel function and how to bind/dispatch it
public protocol ComputeKernel {
    /// Gets the kernel function name, typically used to make the function/pipeline state for the given kernel from the library
    /// - Returns: The kernel function name as a string
    func getFunctionName() -> String;
    
    /// Abstract lambda logic which the conforming object binds arguments and sets up the thread count for dispatch of the corresponding kernel
    /// This function may also throw errors if encoding is not succsesful. Assumes setComputePipelineState was called on the given encoder with the given pipeline prior
    func encode() -> ((MTLComputeCommandEncoder,MTLComputePipelineState) throws -> ());
}

