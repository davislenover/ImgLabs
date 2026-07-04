//
//  ComputeKernel.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Defines the protocol for classes which setup compute kernels

import Foundation
import Metal

/// Denotes a  protocol in which a class is responsible for owning a kernel function and how to bind/dispatch it
/// ComputeKernels are also observable which acts as the primary means for safely returning the finished result
public protocol ComputeKernel : ObservableResult {
    /// Gets the kernel function name, typically used to make the function/pipeline state for the given kernel from the library
    /// The function must be thread-safe
    /// - Returns: The kernel function name as a string
    nonisolated static func getFunctionName() -> String;
    
    /// Abstract lambda logic which the conforming object binds arguments and sets up the thread count for dispatch of the corresponding kernel
    /// Assumes setComputePipelineState was called on the given encoder with the given pipeline prior
    func encode() async -> ((MTLComputeCommandEncoder,MTLComputePipelineState) -> ());
}

