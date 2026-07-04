//
//  KernelRegistry.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-03.
//  Denotes an actor which houses ComputeKernel object creation functions to an associated string

internal final actor KernelRegistry {
    private var kernels: [String: ComputeKernelCreatable] = [:]
    
    public func registerKernel(kernelFactory : ComputeKernelCreatable) {
        self.kernels[type(of:kernelFactory).getFactoryName()] = kernelFactory;
    }
    
    public func getKernel(name: String) async throws -> (MTBufable,MetalComputeContext) async throws -> any ComputeKernel {
        guard let kernel = self.kernels[name] else {
            throw KernelEngineError.failedToFindKernelCreateFunction(name: name);
        }
        return kernel.createKernel;
    }
}

public protocol ComputeKernelCreatable {
    static nonisolated func getFactoryName() -> String;
    /// Creates a ComputeKernel instance, handles the creation of MTLBuffers and passes all required parameters/buffers to the corresponding ComputeKernel instance
    /// - Parameters:
    ///     - buf: An object which conforms to the MTBufable protocol (i.e., to allow the ComputeKernel to gather the data it needs to pass to the device)
    ///     - context: An context object which houses information about the device being used (to point the ComputeKernel instance as to where to allocate memory/dispatch threads)
    func createKernel(bufable: MTBufable, context: MetalComputeContext) async throws -> any ComputeKernel;
}
