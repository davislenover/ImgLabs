//
//  Bufable.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//  Denotes a protocol where implementing objects can convert some form of data to an MTLBuffer

import Metal

/// An object which can convert some form of data to an MTLBuffer
public protocol MTBufable {
    /// Converts specified data to an MTLBuffer, assumes shared memory
    /// This function may throw an error and may suspend while buffer is allocating (in case processing power is needed to determine allocation logic)
    /// - Parameters:
    ///     - device: The given device to allocate the buffer on
    /// - Returns: The buffer as an MTLBuffer type
    func toMTLBuffer(_ device: MTLDevice) async throws -> MTLBuffer;
    
    /// Gets the expected number of elements that the resulting MTLBuffer (from toMTLBuffer) was allocated for
    func MTLBufferSize() throws -> UInt32;
}

/// Wraps an MTLBuffer that already lives on the compute device. As an MTBufable, toMTLBuffer hands back that
/// same buffer with no allocation or copy, so one kernel's output can be fed straight into the next kernel's
/// input while the data stays resident on the GPU
/// Assumes the buffer holds Float elements, as every array-producing kernel in this engine does
public final class DeviceBuffer : MTBufable {
    private let buffer : MTLBuffer;

    /// - Parameter buffer: an MTLBuffer already allocated on the device (typically a kernel's output buffer)
    public init(_ buffer: MTLBuffer) {
        self.buffer = buffer;
    }

    /// Returns the wrapped buffer unchanged. The device argument is ignored: the buffer is assumed to already
    /// live on the device in use, so there is nothing to allocate or copy
    public func toMTLBuffer(_ device: MTLDevice) async throws -> MTLBuffer {
        return self.buffer;
    }

    /// The number of Float elements the wrapped buffer holds
    public func MTLBufferSize() throws -> UInt32 {
        return UInt32(self.buffer.length / MemoryLayout<Float>.stride);
    }
}

