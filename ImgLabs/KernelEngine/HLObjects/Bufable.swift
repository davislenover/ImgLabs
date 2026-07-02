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
}

