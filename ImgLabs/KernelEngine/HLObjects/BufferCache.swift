//
//  BufferCache.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-05.
//  Denotes a stored actor in which identical source data is only uploaded to the GPU once

import Metal

/// Caches the MTLBuffer produced for each MTBufable source, keyed by the source object's reference identity
/// Handing one of these to a factory lets an array that is reused across many kernels (e.g. every pair in an
/// all-pairs similarity matrix) be uploaded to the device a single time instead of once per kernel.
///
/// Because entries are keyed by reference identity, only class-based MTBufable conformers are cached; a
/// value-type conformer is converted every time. Cached buffers are assumed
/// to reflect stable source contents -- if a bufable's data changes, call remove(_:) or clear() first
/// An actor so a single instance can be shared safely across factories (and tasks) without data races
public actor BufferCache {
    // Cache the upload as a Task (mirrors MetalComputeContext's pipeline cache): concurrent requests for
    // the same source await the one in-flight upload instead of racing to duplicate it
    private var cache : [ObjectIdentifier: Task<MTLBuffer, Error>] = [:];

    public init() {}

    /// Returns the MTLBuffer for a bufable, reusing a previously created one when the same object has
    /// already been converted through this cache; otherwise converts it once and stores the result
    /// - Parameters:
    ///     - bufable: The source data to convert to (or fetch from the cache as) an MTLBuffer
    ///     - device: The device the buffer is allocated on
    /// - Returns: The cached or freshly created MTLBuffer
    public func buffer(for bufable: any MTBufable, device: MTLDevice) async throws -> MTLBuffer {
        let key = ObjectIdentifier(bufable as AnyObject);
        // If a task already exists, await it directly
        if let existing = self.cache[key] {
            return try await existing.value;
        }
        // Spawn the upload and store the TASK synchronously, before any await, so racers find it
        let uploadTask = Task { () throws -> MTLBuffer in
            return try await bufable.toMTLBuffer(device);
        }
        self.cache[key] = uploadTask;
        return try await uploadTask.value;
    }

    /// Drops the cached buffer for a single source, if present
    public func remove(_ bufable: any MTBufable) {
        self.cache.removeValue(forKey: ObjectIdentifier(bufable as AnyObject));
    }

    /// Drops all cached buffers, releasing the GPU memory they hold
    public func clear() {
        self.cache.removeAll();
    }
}
