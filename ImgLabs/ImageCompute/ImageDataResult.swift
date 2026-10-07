//
//  ImageDataResult.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-14.
//  Denotes a protocol for results which wrap an ImageData object with a derived value

/// A GPU-computed result that wraps an ImageData object together with a single derived value. Lets callers treat the various per-image results
public protocol ImageDataResult: Actor {
    associatedtype Value

    /// The image this result was computed from
    func imageData() -> ImageData

    /// The value derived from the image (e.g., a hash or a sharpness score)
    func value() -> Value
}
