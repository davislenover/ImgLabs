//
//  ZNCCModel.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  Holds the analysis results (similarity matrix, quality signals, hash distances) and runs the pipeline

import SwiftUI

@Observable
class ZNCCModel {
    private var znccResults : [Float] = [];
    private var qualityResults : [ImageQuality] = [];
    private var hammingResults : [UInt8] = [];
    private let metalContext : MetalComputeContext? = MetalComputeContext();

    /// The most recent similarity matrix, strided (row-major, element (i, j) at i * imageCount + j). Empty
    /// until an analysis has run. Read-only to callers
    public var results : [Float] { self.znccResults; }

    /// Per-image quality signals (sharpness, resolution, file size, format), aligned to the analyzed image
    /// order. Empty until an analysis has run. Consumed by the keeper-selection strategy
    public var quality : [ImageQuality] { self.qualityResults; }

    /// All-pairs Hamming-distance matrix of the images' perceptual hashes, strided (row-major, element
    /// (i, j) at i * imageCount + j = differing bits between image i and j). Empty until an analysis has
    /// run. Consumed by DuplicateView to add near-duplicate edges to the ZNCC clustering
    public var hammingMatrix : [UInt8] { self.hammingResults; }

    /// Discards the current results (e.g. when the underlying images change so the results are stale)
    public func clear() {
        self.znccResults = [];
        self.qualityResults = [];
        self.hammingResults = [];
    }

    /// Runs the main analysis algorithm (generates all scores)
    public func getZNCC(_ imgList : [ImageData]) async throws {
        guard let context = metalContext else {
            return;
        }
        // One session over this image set. The grayscale batch is computed once and shared by the similarity
        // matrix and the sharpness scores; perceptual hashing runs its own 32x32 pipeline
        let session = ImageAnalysisSession(images: imgList, context: context);
        let matrix = try await session.similarityMatrix();
        let sharpness = try await session.sharpnessScores();
        let hashes = try await session.hashes();

        // Pull the plain values off the result actors, keeping input order, so the UI/model holds Sendable arrays
        var sharpnessValues : [Float] = [];
        for result in sharpness {
            await sharpnessValues.append(result.value());
        }
        // Build the all-pairs Hamming-distance matrix, off the hash actors, so the sync clustering can treat
        // it like the ZNCC matrix. Both are strided (row-major) [element (i, j) at i * count + j]
        let hammingMatrix : [UInt8] = await ImageHash.getHammingDistanceMtx(imagesHashes: hashes);
        // Combine the sharpness scores with each image's resolution/file metadata into the quality signals
        // the keeper strategy scores against
        let quality = await ImageQuality.build(images: imgList, sharpness: sharpnessValues);

        self.znccResults = matrix;
        self.qualityResults = quality;
        self.hammingResults = hammingMatrix;
    }
}
