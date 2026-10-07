//
//  KeeperStrategy.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-19.
//  Strategy for choosing which image in a duplicate cluster to keep

import Foundation

// MARK: - Per-image quality

/// A bundle of the quality signals used to pick a cluster's keeper
/// One per analyzed image, in the same order as the similarity matrix
public struct ImageQuality: Sendable {
    /// Variance of the Laplacian -- higher means sharper (more in focus)
    public let sharpness: Float;
    /// Original resolution in pixels (width * height), BEFORE the common-canvas resample
    public let pixelCount: Int;
    /// Size of the source file on disk, in bytes
    public let fileSizeBytes: Int;
    /// Preference for the source file format (higher = more preferred; RAW > lossless > lossy)
    public let formatRank: Int;

    nonisolated public init(sharpness: Float, pixelCount: Int, fileSizeBytes: Int, formatRank: Int) {
        self.sharpness = sharpness;
        self.pixelCount = pixelCount;
        self.fileSizeBytes = fileSizeBytes;
        self.formatRank = formatRank;
    }
    
    /// Assembles the quality bundle for every image from its sharpness score plus metadata read off the
    /// ImageData (original resolution) and its source file (size + format), keeping input order.
    /// - Parameters:
    ///     - images: The analyzed images
    ///     - sharpness: Per-image sharpness scores, index-aligned with images
    /// - Returns: One ImageQuality per image, in the same order
    /// nonisolated so the per-file disk reads run off the main actor (the project defaults isolation to
    /// MainActor); reading ImageData's MainActor-isolated metadata hops to the main actor as needed
    nonisolated static func build(images: [ImageData], sharpness: [Float]) async -> [ImageQuality] {
        var result: [ImageQuality] = [];
        for (index, image) in images.enumerated() {
            let score: Float = index < sharpness.count ? sharpness[index] : 0;
            // Resolution comes from the ORIGINAL size, not the resampled canvas
            let size = await image.originalSize();
            let pixels: Int = (size?.width ?? 0) * (size?.height ?? 0);
            let url = await image.getURL();
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path);
            let bytes: Int = (attributes?[.size] as? Int) ?? 0;
            result.append(ImageQuality(sharpness: score, pixelCount: pixels, fileSizeBytes: bytes, formatRank: Self.formatRank(for: url)));
        }
        return result;
    }

    /// Ranks a source file's format by how much detail it tends to preserve (higher = keep-worthier)
    nonisolated private static func formatRank(for url: URL) -> Int {
        switch url.pathExtension.lowercased() {
        case "arw", "cr2", "cr3", "nef", "dng", "raf", "orf", "rw2", "pef", "srw": return 3; // RAW
        case "png", "tif", "tiff": return 2;                                                  // lossless
        case "heic", "heif", "webp": return 1;
        default: return 0;                                                                    // jpeg & everything else
        }
    }
}

// MARK: - Strategy

/// Chooses which image in a duplicate cluster to keep. Injecting this into duplicateGroups keeps the
/// selection algorithm swappable (and unit-testable) without touching the clustering logic
public protocol KeeperStrategy {
    /// - Parameters:
    ///     - members: Indices (into the images/matrix order) of the images in one cluster
    ///     - matrix: The full ZNCC similarity matrix, strided (row-major, element (i, j) at i * count + j)
    ///     - count: N, the number of images (the side length of the strided matrix)
    ///     - quality: Per-image quality, index-aligned with the images (may be empty if unavailable)
    /// - Returns: The index of the image to keep
    func keeper(from members: [Int], matrix: [Float], count: Int, quality: [ImageQuality]) -> Int;
}

/// The original behavior: keep the medoid -- the image most similar on average to its cluster peers.
/// Uses only the similarity matrix, so it works even when no quality signals are available
public struct MedoidStrategy: KeeperStrategy {
    public init() {}

    public func keeper(from members: [Int], matrix: [Float], count: Int, quality: [ImageQuality]) -> Int {
        guard members.count > 1 else { return members[0]; }
        // The member never "less than" any other by average similarity is the most representative one
        return members.max(by: { a, b in
            averageSimilarity(of: a, within: members, matrix: matrix, count: count)
            < averageSimilarity(of: b, within: members, matrix: matrix, count: count);
        })!;
    }
}

/// Keeps the highest-scoring image by a weighted blend of quality signals: sharpness, resolution, file
/// size, format, and how representative it is of the cluster. Each signal is normalized WITHIN the cluster
/// before weighting, because the raw scales are wildly different (a Laplacian variance vs a pixel count vs
/// a byte count) and only relative ordering within the cluster matters for picking a keeper
public struct WeightedQualityStrategy: KeeperStrategy {

    /// How much each normalized signal counts toward the final score. Defaults favor a sharp, high-resolution
    /// keeper, with smaller nudges from file size/format and cluster representativeness
    public struct Weights {
        public var sharpness: Float;
        public var resolution: Float;
        public var fileSize: Float;
        public var format: Float;
        public var representativeness: Float;

        public init(sharpness: Float = 0.5, resolution: Float = 0.25, fileSize: Float = 0.1, format: Float = 0.05, representativeness: Float = 0.1) {
            self.sharpness = sharpness;
            self.resolution = resolution;
            self.fileSize = fileSize;
            self.format = format;
            self.representativeness = representativeness;
        }
    }

    private let weights: Weights;

    public init(weights: Weights = Weights()) {
        self.weights = weights;
    }

    public func keeper(from members: [Int], matrix: [Float], count: Int, quality: [ImageQuality]) -> Int {
        guard members.count > 1 else { return members[0]; }
        // Without aligned quality can't score the quality signals, fall back to the medoid (no index out of range errors)
        guard quality.count == count else {
            return MedoidStrategy().keeper(from: members, matrix: matrix, count: count, quality: quality);
        }

        // Gather each signal across the cluster, then normalize each to [0, 1] within the cluster
        let sharpness = Self.normalize(members.map { quality[$0].sharpness; });
        let resolution = Self.normalize(members.map { Float(quality[$0].pixelCount); });
        let fileSize = Self.normalize(members.map { Float(quality[$0].fileSizeBytes); });
        let format = Self.normalize(members.map { Float(quality[$0].formatRank); });
        let representativeness = Self.normalize(members.map { averageSimilarity(of: $0, within: members, matrix: matrix, count: count); });

        // Weighted sum per member, keep the highest. Ties resolve to the earliest member for determinism
        var bestMember: Int = members[0];
        var bestScore: Float = -Float.greatestFiniteMagnitude;
        for (position, member) in members.enumerated() {
            let score = self.weights.sharpness * sharpness[position]
                + self.weights.resolution * resolution[position]
                + self.weights.fileSize * fileSize[position]
                + self.weights.format * format[position]
                + self.weights.representativeness * representativeness[position];
            if score > bestScore {
                bestScore = score;
                bestMember = member;
            }
        }
        return bestMember;
    }

    /// Min-max normalizes a cluster's values to [0, 1]. When every value is equal the signal can't
    /// discriminate between members, so it contributes 0 (neutral) rather than dividing by zero
    private static func normalize(_ values: [Float]) -> [Float] {
        guard let low = values.min(), let high = values.max(), high > low else {
            return Array(repeating: 0, count: values.count);
        }
        return values.map { ($0 - low) / (high - low); };
    }
}
