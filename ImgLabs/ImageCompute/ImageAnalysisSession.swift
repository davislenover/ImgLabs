//
//  ImageAnalysisSession.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-19.
//  Coordinates one analysis run over a fixed set of images, computing each shared GPU
//  intermediate exactly once and reusing it across the different analyses.

import Metal

/// Owns a single analysis run over one fixed set of images and makes sure the expensive shared
/// GPU intermediates are computed only once
///
/// The big shared artifact is the **full-canvas grayscale batch**: both the ZNCC similarity matrix
/// and the Laplacian sharpness score start from the grayscale of every image. This session computes
/// that batch lazily on first use, caches it, and hands the same buffers to every consumer so
/// pressing Analyze grayscales each image once instead of once per analysis
///
/// Perceptual hashing is different: it works on 32x32 downscaled copies, a smaller and separate
/// artifact, so it keeps its own pipeline. The session still memoizes the hash result so repeated
/// reads are free
///
/// Each analysis is exposed as its own accessor and is computed lazily
public actor ImageAnalysisSession {
    private let images : [ImageData];
    private let context : MetalComputeContext;

    // Each intermediate is memoized as a Task rather than a plain value. If two callers ask for the
    // same artifact at nearly the same time, the second one awaits the first's in-flight Task instead
    // of kicking off a duplicate computation. This mirrors how MetalComputeContext caches pipeline
    // compilation (a cached Task keyed by function name).
    private var grayscaleTask : Task<[DeviceBuffer], Error>?;
    private var matrixTask : Task<[Float], Error>?;
    private var sharpnessTask : Task<[ImageSharpness], Error>?;
    private var hashTask : Task<[ImageHash], Error>?;

    /// - Parameters:
    ///     - images: The images to analyze. Every result is aligned to this order
    ///     - context: The Metal context all GPU work runs against
    public init(images: [ImageData], context: MetalComputeContext) {
        self.images = images;
        self.context = context;
    }

    // MARK: - Public analyses

    /// The all-pairs ZNCC similarity matrix, strided (row-major, element (i, j) at i * count + j).
    /// Reuses the shared grayscale batch
    public func similarityMatrix() async throws -> [Float] {
        if let existing = self.matrixTask { return try await existing.value; }
        let context = self.context;
        let task = Task {
            let grayscale : [DeviceBuffer] = try await self.grayscaleBatch();
            return try await ImageCorrelation(MetalContext: context).similarityMatrix(grayscaleBuffers: grayscale);
        }
        self.matrixTask = task;
        return try await task.value;
    }

    /// One sharpness score per image (variance of the Laplacian). Reuses the shared grayscale batch
    public func sharpnessScores() async throws -> [ImageSharpness] {
        if let existing = self.sharpnessTask { return try await existing.value; }
        let context = self.context;
        let images = self.images;
        let task = Task {
            let grayscale : [DeviceBuffer] = try await self.grayscaleBatch();
            // Every image shares a common canvas, so one size drives the batched convolution/variance
            guard let size = await images.first?.currentSize() else { throw ImageError.noPixelData; }
            return try await ImageSharpness.toSharpness(images: images, grayscaleBuffers: grayscale, size: size, MTLContext: context);
        }
        self.sharpnessTask = task;
        return try await task.value;
    }

    /// One perceptual hash per image. Uses its own 32x32 pipeline (not the shared grayscale batch)
    public func hashes() async throws -> [ImageHash] {
        if let existing = self.hashTask { return try await existing.value; }
        let context = self.context;
        let images = self.images;
        let task = Task {
            return try await ImageHash.toHash(images: images, MTLContext: context);
        }
        self.hashTask = task;
        return try await task.value;
    }

    // MARK: - Shared intermediate

    /// The full-canvas grayscale of every image (one resident DeviceBuffer per image, in input order)
    /// Computed once on first call then every later call returns the cached buffers
    private func grayscaleBatch() async throws -> [DeviceBuffer] {
        if let existing = self.grayscaleTask { return try await existing.value; }
        let context = self.context;
        let images = self.images;
        let task = Task {
            return try await Self.computeGrayscaleBatch(images: images, context: context);
        }
        self.grayscaleTask = task;
        return try await task.value;
    }

    /// Grayscales every image in one batched GPU submission and returns the resident output buffers
    /// This is the shared first stage hoisted out of the individual analyses
    private static func computeGrayscaleBatch(images: [ImageData], context: MetalComputeContext) async throws -> [DeviceBuffer] {
        guard !images.isEmpty else { return []; }
        let grayScaleFactory : GrayScaleKernelFactory = GrayScaleKernelFactory();
        var kernels : [ComputeKernel] = [];
        var results : [DeviceBufferResult] = [];
        for image in images {
            let kernel : ComputeKernel = try await grayScaleFactory.createKernel(bufable: image, context: context);
            let result : DeviceBufferResult = await DeviceBufferResult();
            await kernel.addObserver(result);
            kernels.append(kernel);
            results.append(result);
        }
        try await MetalRunner.runCompute(from: context, for: kernels); // Suspends until the GPU finishes
        // Collect each kernel's resident grayscale buffer, keeping input order
        var buffers : [DeviceBuffer] = [];
        for result in results {
            guard let buffer = await result.buffer else { throw KernelEngineError.missingKernelResult; }
            buffers.append(buffer);
        }
        return buffers;
    }
}
