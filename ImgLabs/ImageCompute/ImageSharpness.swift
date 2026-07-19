//
//  ImageSharpness.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-14.
//  Denotes a class which wraps an ImageData object with a sharpness score

import Metal

/// Wraps an ImageData object with a sharpness score (the variance of its Laplacian)
/// Higher scores indicate a sharper (more in-focus) image, lower scores indicate blur
/// Every image in a batch is assumed to share the same size (the UI resamples all images to a common canvas on ingestion),
/// so the whole batch is scored with a single convolution and a single variance dispatch over one 3D strided matrix
public actor ImageSharpness: ImageDataResult {
    private let image: ImageData;
    private let scoreVal: Float;

    private init(image: ImageData, scoreVal: Float) {
        self.image = image;
        self.scoreVal = scoreVal;
    }

    public func imageData() -> ImageData {return self.image;}
    /// The derived value for this result (the sharpness score) -- satisfies ImageDataResult
    public func value() -> Float {return self.scoreVal;}

    /// Computes the sharpness score (variance of the Laplacian) of every ImageData object within an array
    /// The pipeline runs entirely on the GPU: grayscale -> Laplacian convolution -> per-image variance
    /// All images are assumed to share the same size, so grayscale outputs are packed into one 3D strided matrix and the
    /// convolution and variance each run as a single dispatch over the whole batch
    /// - Parameters:
    ///     - images: An array of ImageData objects to score (all assumed to be the same size)
    ///     - MTLContext: The Metal context object with a given Metal supported device
    /// - Returns: The ImageData objects wrapped as ImageSharpness objects (aligned to the input order)
    public static func toSharpness(images: [ImageData], MTLContext: MetalComputeContext) async throws -> [ImageSharpness] {
        guard !images.isEmpty else { return []; }
        // Every image shares the same size
        guard let size: (width: Int, height: Int) = await images[0].currentSize() else {
            throw ImageError.noPixelData;
        }

        // Convert every image to grayscale (one float per pixel), batched into a single GPU submission, then
        // delegate to the shared-grayscale variant below. Splitting the grayscale stage out lets a coordinator
        // grayscale once and share the result with the ZNCC similarity matrix
        let grayScaleFactory: GrayScaleKernelFactory = GrayScaleKernelFactory();
        var grayScaleKernels: [ComputeKernel] = [];
        var grayScaleResults: [DeviceBufferResult] = [];
        for image in images {
            let kernel: ComputeKernel = try await grayScaleFactory.createKernel(bufable: image, context: MTLContext);
            let result: DeviceBufferResult = await DeviceBufferResult();
            await kernel.addObserver(result);
            grayScaleKernels.append(kernel);
            grayScaleResults.append(result);
        }
        try await MetalRunner.runCompute(from: MTLContext, for: grayScaleKernels); // Suspends until completion

        // Collect the resident grayscale buffers and run the rest of the pipeline against them
        var grayscaleBuffers: [DeviceBuffer] = [];
        for result in grayScaleResults {
            guard let devBuf: DeviceBuffer = await result.buffer else {
                throw KernelEngineError.missingKernelResult;
            }
            grayscaleBuffers.append(devBuf);
        }
        return try await toSharpness(images: images, grayscaleBuffers: grayscaleBuffers, size: size, MTLContext: MTLContext);
    }

    /// Computes the sharpness score of every image from pre-computed grayscale buffers (one per image, in order)
    /// This is the shared core of the sharpness pipeline: everything except the grayscale stage
    /// - Parameters:
    ///     - images: The images being scored (used to tag each result; aligned to grayscaleBuffers)
    ///     - grayscaleBuffers: The full-canvas grayscale of each image, already resident on the GPU
    ///     - size: The common canvas size shared by every image (drives the batched convolution/variance)
    ///     - MTLContext: The Metal context object with a given Metal supported device
    /// - Returns: The ImageData objects wrapped as ImageSharpness objects (aligned to the input order)
    public static func toSharpness(images: [ImageData], grayscaleBuffers: [DeviceBuffer], size: (width: Int, height: Int), MTLContext: MetalComputeContext) async throws -> [ImageSharpness] {
        guard !images.isEmpty else { return []; }

        // Pack every grayscale image back-to-back into one contiguous 3D strided matrix (each slice is width*height floats)
        var grayBuffers: [MTLBuffer] = [];
        for devBuf in grayscaleBuffers {
            grayBuffers.append(try await devBuf.toMTLBuffer(MTLContext.getDevice()));
        }
        let totalBufLength: Int = grayBuffers.reduce(0) { $0 + $1.length; }
        guard let packedBuffer = MTLContext.getDevice().makeBuffer(length: totalBufLength, options: .storageModeShared) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        // Sequentially copy each grayscale buffer into the packed destination
        var destinationPointer: UnsafeMutableRawPointer = packedBuffer.contents();
        for buffer in grayBuffers {
            memcpy(destinationPointer, buffer.contents(), buffer.length);
            destinationPointer = destinationPointer.advanced(by: buffer.length);
        }
        let packedDeviceBuffer: DeviceBuffer = await DeviceBuffer(packedBuffer);

        // One Laplacian convolution over the whole batch
        // numColumns is the image width (X), numRows is the image height (Y). The Laplacian filter is the factory default
        let convolFactory: ConvoluteImageFactory = ConvoluteImageFactory(numRows: UInt32(size.height), numColumns: UInt32(size.width));
        let convolKernel: ComputeKernel = try await convolFactory.createKernel(bufable: packedDeviceBuffer, context: MTLContext);
        let convolResult: DeviceBufferResult = await DeviceBufferResult();
        await convolKernel.addObserver(convolResult);
        try await MetalRunner.runCompute(from: MTLContext, for: [convolKernel]); // Suspends until completion

        guard let convolBuf: DeviceBuffer = await convolResult.buffer else {
            throw KernelEngineError.missingKernelResult;
        }

        // One variance dispatch over the batch -> one sharpness score per image (variance of the Laplacian)
        let varianceFactory: VarianceFactory = VarianceFactory(elemsPerSet: UInt32(size.width * size.height));
        let varianceKernel: ComputeKernel = try await varianceFactory.createKernel(bufable: convolBuf, context: MTLContext);
        let varianceResult: FloatArrayResult = await FloatArrayResult();
        await varianceKernel.addObserver(varianceResult);
        try await MetalRunner.runCompute(from: MTLContext, for: [varianceKernel]); // Suspends until completion

        // One variance value per image, in the same order they were packed (which matches the input order)
        let scores: [Float] = await varianceResult.result;
        guard scores.count == images.count else {
            throw KernelEngineError.missingKernelResult;
        }
        var sharpnessScores: [ImageSharpness] = [];
        for (index, image) in images.enumerated() {
            sharpnessScores.append(ImageSharpness(image: image, scoreVal: scores[index]));
        }
        return sharpnessScores;
    }
}
