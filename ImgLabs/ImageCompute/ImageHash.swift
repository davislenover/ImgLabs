//
//  ImageHash.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-12.
//  Denotes a class which wraps an ImageData class with a perceptual hashcode

import Metal

/// Wraps an ImageData object with a pHash value (encodes the image content as a 64-bit unsigned integer)
/// pHash value is computed from a DCT transformation of the image (downscaled to 32x32), only frequency waves 0-7 are used
/// 8x8 DCT matrix, each element is compared against the median, 0 if less than and 1 if greater than or equal to median -> 64 binary bits
public actor ImageHash: ImageDataResult {
    private static let dctFactory : ComputeKernelCreatable = DCT32Factory(); // static such that re-calls to create a kernel don't re-calculate preConstMtx
    private static let grayScaleFactory : ComputeKernelCreatable = GrayScaleKernelFactory();
    private let image: ImageData;
    private let hashVal : UInt64;
    
    private init(image: ImageData, hashVal: UInt64) {
        self.image = image;
        self.hashVal = hashVal;
    }
    
    public func imageData() -> ImageData {return self.image;}
    /// The derived value for this result (the perceptual hash)
    public func value() -> UInt64 {return self.hashVal;}
    
    /// Computes the pHash values of all ImageData objects within an array
    /// - Parameters:
    ///     - images: An array of ImageData objects to compute the pHash for
    ///     - MTLContext: The Metal context object with a given Metal supported device
    /// - Returns: The ImageData objects wrapped as ImageHash objects in an array
    public static func toHash(images: [ImageData], MTLContext : MetalComputeContext) async throws -> [ImageHash] {
        // Convert all images to 32x32, collapse all into a strided 3D matrix (buffer on the GPU)
        var img32Copies : [ImageData] = [];
        // Re-size all images to 32x32
        for img in images {
            guard let img32 : ImageData = await img.createResampledCopy(targetWidth: 32, targetHeight: 32) else {
                throw ImageError.failedToResizeImage;
            }
            img32Copies.append(img32);
        }
        // Convert all 32x32 images to grayscale values
        // First compute the grayscale of every image
        var grayScaleKernels : [ComputeKernel] = [];
        var grayScaleImages : [DeviceBufferResult] = [];
        let grayScaleFactory : GrayScaleKernelFactory = GrayScaleKernelFactory();
        for image in img32Copies {
            let grayScaleKernel : ComputeKernel = try await grayScaleFactory.createKernel(bufable: image, context: MTLContext);
            let grayScaleImageResult : DeviceBufferResult = await DeviceBufferResult();
            await grayScaleKernel.addObserver(grayScaleImageResult);
            grayScaleImages.append(grayScaleImageResult);
            grayScaleKernels.append(grayScaleKernel);
        }
        try await MetalRunner.runCompute(from: MTLContext, for: grayScaleKernels); // Will suspend here until completion
        // All DeviceBufferResults will now contain a buffer with 32x32 values (need to stride to access each one)
        // Append all resulting buffers together
        // Calculate total length needed to allocate buffer memory
        var buffers : [MTLBuffer] = [];
        for result in grayScaleImages {
            guard let devBuf : DeviceBuffer = await result.buffer else {
                throw KernelEngineError.missingKernelResult;
            }
            await buffers.append(try devBuf.toMTLBuffer(MTLContext.getDevice()));
        }
        let totalBufLength = buffers.reduce(0) { $0 + $1.length; }
        
        // Allocate the single destination buffer
        guard let destinationBuffer = MTLContext.getDevice().makeBuffer(length: totalBufLength, options: .storageModeShared) else {
            throw KernelEngineError.failedToAllocateMTLBufferMemory;
        }
        
        // Get a pointer pointing to the start of the buffer
        var destinationPointer : UnsafeMutableRawPointer = destinationBuffer.contents();
        
        // Sequentially copy each buffer's contents into the destination
        for buffer in buffers {
            let sourcePointer : UnsafeMutableRawPointer = buffer.contents();
            memcpy(destinationPointer, sourcePointer, buffer.length);
            
            // Advance the destination pointer by the size of the copied buffer
            destinationPointer = destinationPointer.advanced(by: buffer.length);
        }
        // Wrap buffer in a DeviceBuffer (to pass to DCT32 factory)
        let deviceBuffer : DeviceBuffer = await DeviceBuffer(destinationBuffer);
        // Generate kernel
        let dct32Kernel : ComputeKernel = try await Self.dctFactory.createKernel(bufable: deviceBuffer, context: MTLContext);
        let dct32KernelResult : FloatArrayResult = await FloatArrayResult();
        await dct32Kernel.addObserver(dct32KernelResult);
        // Run Kernel
        try await MetalRunner.runCompute(from: MTLContext, for: [dct32Kernel]);
        // Result is a strided 3D matrix, where each Z-axis step is an 8x8 2D strided matrix
        // Per 8x8 matrix, calculate the median value, then compare each element with median to generate hash value
        let dctValues : [Float] = await dct32KernelResult.result;
        let valuesPerImage : Int = 8 * 8; // Each image's DCT occupies a contiguous 8x8 (64 element) block

        var hashes : [ImageHash] = [];
        for (index, image) in images.enumerated() {
            // Slice this image's 8x8 block out of the strided buffer
            let start : Int = index * valuesPerImage;
            let block : ArraySlice<Float> = dctValues[start..<(start + valuesPerImage)];

            // Median of the 64 coefficients, even number of elements, so the mean of the two central sorted values
            let sorted : [Float] = block.sorted();
            let median : Float = (sorted[valuesPerImage / 2 - 1] + sorted[valuesPerImage / 2]) / 2;

            // Build the 64-bit hash: bit i is set when coefficient i is greater than or equal to the median
            var hashVal : UInt64 = 0;
            for (bit, value) in block.enumerated() {
                if value >= median {
                    hashVal |= (UInt64(1) << bit);
                }
            }
            hashes.append(ImageHash(image: image, hashVal: hashVal));
        }
        return hashes;
    }
    
    
    /// Calculates the hamming distance between two pHashes (i.e., by how many bits do the hashes differ)
    /// - Parameters:
    ///     - img1: The first image to compare
    ///     - img2: The second image to compare
    /// - Returns: An unsigned 8-bit integer denoting by how many bits the two 64-bit hashes differ
    public static func getHammingDistance(img1: ImageHash, img2: ImageHash) async -> UInt8 {
        // XOR both hashes then count number of ones, yields the hamming distance
        let result : UInt64 = await img1.value() ^ img2.value();
        let ones : UInt8 = UInt8(result.nonzeroBitCount);
        return ones;
    }
    
    /// Calculates a 2D strided matrix of hamming distances given an array of ImageHash objects (i.e., every image's pHash is compared to all other images)
    /// - Parameters:
    ///     - imagesHashes: An array of ImageHash objects to compare
    /// - Returns: A 2D strided matrix of Uint8's denoting the hamming distance each image's pHash is to one another (i.e., row 0 is input image 0)
    public static func getHammingDistanceMtx(imagesHashes : [ImageHash]) async -> [UInt8] {
        // For every image, get the hamming distance between other images
        // Uses a TaskGroup to calculate the matrix in parallel
        // Allocate a countxcount matrix (strided)
        let count = imagesHashes.count;
        var hammingDistMtx : [UInt8] = Array<UInt8>(repeating: 0, count: count*count);
        await withTaskGroup(of: (Int, [UInt8]).self) { group in
            // Each task computes one full row: image[curIter] against all other images
            for curIter in 0..<count {
                let curImgHashObj : ImageHash = imagesHashes[curIter];
                group.addTask {
                    var row : [UInt8] = Array<UInt8>(repeating: 0, count: count);
                    for otherImgHashObj in imagesHashes.enumerated() {
                        if curIter != otherImgHashObj.offset {
                            row[otherImgHashObj.offset] = await getHammingDistance(img1: curImgHashObj, img2: otherImgHashObj.element);
                        }
                    }
                    return (curIter, row);
                }
            }
            // Copy each completed row into the strided matrix
            for await (curIter, row) in group {
                let start : Int = curIter * count;
                hammingDistMtx.replaceSubrange(start..<(start + count), with: row);
            }
        }
        return hammingDistMtx;
    }
    
}

