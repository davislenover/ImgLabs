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
public actor ImageHash {
    private static let dctFactory : ComputeKernelCreatable = DCT32Factory(); // static such that re-calls to create a kernel don't re-calculate preConstMtx
    private static let grayScaleFactory : ComputeKernelCreatable = GrayScaleKernelFactory();
    private let image: ImageData;
    private let hashVal : UInt64;
    
    private init(image: ImageData, hashVal: UInt64) {
        self.image = image;
        self.hashVal = hashVal;
    }
    
    public func hash() -> UInt64 {return self.hashVal;}
    public func imageData() -> ImageData {return self.image;}
    
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
                fatalError("No buffer in result");
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

            // Median of the 64 coefficients -- even number of elements, so the mean of the two central sorted values
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
    
    /// Captures a kernel's resident output buffer (published as a DeviceBuffer) so it can be handed straight to
    /// the next kernel's factory without a CPU round-trip
    private class DeviceBufferResult : ResultObserver<DeviceBuffer> {
        var buffer : DeviceBuffer? = nil;
        func update(with: DeviceBuffer) {
            self.buffer = with;
        }
    }
    
    private class FloatArrayResult : ResultObserver<[Float]>, MTBufable {
        func toMTLBuffer(_ device: any MTLDevice) async throws -> any MTLBuffer {
            guard let newBuf : MTLBuffer = device.makeBuffer(bytes: self.result, length: self.result.count*MemoryLayout<Float>.stride, options: .storageModeShared) else {
                throw ImageError.failedToConvertDataToMTLBuffer;
            }
            return newBuf;
        }
        
        func MTLBufferSize() throws -> UInt32 {
            return UInt32(self.result.count);
        }
        
        var result: [Float] = [];
        func update(with: [Float]) {
            self.result = with;
        }
    }
}

