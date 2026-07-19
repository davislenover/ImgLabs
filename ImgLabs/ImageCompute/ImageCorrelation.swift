//
//  ImageCorrelation.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-04.
//  Denotes a class which calculates how similar two images are

import Metal

class ImageCorrelation {
    private let computeContext: MetalComputeContext;
    
    init(MetalContext: MetalComputeContext) {
        self.computeContext = MetalContext;
    }
    
    /// Calculates the zero normalized cross correlation between two images
    ///  Both images should be of the same pixel width and height
    /// - Parameters:
    ///     - image1: The first image to compare
    ///     - image2: The second image to compare
    /// - Returns: A floating point value between -1 and 1 (A higher value means more similarity between the two images; 1 means identical)
    func howSimilar(image1: ImageData, image2: ImageData) async throws -> Float {
        // If the exact same images are used in ZNCC, the result should be 1
        let factoryGrayScale = GrayScaleKernelFactory();
        let factoryMean = MeanValueFactory();
        let factorySubtractionOne = SubtractionFactory();
        let factorySubtractionSqr = SubtractionFactory();
        factorySubtractionSqr.setPowValue(value:2.0);
        let factoryDot : DotProductFactory = DotProductFactory();
        
        print("Computing grayscale...");
        // First compute grayscale of both images. As in similarityMatrix, the array intermediates stay resident
        // on the GPU (published as DeviceBuffers) and feed straight into the next stage without a CPU round-trip
        let grayScaleKernelImg1 : ComputeKernel = try await factoryGrayScale.createKernel(bufable: image1, context: self.computeContext);
        let grayScaleValImg1 : DeviceBufferResult = DeviceBufferResult();
        await grayScaleKernelImg1.addObserver(grayScaleValImg1);
        let grayScaleKernelImg2 : ComputeKernel = try await factoryGrayScale.createKernel(bufable: image2, context: self.computeContext);
        let grayScaleValImg2 : DeviceBufferResult = DeviceBufferResult();
        await grayScaleKernelImg2.addObserver(grayScaleValImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [grayScaleKernelImg1,grayScaleKernelImg2]);

        // Next find the mean value of the grayscale
        print("Finding mean of grayscale...");
        let meanKernelImg1 : ComputeKernel = try await factoryMean.createKernel(bufable: grayScaleValImg1.buffer!, context: self.computeContext);
        let meanValImg1 : FloatValueResult = FloatValueResult();
        await meanKernelImg1.addObserver(meanValImg1);
        let meanKernelImg2 : ComputeKernel = try await factoryMean.createKernel(bufable: grayScaleValImg2.buffer!, context: self.computeContext);
        let meanValImg2 : FloatValueResult = FloatValueResult();
        await meanKernelImg2.addObserver(meanValImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [meanKernelImg1,meanKernelImg2]);

        // Next find the subtraction of the grayscale values from the mean
        print("Finding subtraction of grayscale from mean...");
        let subValsImg1 : DeviceBufferResult = DeviceBufferResult();
        factorySubtractionOne.setSubtractionValue(value: meanValImg1.result);
        let subKernelImg1 : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayScaleValImg1.buffer!, context: self.computeContext);
        await subKernelImg1.addObserver(subValsImg1);
        let subValsImg2 : DeviceBufferResult = DeviceBufferResult();
        factorySubtractionOne.setSubtractionValue(value: meanValImg2.result);
        let subKernelImg2 : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayScaleValImg2.buffer!, context: self.computeContext);
        await subKernelImg2.addObserver(subValsImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [subKernelImg1,subKernelImg2]);

        // Do the same as before but with the subtraction result being put to a power of 2
        print("Finding subtraction of grayscale from mean power of 2...");
        let subSqrValsImg1 : DeviceBufferResult = DeviceBufferResult();
        // subVals already has the mean removed; only square it here (subtract 0), otherwise the mean is removed twice
        factorySubtractionSqr.setSubtractionValue(value: 0.0);
        let subSqrKernelImg1 : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: subValsImg1.buffer!, context: self.computeContext);
        await subSqrKernelImg1.addObserver(subSqrValsImg1);
        let subSqrValsImg2 : DeviceBufferResult = DeviceBufferResult();
        // subVals already has the mean removed; only square it here (subtract 0), otherwise the mean is removed twice
        let subSqrKernelImg2 : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: subValsImg2.buffer!, context: self.computeContext);
        await subSqrKernelImg2.addObserver(subSqrValsImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [subSqrKernelImg1,subSqrKernelImg2]);

        // Find the summation of squared results from before
        print("Finding summation of squared results...");
        let sumSqrValsImg1 : FloatValueResult = FloatValueResult();
        let sumSqrKernelImg1 : ComputeKernel = try await factoryMean.createKernel(bufable: subSqrValsImg1.buffer!, context: self.computeContext);
        await sumSqrKernelImg1.addObserver(sumSqrValsImg1);
        let sumSqrValsImg2 : FloatValueResult = FloatValueResult();
        let sumSqrKernelImg2 : ComputeKernel = try await factoryMean.createKernel(bufable: subSqrValsImg2.buffer!, context: self.computeContext);
        await sumSqrKernelImg2.addObserver(sumSqrValsImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [sumSqrKernelImg1,sumSqrKernelImg2]);
        let sumSqrImg1 : Float = sumSqrValsImg1.result * Float(try subSqrValsImg1.buffer!.MTLBufferSize()); // Mean so multiply by number of vals
        let sumSqrImg2 : Float = sumSqrValsImg2.result * Float(try subSqrValsImg2.buffer!.MTLBufferSize()); // Mean so multiply by number of vals

        // Find the dot product of the non squared results
        print("Finding dot product of grayscale subtraction from mean...");
        let dotVal : FloatValueResult = FloatValueResult();
        factoryDot.setArr2(bufable: subValsImg2.buffer!);
        let dotKernel : ComputeKernel = try await factoryDot.createKernel(bufable: subValsImg1.buffer!, context: self.computeContext);
        await dotKernel.addObserver(dotVal);
        try await MetalRunner.runCompute(from: self.computeContext, for: [dotKernel]);

        // return the ZNCC result
        return dotVal.result / (sumSqrImg1 * sumSqrImg2).squareRoot();
    }
    
    /// Compares each image within images array to all other images
    /// - Parameters:
    ///     - images: The images in an array to compare
    /// - Returns: A matrix of floating point values, denoting how simillar image on row y to image on column x is (from 0 to 1, higher values being more simillar)
    func similarityMatrix(images: [ImageData]) async throws -> [[Float]] {
        // Nothing to compare -- return an empty matrix (mirrors the old per-pair loop, which produced none)
        guard !images.isEmpty else { return []; }

        // First compute the grayscale of every image, then hand the grayscale buffers to the
        // shared core below. Splitting the grayscale stage out lets a coordinator
        // compute grayscale once and reuse it
        var grayScaleKernels : [ComputeKernel] = [];
        var grayScaleImages : [DeviceBufferResult] = [];
        let grayScaleFactory : GrayScaleKernelFactory = GrayScaleKernelFactory();
        for image in images {
            let grayScaleKernel : ComputeKernel = try await grayScaleFactory.createKernel(bufable: image, context: self.computeContext);
            let grayScaleImageResult : DeviceBufferResult = DeviceBufferResult();
            await grayScaleKernel.addObserver(grayScaleImageResult);
            grayScaleImages.append(grayScaleImageResult);
            grayScaleKernels.append(grayScaleKernel);
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: grayScaleKernels); // Will suspend here until completion

        // Collect the resident grayscale buffers and run the rest of the pipeline against them
        let grayscaleBuffers : [DeviceBuffer] = grayScaleImages.map { $0.buffer!; };
        return try await self.similarityMatrix(grayscaleBuffers: grayscaleBuffers);
    }

    /// Builds the ZNCC similarity matrix from pre-computed grayscale buffers (one per image, in order)
    /// This is the shared core of the similarity pipeline: everything except the grayscale stage
    /// - Parameters:
    ///     - grayscaleBuffers: The full-canvas grayscale of each image, already resident on the GPU
    /// - Returns: An NxN symmetric matrix; matrix[i][j] is the ZNCC of image i and image j (in [-1, 1])
    /// - Note: The grayscale buffers belong to the caller. This method reads them but never frees them,
    ///         so a coordinator can keep them alive for another analysis (e.g. sharpness)
    func similarityMatrix(grayscaleBuffers: [DeviceBuffer]) async throws -> [[Float]] {
        guard !grayscaleBuffers.isEmpty else { return []; }
        // Intermediate arrays (the mean-centred and squared results) stay resident on the GPU between stages:
        // each array kernel publishes its output as a DeviceBuffer, which the next factory reads directly.
        // Only the small scalars (means, sums) and the final matrix are read back to the CPU

        // The mean and both subtraction factories share one BufferCache. With resident DeviceBuffers this is no
        // longer for upload de-duplication (toMTLBuffer is a no-op passthrough now) -- it exists so the cache's
        // hold on each intermediate can be dropped at a stage boundary. Clearing it, together with releasing the
        // kernels that read/wrote that buffer, is what lets the GPU actually reclaim a spent intermediate
        let statsCache : BufferCache = BufferCache();

        // Find the mean in all grayscale images. Each grayscale DeviceBuffer is read straight off the GPU (no upload)
        let factoryMean : MeanValueFactory = MeanValueFactory(bufferCache: statsCache);
        var grayScaleMeanKernels : [ComputeKernel] = [];
        var grayScaleImageMeans : [FloatValueResult] = [];
        for grayscaleBuffer in grayscaleBuffers {
            let meanKernel : ComputeKernel = try await factoryMean.createKernel(bufable: grayscaleBuffer, context: self.computeContext);
            let meanValue : FloatValueResult = FloatValueResult();
            await meanKernel.addObserver(meanValue);
            grayScaleImageMeans.append(meanValue);
            grayScaleMeanKernels.append(meanKernel);
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: grayScaleMeanKernels);

        // Next find the subtraction of the grayscale values from the mean and also do the same but every result to the power of two
        let factorySubtractionOne = SubtractionFactory(bufferCache: statsCache);
        let factorySubtractionSqr = SubtractionFactory(bufferCache: statsCache);
        factorySubtractionSqr.setPowValue(value:2.0);
        var grayScaleSubtractKernels : [any ComputeKernel] = [];
        var grayScaleSubtractArrays : [DeviceBufferResult] = [];
        var grayScaleSubtractPow2Arrays : [DeviceBufferResult] = [];
        for (index,grayscaleBuffer) in grayscaleBuffers.enumerated() {
            let subtractResult : DeviceBufferResult = DeviceBufferResult();
            let subtractPow2Result : DeviceBufferResult = DeviceBufferResult();
            let grayScaleImageMean : Float = grayScaleImageMeans[index].result;
            factorySubtractionOne.setSubtractionValue(value: grayScaleImageMean);
            factorySubtractionSqr.setSubtractionValue(value: grayScaleImageMean);
            let subtractKernel : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayscaleBuffer, context: self.computeContext);
            await subtractKernel.addObserver(subtractResult);
            let subtractPow2Kernel : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: grayscaleBuffer, context: self.computeContext);
            await subtractPow2Kernel.addObserver(subtractPow2Result);
            grayScaleSubtractKernels.append(subtractKernel);
            grayScaleSubtractKernels.append(subtractPow2Kernel);
            grayScaleSubtractArrays.append(subtractResult);
            grayScaleSubtractPow2Arrays.append(subtractPow2Result);

        }
        try await MetalRunner.runCompute(from: self.computeContext, for: grayScaleSubtractKernels);

        // Early release: the subtraction kernels are spent, so drop them and the shared cache's hold on their
        // inputs. The grayscale buffers themselves belong to the caller and are intentionally left intact as
        // coordinator may still need them for another analysis. The subtraction outputs survive via
        // grayScaleSubtractArrays / grayScaleSubtractPow2Arrays
        grayScaleSubtractKernels.removeAll();
        await statsCache.clear();

        // Find the summation of the power of two results from before
        // Do this by finding the mean of the power 2 arrays, then multiply by their count
        var meanPow2Vals : [FloatValueResult] = [];
        var meanPow2Kernels : [ComputeKernel] = [];
        for pow2ImageValueArray in grayScaleSubtractPow2Arrays {
            let meanPow2Result : FloatValueResult = FloatValueResult();
            let meanPow2Kernel : ComputeKernel = try await factoryMean.createKernel(bufable: pow2ImageValueArray.buffer!, context: self.computeContext);
            await meanPow2Kernel.addObserver(meanPow2Result);
            meanPow2Vals.append(meanPow2Result);
            meanPow2Kernels.append(meanPow2Kernel);
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: meanPow2Kernels);
        var sumPow2 : [Float] = [];
        for (index,meanPow2) in meanPow2Vals.enumerated() {
            // Mean of the squared values * their count == the sum of squares (element count read from the buffer)
            let elementCount : Float = Float(try grayScaleSubtractPow2Arrays[index].buffer!.MTLBufferSize());
            sumPow2.append(meanPow2.result * elementCount);
        }

        // Early release: the squared arrays and the mean kernels that consumed them are spent (their sums are now
        // in sumPow2). Drop them and re-clear the shared cache so only the mean-centred buffers remain resident
        // for the dot stage
        grayScaleSubtractPow2Arrays.removeAll();
        meanPow2Kernels.removeAll();
        await statsCache.clear();

        // Compute every pairwise dot product in a single batched dispatch instead of separate DotProduct
        // kernels. The factory assembles the resident mean-centred buffers into one strided device buffer; the
        // batched kernel reduces each lower-triangle pair (row, col), returning one [Float] of results
        let imageCount : Int = grayScaleSubtractArrays.count;
        let subtractBuffers : [any MTBufable] = grayScaleSubtractArrays.map { $0.buffer! };
        let dotProductFactory : BatchedDotProductFactory = BatchedDotProductFactory();
        dotProductFactory.setImageArrays(subtractBuffers);
        let dotProductKernel : ComputeKernel = try await dotProductFactory.createKernel(bufable: subtractBuffers[0], context: self.computeContext);
        let dotProductResults : FloatArrayResult = FloatArrayResult();
        await dotProductKernel.addObserver(dotProductResults);
        try await MetalRunner.runCompute(from: self.computeContext, for: [dotProductKernel]);

        // Populate results into a full square matrix
        // Only the lower triangle (including the diagonal) was computed, so mirror each value across
        // the diagonal since ZNCC is symmetric (zncc(i,j) == zncc(j,i))
        let dotProducts : [Float] = dotProductResults.result;
        var znccResults : [[Float]] = Array(repeating: Array(repeating: Float(0), count: imageCount), count: imageCount);
        for row in 0..<imageCount {
            for col in 0...row {
                let dot : Float = dotProducts[BatchedDotProductFactory.pairIndex(row: row, col: col)];
                let zncc : Float = dot / (sumPow2[row] * sumPow2[col]).squareRoot();
                znccResults[row][col] = zncc;
                znccResults[col][row] = zncc; // Mirror across the diagonal (no-op when row == col)
            }
        }
        return znccResults;
    }
}

