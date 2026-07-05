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
    /// - Returns: A floating point value between 0 and 1 (A higher value means more similarity between the two images)
    func howSimilar(image1: ImageData, image2: ImageData) async throws -> Float {
        // If the exact same images are used in ZNCC, the result should be 1
        let factoryGrayScale = GrayScaleKernelFactory();
        let factoryMean = MeanValueFactory();
        let factorySubtractionOne = SubtractionFactory();
        let factorySubtractionSqr = SubtractionFactory();
        factorySubtractionSqr.setPowValue(value:2.0);
        let factoryDot : DotProductFactory = DotProductFactory();
        
        print("Computing grayscale...");
        // First compute grayscale of both images
        let grayScaleKernelImg1 : ComputeKernel = try await factoryGrayScale.createKernel(bufable: image1, context: self.computeContext);
        let grayScaleValImg1 : FloatArrayResult = FloatArrayResult();
        await grayScaleKernelImg1.addObserver(grayScaleValImg1);
        let grayScaleKernelImg2 : ComputeKernel = try await factoryGrayScale.createKernel(bufable: image2, context: self.computeContext);
        let grayScaleValImg2 : FloatArrayResult = FloatArrayResult();
        await grayScaleKernelImg2.addObserver(grayScaleValImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [grayScaleKernelImg1,grayScaleKernelImg2]);
        
        // Next find the mean value of the grayscale
        print("Finding mean of grayscale...");
        let meanKernelImg1 : ComputeKernel = try await factoryMean.createKernel(bufable: grayScaleValImg1, context: self.computeContext);
        let meanValImg1 : FloatValueResult = FloatValueResult();
        await meanKernelImg1.addObserver(meanValImg1);
        let meanKernelImg2 : ComputeKernel = try await factoryMean.createKernel(bufable: grayScaleValImg2, context: self.computeContext);
        let meanValImg2 : FloatValueResult = FloatValueResult();
        await meanKernelImg2.addObserver(meanValImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [meanKernelImg1,meanKernelImg2]);
        
        // Next find the subtraction of the grayscale values from the mean
        print("Finding subtraction of grayscale from mean...");
        let subValsImg1 : FloatArrayResult = FloatArrayResult();
        factorySubtractionOne.setSubtractionValue(value: meanValImg1.result);
        let subKernelImg1 : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayScaleValImg1, context: self.computeContext);
        await subKernelImg1.addObserver(subValsImg1);
        let subValsImg2 : FloatArrayResult = FloatArrayResult();
        factorySubtractionOne.setSubtractionValue(value: meanValImg2.result);
        let subKernelImg2 : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayScaleValImg2, context: self.computeContext);
        await subKernelImg2.addObserver(subValsImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [subKernelImg1,subKernelImg2]);
        
        // Do the same as before but with the subtraction result being put to a power of 2
        print("Finding subtraction of grayscale from mean power of 2...");
        let subSqrValsImg1 : FloatArrayResult = FloatArrayResult();
        // subVals already has the mean removed; only square it here (subtract 0), otherwise the mean is removed twice
        factorySubtractionSqr.setSubtractionValue(value: 0.0);
        let subSqrKernelImg1 : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: subValsImg1, context: self.computeContext);
        await subSqrKernelImg1.addObserver(subSqrValsImg1);
        let subSqrValsImg2 : FloatArrayResult = FloatArrayResult();
        // subVals already has the mean removed; only square it here (subtract 0), otherwise the mean is removed twice
        let subSqrKernelImg2 : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: subValsImg2, context: self.computeContext);
        await subSqrKernelImg2.addObserver(subSqrValsImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [subSqrKernelImg1,subSqrKernelImg2]);
        
        // Find the summation of squared results from before
        print("Finding summation of squared results...");
        let sumSqrValsImg1 : FloatValueResult = FloatValueResult();
        let sumSqrKernelImg1 : ComputeKernel = try await factoryMean.createKernel(bufable: subSqrValsImg1, context: self.computeContext);
        await sumSqrKernelImg1.addObserver(sumSqrValsImg1);
        let sumSqrValsImg2 : FloatValueResult = FloatValueResult();
        let sumSqrKernelImg2 : ComputeKernel = try await factoryMean.createKernel(bufable: subSqrValsImg2, context: self.computeContext);
        await sumSqrKernelImg2.addObserver(sumSqrValsImg2);
        try await MetalRunner.runCompute(from: self.computeContext, for: [sumSqrKernelImg1,sumSqrKernelImg2]);
        let sumSqrImg1 : Float = sumSqrValsImg1.result * Float(subSqrValsImg1.result.count); // Mean so multiply by number of vals
        let sumSqrImg2 : Float = sumSqrValsImg2.result * Float(subSqrValsImg2.result.count); // Mean so multiply by number of vals
        
        // Find the dot product of the non squared results
        print("Finding dot product of grayscale subtraction from mean...");
        let dotVal : FloatValueResult = FloatValueResult();
        factoryDot.setArr2(bufable: subValsImg2);
        let dotKernel : ComputeKernel = try await factoryDot.createKernel(bufable: subValsImg1, context: self.computeContext);
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
        // First compute the grayscale of every image
        var grayScaleKernels : [ComputeKernel] = [];
        var grayScaleImages : [FloatArrayResult] = [];
        let grayScaleFactory : GrayScaleKernelFactory = GrayScaleKernelFactory();
        for image in images {
            let grayScaleKernel : ComputeKernel = try await grayScaleFactory.createKernel(bufable: image, context: self.computeContext);
            let grayScaleImageResult : FloatArrayResult = FloatArrayResult();
            await grayScaleKernel.addObserver(grayScaleImageResult);
            grayScaleImages.append(grayScaleImageResult);
            grayScaleKernels.append(grayScaleKernel);
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: grayScaleKernels); // Will suspend here until completion
        
        // Each grayscale array is fed to the mean factory and BOTH subtraction factories, so share one
        // cache across them: every grayscale array is uploaded to the GPU once instead of three times
        let statsCache : BufferCache = BufferCache();

        // Find the mean in all grayscale images
        let factoryMean : MeanValueFactory = MeanValueFactory(bufferCache: statsCache);
        var grayScaleMeanKernels : [ComputeKernel] = [];
        var grayScaleImageMeans : [FloatValueResult] = [];
        for grayScaleImage in grayScaleImages {
            let meanKernel : ComputeKernel = try await factoryMean.createKernel(bufable: grayScaleImage, context: self.computeContext);
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
        var grayScaleSubtractArrays : [FloatArrayResult] = [];
        var grayScaleSubtractPow2Arrays : [FloatArrayResult] = [];
        for (index,grayScaleImage) in grayScaleImages.enumerated() {
            let subtractResult : FloatArrayResult = FloatArrayResult();
            let subtractPow2Result : FloatArrayResult = FloatArrayResult();
            let grayScaleImageMean : Float = grayScaleImageMeans[index].result;
            factorySubtractionOne.setSubtractionValue(value: grayScaleImageMean);
            factorySubtractionSqr.setSubtractionValue(value: grayScaleImageMean);
            let subtractKernel : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayScaleImage, context: self.computeContext);
            await subtractKernel.addObserver(subtractResult);
            let subtractPow2Kernel : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: grayScaleImage, context: self.computeContext);
            await subtractPow2Kernel.addObserver(subtractPow2Result);
            grayScaleSubtractKernels.append(subtractKernel);
            grayScaleSubtractKernels.append(subtractPow2Kernel);
            grayScaleSubtractArrays.append(subtractResult);
            grayScaleSubtractPow2Arrays.append(subtractPow2Result);
            
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: grayScaleSubtractKernels);
        
        // Find the summation of the power of two results from before
        // Do this by finding the mean of the power 2 arrays, then multiply by their count
        var meanPow2Vals : [FloatValueResult] = [];
        var meanPow2Kernels : [ComputeKernel] = [];
        for pow2ImageValueArray in grayScaleSubtractPow2Arrays {
            let meanPow2Result : FloatValueResult = FloatValueResult();
            let meanPow2Kernel : ComputeKernel = try await factoryMean.createKernel(bufable: pow2ImageValueArray, context: self.computeContext);
            await meanPow2Kernel.addObserver(meanPow2Result);
            meanPow2Vals.append(meanPow2Result);
            meanPow2Kernels.append(meanPow2Kernel);
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: meanPow2Kernels);
        var sumPow2 : [Float] = [];
        for (index,meanPow2) in meanPow2Vals.enumerated() {
            sumPow2.append(meanPow2.result * Float(grayScaleSubtractPow2Arrays[index].result.count));
        }
       
        // Compute the dot product of each non power of two subtraction from mean results with each other result
        // Since the dot of image[0]/image[1] is the same as image[1][0] avoid those repeat computations
        let dotProductFactory : DotProductFactory = DotProductFactory();
        var dotProductResults : [[FloatValueResult]] = [];
        var dotProductKernels : [ComputeKernel] = [];
        for (index,imgOneSubArray) in grayScaleSubtractArrays.enumerated() {
            // Compute dot product of image 0...index
            var dotProductResultsForIndex : [FloatValueResult] = [];
            for dotProd2Index in 0...index {
                let dotProduct : FloatValueResult = FloatValueResult();
                dotProductFactory.setArr2(bufable: grayScaleSubtractArrays[dotProd2Index]);
                let dotProductKernel : ComputeKernel = try await dotProductFactory.createKernel(bufable: imgOneSubArray, context: self.computeContext);
                await dotProductKernel.addObserver(dotProduct);
                dotProductKernels.append(dotProductKernel);
                dotProductResultsForIndex.append(dotProduct);
            }
            dotProductResults.append(dotProductResultsForIndex);
        }
        try await MetalRunner.runCompute(from: self.computeContext, for: dotProductKernels);
        
        // Populate results into a full square matrix
        // Only the lower triangle (including the diagonal) was computed, so mirror each value across
        // the diagonal since ZNCC is symmetric (zncc(i,j) == zncc(j,i))
        let matrixSize : Int = dotProductResults.count;
        var znccResults : [[Float]] = Array(repeating: Array(repeating: Float(0), count: matrixSize), count: matrixSize);

        for (img1Idx,dotProdResultsForImage) in dotProductResults.enumerated() {
            for (img2Idx,dotProdResult) in dotProdResultsForImage.enumerated() {
                let zncc : Float = dotProdResult.result / (sumPow2[img1Idx] * sumPow2[img2Idx]).squareRoot();
                znccResults[img1Idx][img2Idx] = zncc;
                znccResults[img2Idx][img1Idx] = zncc; // Mirror across the diagonal (no-op when img1Idx == img2Idx)
            }
        }
        return znccResults;
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

    private class FloatValueResult : ResultObserver<Float> {
        var result: Float = 0;
        func update(with: Float) {
            self.result = with;
        }
    }
    
}

