//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI
import PhotosUI

// View is the fundamental building block of SwiftUI
// It is declarative, i.e., the expected result but supports imperative code
struct ContentView: View { // This a custom view, it conatains a body
    @State private var loadedImageList: [ImageData] = [];
    @State private var isProcessing : Bool = false;
    @State private var numberOfImages: Int = 0;
    
    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12)
    ]
    
    
    func openMacFinder() { // mutating -- method is allowed to change properties of this struct (number of images variable in this case)
        let openPanel = NSOpenPanel()
        openPanel.title = "Choose an Image"
        openPanel.showsHiddenFiles = false
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        openPanel.allowsMultipleSelection = true
        openPanel.allowedContentTypes = [.image] // Filters for images

        // Open the native Finder sheet
        openPanel.begin { response in
            if response == .OK {
                // Animate the entrance of the loading bar smoothly
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self.numberOfImages = openPanel.urls.count
                    self.isProcessing = true
                }
                Task { // Informs swift that items inside this block can be ran on another CPU thread
                    for selectedURL in openPanel.urls { // Multiple images
                        if let nsImage = NSImage(contentsOf: selectedURL),
                           let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                            let newImage = ImageData(img: cgImage);
                            await MainActor.run { self.loadedImageList.append(newImage); } // Context switch back to main ui thread to append, thus forcing view update given the array is a state variable
                            print("Loaded an Image!\n");
                        }
                    }
                    await MainActor.run { // Update main thread UI again to indicate complete
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                            self.isProcessing = false
                        }
                    }
                }
            }
        }
    }
    
    var body: some View { // some means that body returns an object which conforms to the View type (i.e., don't need to specify exactly what is returned)
        Text("ImgLabs").font(.largeTitle.bold()).padding()
        VStack {
            if !self.isProcessing && self.loadedImageList.isEmpty {
                ZStack { // Z direction stack (i.e., stack items on top of eachother)
                    RoundedRectangle(cornerRadius: 5)
                        .fill(Color.gray).padding(10)
                
                    Button("Import Image") {
                        openMacFinder()
                    }
                    .shadow(color: .black.opacity(0.7), radius: 20)
                    .shadow(color: .black.opacity(0.7), radius: 20)
                }.transition(.slide)
            } else if self.isProcessing {
                ZStack { // Z direction stack (i.e., stack items on top of eachother)
                    RoundedRectangle(cornerRadius: 5)
                        .fill(Color.clear).padding(10)
                
                    // Show progress bar with images that are loading
                    ProgressView("Loaded \(self.loadedImageList.count) out of \(self.numberOfImages)", value: Float(self.loadedImageList.count), total: Float(self.numberOfImages)).padding(25)
                }
            } else {
                // Loaded image list is not empty, display list
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(loadedImageList) { item in // Item is of type ImageData
                            // Render the thumbnail (Convert CGImage/NSImage to SwiftUI Image)
                            Image(nsImage: NSImage(cgImage: item.getCGImage()!, size: .zero))
                                .resizable()
                                .scaledToFit()
                                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 120, maxHeight: 120)
                                .clipped()
                                .cornerRadius(8)
                                .shadow(radius: 2)
                                .onTapGesture {
                                    Task {
                                        print("Tapped on an item!");
                                        let computeContext : MetalComputeContext = MetalComputeContext()!;
                                        
                                        
                                        // If the exact same images are used in ZNCC, the result should be 1
                                        let factoryGrayScale = GrayScaleKernelFactory();
                                        let factoryMean = MeanValueFactory();
                                        let factorySubtractionOne = SubtractionFactory();
                                        let factorySubtractionSqr = SubtractionFactory();
                                        factorySubtractionSqr.setPowValue(value:2.0);
                                        let factoryDot : DotProductFactory = DotProductFactory();
                                        
                                        print("Computing grayscale...");
                                        // First compute grayscale
                                        let grayScaleKernel : ComputeKernel = try await factoryGrayScale.createKernel(bufable: item, context: computeContext);
                                        let grayScaleVal : FloatArrayResult = FloatArrayResult();
                                        await grayScaleKernel.addObserver(grayScaleVal);
                                        try await MetalRunner.runCompute(from: computeContext, for: [grayScaleKernel]);
                                        // Next find the mean value of the grayscale
                                        print("Finding mean of grayscale...");
                                        let meanKernel : ComputeKernel = try await factoryMean.createKernel(bufable: grayScaleVal, context: computeContext);
                                        let meanVal : FloatValueResult = FloatValueResult();
                                        await meanKernel.addObserver(meanVal);
                                        try await MetalRunner.runCompute(from: computeContext, for: [meanKernel]);
                                        // Next find the subtraction of the grayscale values from the mean
                                        print("Finding subtraction of grayscale from mean...");
                                        let subVals : FloatArrayResult = FloatArrayResult();
                                        factorySubtractionOne.setSubtractionValue(value: meanVal.result);
                                        let subKernel : ComputeKernel = try await factorySubtractionOne.createKernel(bufable: grayScaleVal, context: computeContext);
                                        await subKernel.addObserver(subVals);
                                        try await MetalRunner.runCompute(from: computeContext, for: [subKernel]);
                                        // Do the same as before but with the subtraction result being put to a power of 2
                                        print("Finding subtraction of grayscale from mean power of 2...");
                                        let subSqrVals : FloatArrayResult = FloatArrayResult();
                                        // subVals already has the mean removed; only square it here (subtract 0), otherwise the mean is removed twice
                                        factorySubtractionSqr.setSubtractionValue(value: 0.0);
                                        let subSqrKernel : ComputeKernel = try await factorySubtractionSqr.createKernel(bufable: subVals, context: computeContext);
                                        await subSqrKernel.addObserver(subSqrVals);
                                        try await MetalRunner.runCompute(from: computeContext, for: [subSqrKernel]);
                                        // Find the summation of squared results from before
                                        print("Finding summation of squared results...");
                                        let sumSqrVals : FloatValueResult = FloatValueResult();
                                        let sumSqrKernel : ComputeKernel = try await factoryMean.createKernel(bufable: subSqrVals, context: computeContext);
                                        await sumSqrKernel.addObserver(sumSqrVals);
                                        try await MetalRunner.runCompute(from: computeContext, for: [sumSqrKernel]);
                                        let sumSqr : Float = sumSqrVals.result * Float(subSqrVals.result.count); // Mean so multiply by number of vals
                                        // Find the dot product of the non squared results
                                        print("Finding dot product of grayscale subtraction from mean...");
                                        let dotVal : FloatValueResult = FloatValueResult();
                                        factoryDot.setArr2(bufable: subVals);
                                        let dotKernel : ComputeKernel = try await factoryDot.createKernel(bufable: subVals, context: computeContext);
                                        await dotKernel.addObserver(dotVal);
                                        print("Created kernel, now running...");
                                        try await MetalRunner.runCompute(from: computeContext, for: [dotKernel]);
                                        print("Dot product value is \(dotVal.result)")
                                        // Print the ZNCC result
                                        let zncc : Float = dotVal.result / (sumSqr * sumSqr).squareRoot();
                                        print("ZNCC is \(zncc)");
                                    }
                                }
                        }
                    }.padding()
                }.transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }.animation(.spring(response: 0.4, dampingFraction: 0.8), value: self.isProcessing)
    }
}

#Preview {
    ContentView()
}

class FloatArrayResult : ResultObserver<[Float]>, MTBufable {
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
        print("Update happened!");
        self.result = with;
    }
}

class FloatValueResult : ResultObserver<Float> {
    var result: Float = 0;
    func update(with: Float) {
        print("Update happened!");
        self.result = with;
    }
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
