//
//  KernelControl.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-05.
//

import Metal

class KernelControl {
    
    private var dev : MTLDevice; // No need for ? because it's assignment is forced on creation
    private static let MULTIPLY_STR : String = "multiply";
    
    // Data houses an array of UInt8 values, dataSize denotes how long that is and multiplier will be multiply all values in data by the given multiplier
    // inout means pass by reference
    init(device : MTLDevice, data : inout [UInt8], dataSize : Int, multiplier : UInt8) {
        self.dev = device;
        
        if let lib : MTLLibrary = self.dev.makeDefaultLibrary() { // All .metal files are compiled and their kernel functions are added to the default library
            // Ask the library for the function multiply()
            if let kernel : MTLFunction = lib.makeFunction(name: KernelControl.MULTIPLY_STR) { // Proxy function, not the actual one, needs to be passed to a pipeline
                // Create the pipeline (this is how commands are sent to the GPU) for the given device
                do { // Basically try/catch block
                    let pipeline : MTLComputePipelineState = try self.dev.makeComputePipelineState(function: kernel); // If failure, catch block is executed
                    // Creating a pipeline with a single compute function
                    if let cmdQueue : MTLCommandQueue = self.dev.makeCommandQueue() { // Get command queue
                        // Allocate memory on GPU for task
                        // .storageModeShared means both the CPU and GPU can access this data
                        var multiplierCpy : UInt8 = multiplier; // Can't pass immutable value to GPU buffer, thus make it mutable
                        if let dataBuffer : MTLBuffer = self.dev.makeBuffer(bytes: data, length: Int(dataSize), options: [.storageModeShared]), let multiplyBuf : MTLBuffer = self.dev.makeBuffer(bytes: &multiplierCpy, length: MemoryLayout.size(ofValue: multiplierCpy), options: [.storageModeShared]) {
                            // Now create a command buffer and command encoder (i.e., to write instructions onto the command buffer)
                            // Create a compute command encoder because we will be processing data on the compute engine
                            if let cmdBuffer : MTLCommandBuffer = cmdQueue.makeCommandBuffer(), let cmdEncoder : MTLComputeCommandEncoder = cmdBuffer.makeComputeCommandEncoder() {
                                cmdEncoder.setComputePipelineState(pipeline);
                                cmdEncoder.setBuffer(dataBuffer, offset: 0, index: 0); // This is buffer(0)
                                cmdEncoder.setBuffer(multiplyBuf, offset: 0, index: 1); // buffer(1)
                                
                                // Tell the GPU how many threads to use (like HIP/CUDA this is specified as a grid)
                                // In this case, we will have a grid (WxHxD) of dataSizex1x1 (i.e., just map threads directly like the array)
                                let threadsPerGrid = MTLSize(width: Int(dataSize), height: 1, depth: 1);
                                // Within this grid, we can specify threadGroups which denote how many threads can share local memory
                                // Note that internally, the GPU will run 32 threads at a time, but it won't move any set of 32 threads within the group down to the next line of code until all threads within the group have made it there
                                // Here we will just use 32
                                // Note also that if the threadGroupSize is bigger than the grid, the extra threads will be deactivated
                                let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1);
                                // Encode this to tell the GPU this will be how many threads will be used
                                cmdEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadGroupSize);
                                // End the encoding
                                cmdEncoder.endEncoding();
                                // Finally commit to the GPU...the kernel is now running!
                                cmdBuffer.commit();
                                cmdBuffer.waitUntilCompleted();
                                // Copy the result into data
                                // Access the raw memory pointer from the buffer
                                let rawPointer : UnsafeMutableRawPointer = dataBuffer.contents();
                                // Bind the raw pointer to the Swift type (UInt8 for uchar)
                                let typedPointer : UnsafeMutablePointer<UInt8> = rawPointer.bindMemory(to: UInt8.self, capacity: Int(dataSize));
                                // Copy the data out into a standard, safe Swift array
                                let resultsArray = Array(UnsafeBufferPointer(start: typedPointer, count: Int(dataSize)));
                                for (i, result) in resultsArray.enumerated() {
                                    data[i] = result;
                                }
                            }
                        }
                    }
                } catch (let e) {
                    print("Error creating pipeline state: \(e)");
                }
            }
        } else {
            // TODO, handle error case
            return;
        }
        
        
    }
}
