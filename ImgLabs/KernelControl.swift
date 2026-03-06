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
    
    init(device : MTLDevice) {
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
