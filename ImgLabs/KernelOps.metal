//
//  KernelOps.metal
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-05.
//
#include <metal_stdlib>
// Define offsets per pixel
constant uint8_t ALPHA_IDX = 3;
constant uint8_t RED_IDX = 0;
constant uint8_t BLUE_IDX = 1;
constant uint8_t GREEN_IDX = 2;
constant uint8_t MAX_COLOR_VAL = 255;

// Metal requires that kernel input arguments must have their locations specified (so the GPU doesn't need to waste time finding them in memory)
// For arguments, this is known as a argument buffer table ([buffer(0)] indicates look in slot 0 of the table)
// [[]] indicates to the compiler that inside is some attribute that it needs to handle, outer brackets signal the compiler, inner contains the instructions (separated by a comma)
// constant and device keywords specify to Metal the memory space of these variables
// device means read and write on GPU, constant means read only (which is faster) -- this is different than const!
// [[thread_position_in_grid]] -- For this index variable, populate it with whatever unique thread the GPU will use
kernel void multiply(device uchar* data [[buffer(0)]], constant uchar* multiplier [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    data[index] = data[index] * *multiplier; // Each thread is responsible for one element
}

/*
 Takes R,G,B values in separate arrays, applies weights to produce a grayscale number, asumes 8-bits per channel
 The following formula is performed per pixel: Y = RED_WEIGHT*R + GREEN_WEIGHT*G + BLUE_WEIGHT*B
 Alpha channel is blended into a programmable background color, defined by BACK_RED, BACK_GREEN, BACK_BLUE
 Thus, BlendedChannel = (Alpha * Foreground) + ((1 - Alpha) * Background) where foreground and background are a given channel (like Red)
 
 rgbaArray is a 1D array where each set of 4 elements makes up RGBA values (each element is 1 byte) for one pixel
 */
kernel void convertToGrayScale(constant uchar* rgbaArray [[buffer(0)]], constant float3* backgroundColor [[buffer(1)]], constant float3* rgbWeights [[buffer(2)]], device float* grayScaleResult [[buffer(3)]], uint32_t threadId [[thread_position_in_grid]]) {
    // Every thread will be responsible for one pixel
    uint32_t baseIdx = threadId * 4; // Calculate base offset for given pixel
    // Normalize alpha value between 0 and 1
    float normAlpha = ((float) rgbaArray[baseIdx+ALPHA_IDX]) / MAX_COLOR_VAL;
    float3 colorVals = float3((float)rgbaArray[baseIdx+RED_IDX],(float)rgbaArray[baseIdx+BLUE_IDX], (float)rgbaArray[baseIdx+GREEN_IDX]);
    float3 blendedChannels = (colorVals * normAlpha) + (*backgroundColor * (1.0f - normAlpha)); // Metal automatically recognizes to multiply vector by scalar
    // A*B + C*D + E*F
    grayScaleResult[threadId] = metal::dot(blendedChannels,*rgbWeights);
}

/*
 Calculates the sum of all elements within an array
 If there are more threads than the number of elements in the array, out of bounds threads will use a value of 0 to add
 */
kernel void calculateArraySum(
    device const float* values          [[buffer(0)]],
    constant uint32_t& maxLength       [[buffer(1)]],
    device metal::atomic_float* globalSum [[buffer(2)]],
    threadgroup float* localSharedMem  [[threadgroup(0)]], // Shared memory allocation per thread group
    uint32_t threadId                  [[thread_position_in_grid]],
    uint32_t localId                   [[thread_position_in_threadgroup]],
    uint32_t groupSize                 [[threads_per_threadgroup]])
{
    // 1. Load data from slow global memory into fast local shared memory
    // If the thread is out of bounds, load 0.0 so it doesn't affect the sum
    localSharedMem[localId] = (threadId < maxLength) ? values[threadId] : 0.0f;
    
    // Synchronize to ensure all threads in this group finished writing to shared memory
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    
    // 2. Perform tree-based reduction inside the local threadgroup
    for (uint32_t stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            localSharedMem[localId] += localSharedMem[localId + stride];
        }
        // Sync after every step of the tree reduction loop
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    
    // 3. Only Thread 0 of this group writes the final group subtotal globally
    if (localId == 0) {
        metal::atomic_fetch_add_explicit(globalSum, localSharedMem[0], metal::memory_order::memory_order_relaxed);
    }
}

