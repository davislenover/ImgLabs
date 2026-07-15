//
//  ImgMath.metal
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//

#include <metal_stdlib>
// Define offsets per pixel
constant uint8_t ALPHA_IDX = 3;
constant uint8_t RED_IDX = 0;
constant uint8_t BLUE_IDX = 1;
constant uint8_t GREEN_IDX = 2;
constant uint8_t MAX_COLOR_VAL = 255;
/*
 Takes R,G,B values in separate arrays, applies weights to produce a grayscale number, asumes 8-bits per channel
 The following formula is performed per pixel: Y = RED_WEIGHT*R + GREEN_WEIGHT*G + BLUE_WEIGHT*B
 Alpha channel is blended into a programmable background color, defined by BACK_RED, BACK_GREEN, BACK_BLUE
 Thus, BlendedChannel = Foreground + ((1 - Alpha) * Background) where foreground and background are a given channel (like Red)
 
 rgbaArray is a 1D array where each set of 4 elements makes up RGBA values (each element is 1 byte) for one pixel, assumes alpha is premultiplied to RGB values (thus, don't need to re-multiply alpha to color channels)
 */
kernel void convertToGrayScale(constant uint8_t* rgbaArray [[buffer(0)]], constant float4* backgroundColor [[buffer(1)]], constant float4* rgbWeights [[buffer(2)]], device float* grayScaleResult [[buffer(3)]], constant uint32_t* numOfPixels [[buffer(4)]], uint32_t threadId [[thread_position_in_grid]]) {
    if (threadId >= *numOfPixels) {return;}
    // Every thread will be responsible for one pixel
    uint32_t baseIdx = threadId * 4; // Calculate base offset for given pixel
    // Normalize alpha value between 0 and 1
    float normAlpha = ((float) rgbaArray[baseIdx+ALPHA_IDX]) / MAX_COLOR_VAL;
    float3 colorVals = float3((float)rgbaArray[baseIdx+RED_IDX],(float)rgbaArray[baseIdx+GREEN_IDX], (float)rgbaArray[baseIdx+BLUE_IDX]);
    // rgb extracts first 3 values, input as 4 for memory alignment (GPU is 16-byte aligned -- float4 - 4 floats of 4 bytes, no padding needed)
    float3 blendedChannels = colorVals + (backgroundColor->rgb * (1.0f - normAlpha)); // Metal automatically recognizes to multiply vector by scalar
    // A*B + C*D + E*F
    grayScaleResult[threadId] = metal::dot(blendedChannels,rgbWeights->rgb);
}

/*
 Find discrete cosine transform of a square 2D 32x32 matrix
 Utilizes a pre computed matrix to perform a dot product twice on the 3D input matrix where each z dimension is one set
 Threadgroup contains 32x8 threads (images processed are 32x32, frequency is 0-7)
 - Parameters:
    - preConstMtx: The pre-compute matrix of constants from preConstMtx[u][x] = α(u) · cos((2x+1)·u·π / 2N) (strided)
    where u is the frequency component, N is the dimension of the input matrix, x is between 0...N-1
    - inputMtx: The input 3D square matrix to take the DCT of (strided as a 1D matrix)
    - numRowsAndColumns: The X and Y dimensions of the inputMtx
    - depth: The number of sets within the inputMtx (i.e., Z axis length)
    - maxFreq: The maximum frequency
    - result: A 3D matrix containing the 2D cosine frequency contents per set (strided, 64 floats per set -- result + imageIndex * 64)
 */
kernel void convertDCT(constant float* preConstMtx [[buffer(0)]],
                       device const float* inputMtx [[buffer(1)]],
                       constant uint32_t& numRowsAndColumns [[buffer(2)]],
                       constant uint32_t& depth [[buffer(3)]],
                       constant uint32_t& maxFreq [[buffer(4)]],
                       device float* result [[buffer(5)]],
                       uint2 groupId [[threadgroup_position_in_grid]],
                       uint2 localId [[thread_position_in_threadgroup]]) {
    // Assign each thread (u,j) -- each thread will iterate over x, summing a dot product
    // Each threadgroup will be responsible for one Z axis (set)
    // Pass 1: T[u][j] = Σx C[u][x]·f[x][j]  (DCT down each column)
    // Pass 2: F[u][v] = Σy T[u][y]·C[v][y]
    uint32_t imageIndex = groupId.x;   // grid is (numImages, 1) threadgroups
    if (imageIndex >= depth) { return; }
    uint32_t j = localId.x;            // 0...31
    uint32_t u = localId.y;            // 0...7
    const uint32_t N = numRowsAndColumns; // 32
    threadgroup float T[8][32]; // Shared per threadgroup matrix
    
    device const float* img = inputMtx + imageIndex * N * N; // Selects the current Z axis
    // Pass 1: T[u][j] = Σx C[u][x]·f[x][j]  (DCT down each column)
    float sum = 0.0f;
    for (uint32_t x = 0; x < N; x++) {
        sum += preConstMtx[u * N + x] * img[x * N + j];
    }
    T[u][j] = sum;

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Pass 2: F[u][v] = Σy T[u][y]·C[v][y]  — only 8×8 threads needed as u and v are frequency from 0...7
    if (j < maxFreq) {
        const uint32_t v = j;                   // reuse x-lane as frequency v
        float acc = 0.0f;
        for (uint32_t y = 0; y < N; y++) {
            acc += T[u][y] * preConstMtx[v * N + y];
        }
        result[imageIndex * maxFreq * maxFreq + u * maxFreq + v] = acc;
    }
}

/*
 Performs convolution of an (assumed to be) 3x3 matrix over a grayscale (one element per pixel) image set (a 3D strided matrix where each 2D matrix is a list of values)
  - Parameters:
    - convolMtx: A 2D 3x3 matrix that will "slide" over each set of 2D matricies within inputMtx
    - inputMtx: The input 3D strided matrix, with 2D matricies the convolMtx will "slide" over
    - numRows: The row dimensions of each 2D matrix in inputMtx
    - numColumns: The column dimensions of each 2D matrix in inputMtx
    - depth: The number of 2D matricies within inputMtx
    - result: A 3D strided matrix where each 2D matrix is the result of the convolution
 The thread grid created should match a thread to a specific X,Y pixel in each image where X is multiplied by the number of images -- (i.e., X,Y,Z -> X*Z,Y,1)
 */
kernel void convoluteImage(constant float* convolMtx [[buffer(0)]],
                           device const float* inputMtx [[buffer(1)]],
                           constant uint32_t& numRows [[buffer(2)]],
                           constant uint32_t& numColumns [[buffer(3)]],
                           constant uint32_t& depth [[buffer(4)]],
                           device float* result [[buffer(5)]],
                           uint2 threadId [[thread_position_in_grid]]) {
    
    // Each thread will be responsible for one pixel position in one image (and thus, one pixel position in result)
    // Grid is "3D" -- 2D but with a multiplied X from Z axis (which is very lage), better for cache
    const uint32_t imgIdx = threadId.x / numColumns; // Z
    const uint32_t imgPixelX = (threadId.x % numColumns);
    const uint32_t imgPixelY = threadId.y;
    
    if (imgIdx >= depth || imgPixelX >= numColumns || imgPixelY >= numRows) {
        return;
    }
    
    // Get corresponding pixel offset
    const uint64_t sliceStrideOffset = (uint64_t)imgIdx * numRows * numColumns;
    const uint64_t strideResultOffset = sliceStrideOffset + (imgPixelY * numColumns) + imgPixelX;
    
    // Get the values of all values surrounding the center pixel (threadPixel), multiply by their corresponding value in the convolMtx, add to total
    // Deal with edges by clamping the value to 0
    float conVolPixelResult = 0;
    // Loop through the 3x3 spatial coordinate offsets
    for (int yOffset = -1; yOffset <= 1; yOffset++) {
        for (int xOffset = -1; xOffset <= 1; xOffset++) {
                
            const int targetX = (int)imgPixelX + xOffset;
            const int targetY = (int)imgPixelY + yOffset;
            
            // Center midpoint (xOffset=0, yOffset=0) evaluates to Index 4
            const uint8_t conVolMtxOffset = ((yOffset + 1) * 3) + (xOffset + 1);
            const float conVolValue = convolMtx[conVolMtxOffset];
                
            // If target is out of bounds, treat its value as 0 and continue to next pass
            if (targetX < 0 || targetX >= (int)numColumns || targetY < 0 || targetY >= (int)numRows) {
                continue;
            }
                
            // Calculate the source thread input offset index
            const uint64_t strideOffset = sliceStrideOffset + (targetY * numColumns) + targetX;
            const float pixelValue = inputMtx[strideOffset];
                
            conVolPixelResult += (pixelValue * conVolValue);
        }
    }
    result[strideResultOffset] = conVolPixelResult;
}

/*
 Accumulates the sum and sum-of-squares of each 2D matrix within a 3D strided matrix (one pair of totals per z-axis set)
 The final variance = (Σx²/N) - (Σx/N)² should be computed on the CPU once the totals are ready
 The grid is 2D (groupsPerSlice, depth): the Y axis selects the set, the X axis selects which cooperating threadgroup within that set
 Each threadgroup reduces its slice of the set in shared memory, then thread 0 atomically adds it's partials into that set's accumulator
 groupSize (threads per threadgroup) should be a power of two (for the tree reduction) and as large as the device allows
 - Parameters:
    - inputMtx: The input 3D strided matrix, with 2D matricies (sets) laid back-to-back (each elemsPerSet floats)
    - elemsPerSet: The number of elements in each 2D matrix (i.e., numRows * numColumns)
    - depth: The number of 2D matricies within inputMtx
    - groupsPerSlice: The number of threadgroups cooperating on each set (the grid's X dimension)
    - accum: Two floats per set laid back-to-back accum[2*set] = Σx, accum[2*set+1] = Σx² (must start zeroed)
 */
kernel void calculateVariance(
    device const float* inputMtx        [[buffer(0)]],
    constant uint32_t& elemsPerSet      [[buffer(1)]],
    constant uint32_t& depth            [[buffer(2)]],
    constant uint32_t& groupsPerSlice   [[buffer(3)]],
    device metal::atomic_float* accum   [[buffer(4)]], // (Σx, Σx²) per set, accumulated across cooperating threadgroups
    threadgroup float2* localSharedMem  [[threadgroup(0)]], // (Σx, Σx²) partial per thread
    uint2 groupId                       [[threadgroup_position_in_grid]],
    uint2 localIdVec                    [[thread_position_in_threadgroup]],
    uint2 groupSizeVec                  [[threads_per_threadgroup]])
{
    // The threadgroup is 1D (height 1), so only the x components carry the local index and group size
    const uint32_t localId = localIdVec.x;
    const uint32_t groupSize = groupSizeVec.x;

    const uint32_t sliceIdx = groupId.y; // Which set (2D matrix) this threadgroup helps reduce
    if (sliceIdx >= depth) { return; }
    const uint32_t chunkIdx = groupId.x; // Which cooperating threadgroup within the set

    device const float* set = inputMtx + (uint64_t)sliceIdx * elemsPerSet;

    // All threadgroups for this set together stride over the whole set; each thread accumulates (Σx, Σx²)
    // Starting offset is unique per (threadgroup, thread); the stride skips over every other cooperating thread
    float2 partial = float2(0.0f, 0.0f);
    const uint32_t stride = groupsPerSlice * groupSize; // Total threads cooperating on one set
    for (uint32_t idx = chunkIdx * groupSize + localId; idx < elemsPerSet; idx += stride) {
        const float v = set[idx];
        partial.x += v;        // Σx
        partial.y += v * v;    // Σx²
    }
    localSharedMem[localId] = partial;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Tree-based reduction within the threadgroup (assumes groupSize is a power of two)
    // float2 reduces both Σx and Σx² in a single pass
    for (uint32_t s = groupSize / 2; s > 0; s >>= 1) {
        if (localId < s) {
            localSharedMem[localId] += localSharedMem[localId + s];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Thread 0 folds this threadgroup's partials into the set's accumulator
    // Only one atomic pair per threadgroup, so contention stays low even with many cooperating groups
    if (localId == 0) {
        metal::atomic_fetch_add_explicit(&accum[2 * sliceIdx],     localSharedMem[0].x, metal::memory_order::memory_order_relaxed);
        metal::atomic_fetch_add_explicit(&accum[2 * sliceIdx + 1], localSharedMem[0].y, metal::memory_order::memory_order_relaxed);
    }
}
