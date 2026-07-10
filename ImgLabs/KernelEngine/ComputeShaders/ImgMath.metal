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
