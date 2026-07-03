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
