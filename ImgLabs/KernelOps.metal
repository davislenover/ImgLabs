//
//  KernelOps.metal
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-05.
//

// Metal requires that kernel input arguments must have their locations specified (so the GPU doesn't need to waste time finding them in memory)
// For arguments, this is known as a argument buffer table ([buffer(0)] indicates look in slot 0 of the table)
// [[]] indicates to the compiler that inside is some attribute that it needs to handle, outer brackets signal the compiler, inner contains the instructions (separated by a comma)
// [[thread_position_in_grid]] -- For this index variable, populate it with whatever unique thread the GPU will use
kernel void multiply(device uchar* data [[buffer(0)]], constant uchar* multiplier [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    data[index] = data[index] * *multiplier; // Each thread is responsible for one element
}
