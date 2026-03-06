//
//  KernelOps.metal
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-05.
//

kernel void multiply(device uchar* data, uchar multiplier, uint index [[thread_position_in_grid]]) {
    data[index] = data[index] * multiplier; // Each thread is responsible for one element
}
