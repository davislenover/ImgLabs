//
//  KernelOps.metal
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-05.
//
#include <metal_stdlib>
#include <metal_logging>
constant float ZERO_CHECK_VAL = 0.000001f;

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
    // Load data from slow global memory into fast local shared memory
    // If the thread is out of bounds, load 0.0 so it doesn't affect the sum
    localSharedMem[localId] = (threadId < maxLength) ? values[threadId] : 0.0f;
    
    // Synchronize to ensure all threads in this group finished writing to shared memory
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    
    // Perform tree-based reduction inside the local threadgroup
    for (uint32_t stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            localSharedMem[localId] += localSharedMem[localId + stride];
        }
        // Sync after every step of the tree reduction loop
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    
    // Only Thread 0 of this group writes the final group subtotal globally
    if (localId == 0) {
        metal::atomic_fetch_add_explicit(globalSum, localSharedMem[0], metal::memory_order::memory_order_relaxed);
    }
}

/*
 Calculates the subtraction of all elements in an array from a constant, then use as a base for a power operation
 maxLength should equal the number of elements in the array as well as the number of threads to create
 It's important that the programmer ensure values entered do not cause a divison by 0 (i.e., any element in the array is 0 and the power is negative)
*/
kernel void calculateSubtractionPow(
    device float* values [[buffer(0)]], // Values to subtract the constant from (a 1D floating point array)
    constant uint32_t& maxLength [[buffer(1)]], // The length of the values array (NOT values.count-1)
    constant float& conToSubtract [[buffer(2)]], // The constant value to subtract (stored in read-only memory)
    constant float& pwr [[buffer(3)]], // The power value
    device float* resultArr [[buffer(4)]],
    uint32_t threadId [[thread_position_in_grid]])
{
    if (threadId < maxLength) {
        float curVal = values[threadId] - conToSubtract;
        // Base may be negative which is undefined behaviour for pow func so extract sign and check for odd pwr
        float result = metal::pow(metal::abs(curVal),pwr);
        // Check if the result if negative via odd pwr
        float remainder = metal::abs(metal::fmod(pwr, 2.0f));
        if (metal::abs(remainder - 1.0f) < ZERO_CHECK_VAL) {
            result *= metal::sign(curVal);
        }
        resultArr[threadId] = result;
    }
}

/*
 Calculates the dot product of two floating point vectors (arrays), stores the result in a new array
 i.e., result = arr1[0]*arr2[0] + arr1[1]*arr2[1] + ... + arr1[maxLengthArr-1]*arr2[maxLengthArr-1]
 Note that array lengths MUST be at least maxLengthArr, it is the programmers responsibility to check if the arrays are equal length
 maxLengthArr should also equal the number of threads created, the number of threads per threadGroup should be as large as possible to minimize addition contention
 */
kernel void dotArr(
    device float* arr1 [[buffer(0)]],
    device float* arr2 [[buffer(1)]],
    constant uint32_t& maxLengthArr [[buffer(2)]], // Should match the length of both arrays
    device metal::atomic_float* result [[buffer(3)]], // Output
    threadgroup float* localSharedMem  [[threadgroup(0)]], // Shared memory allocation per thread group
    uint32_t threadId                  [[thread_position_in_grid]],
    uint32_t localId                   [[thread_position_in_threadgroup]],
    uint32_t groupSize                 [[threads_per_threadgroup]])
{
    // Allocate memory to store a block of results for a thread group, perform product
    localSharedMem[localId] = (threadId < maxLengthArr) ? arr1[threadId]*arr2[threadId] : 0.0f;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    // Perform thread group reduction
    for (uint32_t stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            localSharedMem[localId] += localSharedMem[localId + stride];
        }
        // Sync after every step of the tree reduction loop
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    // Afterwards, localSharedMem[0] has result for the given thread group
    // Thus, the first thread in every thread group will add the results to a final output
    // Note there is contention but only the first thread in every group is doing this, thus it's less than getting every thread to do it
    if (localId == 0) {
        metal::atomic_fetch_add_explicit(result, localSharedMem[0], metal::memory_order::memory_order_relaxed);
    }
}

/*
 Computes many dot products in a single dispatch: one threadgroup per pair.
 `data` holds the arrays for every image back-to-back, so image k occupies the range
 data[k*elemsPerImage ... (k+1)*elemsPerImage - 1]. `pairs` lists the (i,j) image indices to correlate,
 one entry per threadgroup. Each threadgroup reduces the dot product of images i and j and writes the single
 result to results[groupId]. No atomics are needed, every threadgroup owns a distinct output
 slot, so there is no cross-group contention
 groupSize (threads per threadgroup) should be a power of two (for the tree reduction) and as large as the
 device allows to keep the reduction shallow. The number of threadgroups dispatched must equal the pair count
 */
kernel void batchedDotArr(
    device const float* data           [[buffer(0)]], // Concatenated arrays for every image (imageCount * elemsPerImage floats)
    constant uint2* pairs              [[buffer(1)]], // The (i, j) image indices to correlate, one per threadgroup
    constant uint32_t& elemsPerImage   [[buffer(2)]], // Number of elements in each image's array
    device float* results              [[buffer(3)]], // One dot product per pair, indexed by threadgroup id
    threadgroup float* localSharedMem  [[threadgroup(0)]], // Shared memory allocation per thread group
    uint32_t groupId                   [[threadgroup_position_in_grid]],
    uint32_t localId                   [[thread_position_in_threadgroup]],
    uint32_t groupSize                 [[threads_per_threadgroup]])
{
    // This threadgroup is responsible for one pair; find where each image's array begins
    uint2 pair = pairs[groupId];
    uint32_t baseI = pair.x * elemsPerImage;
    uint32_t baseJ = pair.y * elemsPerImage;

    // Each thread strides through the arrays accumulating a partial dot product (handles arrays longer than
    // the threadgroup by looping). Threads past the end contribute nothing since the loop simply won't run
    float partial = 0.0f;
    for (uint32_t idx = localId; idx < elemsPerImage; idx += groupSize) {
        partial += data[baseI + idx] * data[baseJ + idx];
    }
    localSharedMem[localId] = partial;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Tree-based reduction within the threadgroup (assumes groupSize is a power of two)
    for (uint32_t stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            localSharedMem[localId] += localSharedMem[localId + stride];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // The first thread writes this pair's result; each group owns a distinct slot, so no atomics needed
    if (localId == 0) {
        results[groupId] = localSharedMem[0];
    }
}
