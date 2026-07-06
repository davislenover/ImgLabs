//
//  PerformanceBenchmark.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-06.
//  A quick GPU-vs-CPU comparison for the ZNCC similarity matrix. Runs the exact same all-pairs computation on
//  the GPU (via ImageCorrelation), on a single-threaded CPU reference, and on a multi-threaded CPU reference;
//  times all three, tracks peak memory, checks that they agree, and reports the speed/memory tradeoff.
//  Intended for ad-hoc profiling, not shipping UI.
//

import Foundation
import CoreGraphics
import Metal
import Darwin // mach task_info, for reading the process memory footprint

enum PerformanceBenchmark {

    // The grayscale weights and premultiplied-RGBA handling here mirror convertToGrayScale in ImgMath.metal,
    // so the CPU reference scores identical pixels to the GPU path (the background is black, so with alpha
    // already premultiplied the blend is a no-op and grayscale is just the weighted RGB sum).
    private static let redWeight   : Float = 0.299;
    private static let greenWeight : Float = 0.587;
    private static let blueWeight  : Float = 0.114;

    /// The outcome of one benchmark run
    struct Result {
        let imageCount : Int;
        let canvas : Int;
        let coreCount : Int;            // Logical cores the multi-threaded CPU pass could use
        let gpuColdSeconds : Double;    // First GPU run, including one-time pipeline compilation & buffer setup
        let gpuSeconds : Double;        // Warm GPU run (best of the timed passes, pipelines already compiled)
        let cpuSeconds : Double;        // Single-threaded CPU (best of timed passes)
        let cpuParallelSeconds : Double;// Multi-threaded CPU across all cores (best of timed passes)
        let baselineBytes : UInt64;     // Process footprint before either path allocated its working set
        let gpuPeakBytes : UInt64;      // Peak process footprint observed during the GPU path
        let cpuPeakBytes : UInt64;      // Peak process footprint observed during the CPU paths
        let maxAbsDifference : Float;   // Largest disagreement between the GPU and CPU matrices (should be ~0)

        var speedup : Double { self.cpuSeconds / self.gpuSeconds; }                         // warm GPU vs 1-thread CPU
        var coldSpeedup : Double { self.cpuSeconds / self.gpuColdSeconds; }                 // cold GPU vs 1-thread CPU
        var parallelSpeedup : Double { self.cpuParallelSeconds / self.gpuSeconds; }         // warm GPU vs N-thread CPU
        var coldParallelSpeedup : Double { self.cpuParallelSeconds / self.gpuColdSeconds; } // cold GPU vs N-thread CPU

        /// A one-line summary suitable for a status label
        var summary : String {
            String(format: "GPU %.1fx vs 1-core, %.1fx vs %d-core | peak GPU %@ / CPU %@",
                   self.speedup, self.parallelSpeedup, self.coreCount,
                   PerformanceBenchmark.formatBytes(self.gpuPeakBytes),
                   PerformanceBenchmark.formatBytes(self.cpuPeakBytes));
        }

        /// A multi-line report suitable for the console
        var report : String {
            let gpuDelta = PerformanceBenchmark.formatSignedBytes(Int64(bitPattern: self.gpuPeakBytes) - Int64(bitPattern: self.baselineBytes));
            let cpuDelta = PerformanceBenchmark.formatSignedBytes(Int64(bitPattern: self.cpuPeakBytes) - Int64(bitPattern: self.baselineBytes));
            return """
            ── ImgLabs ZNCC benchmark ─────────────────────────────
              Workload      : \(self.imageCount) images at \(self.canvas)x\(self.canvas) px \
            (\(self.imageCount * (self.imageCount - 1) / 2) pairs), \(self.coreCount) cores
              — Time (best of timed passes) —
              GPU cold (1st): \(String(format: "%.4f s", self.gpuColdSeconds)) (incl. pipeline compile + setup)
              GPU warm      : \(String(format: "%.4f s", self.gpuSeconds))
              CPU 1-thread  : \(String(format: "%.4f s", self.cpuSeconds))
              CPU \(self.coreCount)-thread : \(String(format: "%.4f s", self.cpuParallelSeconds))
              Speedup (warm): \(String(format: "%.2fx vs 1-thread  |  %.2fx vs %d-thread", self.speedup, self.parallelSpeedup, self.coreCount))
              Speedup (cold): \(String(format: "%.2fx vs 1-thread  |  %.2fx vs %d-thread", self.coldSpeedup, self.coldParallelSpeedup, self.coreCount))
              — Peak memory (whole-process phys_footprint; Δ vs baseline) —
              Baseline      : \(PerformanceBenchmark.formatBytes(self.baselineBytes))
              GPU peak      : \(PerformanceBenchmark.formatBytes(self.gpuPeakBytes)) (Δ \(gpuDelta))
              CPU peak      : \(PerformanceBenchmark.formatBytes(self.cpuPeakBytes)) (Δ \(cpuDelta))
              — Accuracy —
              Max |Δ|       : \(String(format: "%.2e", self.maxAbsDifference)) (GPU vs CPU agreement)
            ───────────────────────────────────────────────────────
            """;
        }
    }

    /// Runs the benchmark end to end against a supplied set of images (e.g. the user's imported photos, or
    /// synthetic ones from `syntheticImages`). Every path scores the very same pixels
    /// - Parameters:
    ///     - images: The images to compare; the matrix is images.count x images.count (needs at least two)
    ///     - runs: How many timed passes to take per path; the fastest (least noisy) is reported
    ///     - context: The Metal context the GPU path runs against
    /// - Returns: A Result, or nil if fewer than two images were given or the GPU run could not be produced
    static func run(images: [ImageData], runs: Int = 3, context: MetalComputeContext) async -> Result? {
        guard images.count >= 2 else { return nil; }
        // Report the per-side size the images are actually compared at (their shared canvas)
        let canvas = images.first?.currentSize()?.width ?? 0;
        let cores = ProcessInfo.processInfo.activeProcessorCount;
        let correlation = ImageCorrelation(MetalContext: context);
        let baseline = Self.currentFootprintBytes();

        // GPU path (cold + warm), with peak memory sampled across the whole section. The first run compiles
        // pipeline states and pays one-time setup costs, so it is timed separately as the "cold" number
        var gpuCold = 0.0;
        var gpuBest = Double.greatestFiniteMagnitude;
        var gpuMatrix : [[Float]] = [];
        let gpuPeak = await Self.measuringPeakFootprint {
            let coldStart = Date();
            guard let first = try? await correlation.similarityMatrix(images: images) else { return; }
            gpuCold = Date().timeIntervalSince(coldStart);
            gpuMatrix = first;
            for _ in 0..<max(1, runs) {
                let start = Date();
                _ = try? await correlation.similarityMatrix(images: images);
                gpuBest = min(gpuBest, Date().timeIntervalSince(start));
            }
        };
        guard !gpuMatrix.isEmpty else { return nil; }

        // CPU paths (single- and multi-threaded), with their own peak-memory section. The mean-centered arrays
        // built by prepare() are the dominant CPU allocation, so they are included in the measured window
        var cpuBest = Double.greatestFiniteMagnitude;
        var cpuParallelBest = Double.greatestFiniteMagnitude;
        var cpuMatrix : [[Float]] = [];
        let cpuPeak = await Self.measuringPeakFootprint {
            let prepared = images.map { Self.prepare($0) };
            for _ in 0..<max(1, runs) {
                let start = Date();
                cpuMatrix = Self.cpuSimilarityMatrix(prepared);
                cpuBest = min(cpuBest, Date().timeIntervalSince(start));
            }
            for _ in 0..<max(1, runs) {
                let start = Date();
                _ = Self.cpuSimilarityMatrixParallel(prepared);
                cpuParallelBest = min(cpuParallelBest, Date().timeIntervalSince(start));
            }
        };

        return Result(imageCount: images.count, canvas: canvas, coreCount: cores,
                      gpuColdSeconds: gpuCold, gpuSeconds: gpuBest,
                      cpuSeconds: cpuBest, cpuParallelSeconds: cpuParallelBest,
                      baselineBytes: baseline, gpuPeakBytes: gpuPeak, cpuPeakBytes: cpuPeak,
                      maxAbsDifference: Self.maxAbsDifference(gpuMatrix, cpuMatrix));
    }

    /// Builds a set of synthetic random images at the given canvas size. Used as a fallback for the benchmark
    /// when the user has not imported at least two of their own images
    static func syntheticImages(count: Int, canvas: Int) -> [ImageData] {
        (0..<count).map { _ in ImageData(img: Self.randomImage(size: canvas),
                                         targetWidth: canvas, targetHeight: canvas,
                                         filePath: URL(fileURLWithPath: "/dev/null")) };
    }

    // MARK: - CPU reference

    // One image reduced to what the ZNCC matrix needs: its mean-centered grayscale values and their sum of
    // squares. Mirrors what ImageCorrelation precomputes per image before the pairwise dot products
    private struct Prepared {
        let centered : [Float]; // grayscale value minus the image mean, per pixel
        let sumSquares : Float; // Σ(centered²) — the denominator term for this image
    }

    private static func prepare(_ image: ImageData) -> Prepared {
        guard let rgba = image.rawRGBA() else { return Prepared(centered: [], sumSquares: 0); }
        let pixelCount = rgba.count / 4;

        // Grayscale: weighted sum of the (premultiplied) RGB channels, matching the Metal kernel
        var gray = [Float](repeating: 0, count: pixelCount);
        var sum : Float = 0;
        for i in 0..<pixelCount {
            let base = i * 4;
            let value = Self.redWeight   * Float(rgba[base])
                      + Self.greenWeight * Float(rgba[base + 1])
                      + Self.blueWeight  * Float(rgba[base + 2]);
            gray[i] = value;
            sum += value;
        }
        let mean = pixelCount > 0 ? sum / Float(pixelCount) : 0;

        // Center each value on the mean and accumulate the sum of squares
        var sumSquares : Float = 0;
        for i in 0..<pixelCount {
            let centered = gray[i] - mean;
            gray[i] = centered;
            sumSquares += centered * centered;
        }
        return Prepared(centered: gray, sumSquares: sumSquares);
    }

    /// Single-threaded reference: computes the lower triangle of the ZNCC matrix and mirrors it
    private static func cpuSimilarityMatrix(_ prepared: [Prepared]) -> [[Float]] {
        let n = prepared.count;
        var matrix = Array(repeating: Array(repeating: Float(0), count: n), count: n);
        for i in 0..<n {
            for j in 0...i {
                let zncc = Self.zncc(prepared[i], prepared[j]);
                matrix[i][j] = zncc;
                matrix[j][i] = zncc;
            }
        }
        return matrix;
    }

    /// Multi-threaded reference: the same computation with the rows fanned across all cores via
    /// DispatchQueue.concurrentPerform. Row i owns cells (i, 0...i) and their mirror (0...i, i); those index
    /// sets are disjoint across rows, so the threads write non-overlapping memory (no locking needed)
    /// Results go into a flat backing buffer to avoid mutating a shared [[Float]] from multiple threads
    private static func cpuSimilarityMatrixParallel(_ prepared: [Prepared]) -> [[Float]] {
        let n = prepared.count;
        var flat = [Float](repeating: 0, count: n * n);
        flat.withUnsafeMutableBufferPointer { buffer in
            DispatchQueue.concurrentPerform(iterations: n) { i in
                for j in 0...i {
                    let zncc = Self.zncc(prepared[i], prepared[j]);
                    buffer[i * n + j] = zncc;
                    buffer[j * n + i] = zncc;
                }
            }
        }
        // Reshape the flat buffer back into rows
        return (0..<n).map { i in Array(flat[(i * n)..<(i * n + n)]) };
    }

    /// The ZNCC score for one pair of prepared (mean-centered) images
    private static func zncc(_ a: Prepared, _ b: Prepared) -> Float {
        var dot : Float = 0;
        let count = min(a.centered.count, b.centered.count);
        for k in 0..<count { dot += a.centered[k] * b.centered[k]; }
        let denom = (a.sumSquares * b.sumSquares).squareRoot();
        return denom > 0 ? dot / denom : 0;
    }

    // MARK: - Helpers

    private static func maxAbsDifference(_ lhs: [[Float]], _ rhs: [[Float]]) -> Float {
        guard lhs.count == rhs.count else { return .greatestFiniteMagnitude; }
        var worst : Float = 0;
        for i in 0..<lhs.count {
            for j in 0..<lhs[i].count {
                worst = max(worst, abs(lhs[i][j] - rhs[i][j]));
            }
        }
        return worst;
    }

    /// Generates a random opaque RGBA image at the given size. Content is irrelevant to timing; alpha is fixed
    /// at 255 so the premultiplied-RGBA assumption holds
    private static func randomImage(size: Int) -> CGImage {
        let bytesPerPixel = 4;
        var data = [UInt8](repeating: 0, count: size * size * bytesPerPixel);
        for i in 0..<(size * size) {
            let base = i * bytesPerPixel;
            data[base]     = UInt8.random(in: 0...255);
            data[base + 1] = UInt8.random(in: 0...255);
            data[base + 2] = UInt8.random(in: 0...255);
            data[base + 3] = 255;
        }
        let context = CGContext(data: &data, width: size, height: size,
                                bitsPerComponent: 8, bytesPerRow: size * bytesPerPixel,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!;
        return context.makeImage()!;
    }

    // MARK: - Memory

    /// Tracks the largest footprint sample seen. An actor so the sampler and the work can update it safely
    private actor PeakTracker {
        private(set) var peak : UInt64 = 0;
        func consider(_ value: UInt64) { if value > self.peak { self.peak = value; } }
    }

    /// Runs `body`, polling the process footprint on a background task throughout, and returns the peak seen
    private static func measuringPeakFootprint(_ body: () async -> Void) async -> UInt64 {
        let tracker = PeakTracker();
        await tracker.consider(Self.currentFootprintBytes());
        // Sample on a detached task so it keeps polling on its own thread while `body` occupies this one
        let sampler = Task.detached {
            while !Task.isCancelled {
                await tracker.consider(Self.currentFootprintBytes());
                try? await Task.sleep(nanoseconds: 5_000_000); // 5 ms
            }
        };
        await body();
        await tracker.consider(Self.currentFootprintBytes());
        sampler.cancel();
        return await tracker.peak;
    }

    /// The current process memory footprint in bytes (phys_footprint — the same figure Xcode's memory gauge
    /// and the system's memory-limit accounting use). Returns 0 if the query fails
    private static func currentFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t();
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size);
        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count);
            }
        };
        return result == KERN_SUCCESS ? info.phys_footprint : 0;
    }

    private static func formatBytes(_ bytes: UInt64) -> String {
        let mb = Double(bytes) / (1024 * 1024);
        return mb >= 1024 ? String(format: "%.2f GB", mb / 1024) : String(format: "%.0f MB", mb);
    }

    private static func formatSignedBytes(_ bytes: Int64) -> String {
        (bytes >= 0 ? "+" : "-") + formatBytes(UInt64(abs(bytes)));
    }
}
