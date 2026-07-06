# ImgLabs

**A GPU-accelerated duplicate-image finder for macOS — built on Swift, SwiftUI, and Metal.**

ImgLabs finds duplicate and near-duplicate images. It scores every pair of imported images with **Zero-Normalized Cross-Correlation (ZNCC)** then clusters the matches so near-identical shots can be reviewed and thinned down to a single keeper. The entire numerical pipeline runs as compute shaders on the GPU: comparing two 12-megapixel images is tens of millions of floating-point operations, and ImgLabs pushes that work off the CPU onto Metal, where it runs in parallel across thousands of threads.

Powering the app is the **Kernel Engine**: a reusable, thread-safe framework for authoring, batching, and dispatching Metal compute kernels with strongly-typed, `async`/`await`-friendly results. The engine is deliberately decoupled from ImgLabs itself and can be dropped into any Metal project that needs a clean abstraction over raw GPU plumbing.

> **Status:** Usable end to end — import images, press **Analyze**, and review the duplicate clusters with an adjustable sensitivity threshold. Active development continues on additional operations and UI polish.

---

## Highlights

- **Duplicate detection** — imported images are scored pairwise, clustered with union-find, and grouped into keep/remove sets, with one representative image (the medoid) kept per cluster.
- **Interactive sensitivity** — a threshold slider re-clusters the results live, trading precision for recall without recomputing the similarity matrix.
- **GPU-accelerated ZNCC** — every stage (grayscale, mean, mean-subtraction, squaring, summation, dot product) runs as a Metal kernel. No per-pixel work happens on the CPU.
- **Upload-once buffer cache** — source pixel data reused across the all-pairs matrix is uploaded to the GPU a single time via a reference-identity `BufferCache` actor, instead of once per comparison.
- **Structured concurrency** — Metal objects aren't uniformly thread-safe, so the engine leans on Swift `actor`s to serialize access, caches compiled pipeline states, and confines command encoding to a single thread across `await` boundaries.
- **A reusable framework** — the Kernel Engine exposes a focused set of protocols (`ComputeKernel`, `ComputeKernelCreatable`, `MTBufable`, `ResultObserver`) that make adding a new GPU operation possible without touching the scheduler.

## Requirements

- macOS with a Metal-capable GPU
- Xcode (uses recent SwiftUI APIs, including the Liquid Glass `glassEffect` material)
- Swift concurrency (`actor` / `async`/`await`)

## Building & running

```sh
open ImgLabs.xcodeproj
```

Build and run the `ImgLabs` scheme (⌘R), import images from the native macOS file picker, then press **Analyze**. Adjust the **Sensitivity** slider to control how aggressively near-duplicates are grouped.

---

## How it works

ImgLabs is organized into layers, each depending only on the one beneath it: the UI drives image compute, which composes GPU work through the Kernel Engine, which dispatches the raw Metal shader functions.

![ImgLabs high-level layers](ImgLabs/Diagrams/Architecture/layers.png)

*Source: [`layers.puml`](ImgLabs/Diagrams/Architecture/layers.puml).*

### From import to duplicates

A run flows from imported files to reviewable duplicate groups. Because imported images can differ in size, they're first resampled onto a common canvas (the smallest width/height across the set) so the pixel-wise correlation compares equal-length arrays. The GPU then builds an all-pairs similarity matrix — computing only the lower triangle, since ZNCC is symmetric — which is clustered on the CPU into duplicate groups.

![Duplicate detection pipeline](ImgLabs/Diagrams/Pipeline/duplicatePipeline.png)

*Source: [`duplicatePipeline.puml`](ImgLabs/Diagrams/Pipeline/duplicatePipeline.puml).*

Clustering uses a **union-find** (disjoint-set) structure: every image pair scoring at or above the threshold is connected, and connected images collapse into a single group (`DuplicateFinder.swift`). For each group, ImgLabs keeps the **medoid** — the image with the highest average similarity to the rest of its cluster — and flags the others for removal. The threshold is a live control, so re-clustering is instant and never re-runs the GPU work.

### The algorithm: ZNCC

Zero-Normalized Cross-Correlation is a lighting-invariant measure of similarity between two signals. For two images `A` and `B` it yields a value in `[-1, 1]`, where `1` means identical:

$$
\text{ZNCC}(A, B) = \frac{\sum_i (A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum_i (A_i - \bar{A})^2 \; \sum_i (B_i - \bar{B})^2}}
$$

where $\bar{A}$ and $\bar{B}$ are the mean pixel values of images $A$ and $B$, and $i$ ranges over every pixel.

Subtracting the mean removes overall brightness; normalizing by the standard deviations removes contrast — so ZNCC compares *structure*, not exposure. `ImageCorrelation` realizes this as a chain of GPU passes: **grayscale → mean → subtract-mean → square → sum → dot product**, each one a `ComputeKernel` run through the engine.

### The Kernel Engine

The engine separates *what* a kernel computes from *how* it is scheduled on the GPU. Its core types live in `KernelEngine/HLObjects/`:

| Type | Kind | Responsibility |
| --- | --- | --- |
| `MetalComputeContext` | `actor` | Owns the `MTLDevice`, command queue, and shader library. Caches compiled pipeline states (keyed by function name, compiled lazily via `Task`) and holds the kernel-factory registry. |
| `ComputeKernel` | `protocol` | Describes one kernel: its Metal function name and an `encode()` closure that binds buffers and configures thread dispatch. Is itself observable. |
| `ComputeKernelCreatable` | `protocol` | A factory that allocates the required `MTLBuffer`s and instantiates a `ComputeKernel`. |
| `MetalRunner` | `actor` | Batches kernels onto one command buffer, dispatches to the GPU, `await`s completion, then notifies observers. |
| `BufferCache` | `actor` | Caches the `MTLBuffer` produced for each source, keyed by reference identity, so data reused across many kernels is uploaded to the device only once. |
| `MTBufable` | `protocol` | Anything that can turn itself into an `MTLBuffer` — an `ImageData`, or an intermediate result array feeding the next stage. |
| `ObservableResult` / `ResultObserver` / `ObserverStore` | `protocol` / `actor` | A typed, `async` observer pattern for pulling results back off the GPU once a run completes. |

**Concurrency model.** Because Metal's mutable objects can't be freely shared across threads, the engine:

- serializes device/pipeline/registry access behind `MetalComputeContext` (an `actor`);
- pre-fetches every pipeline state and `encode()` closure *before* touching the command buffer, then performs the actual encoding on a single thread — so no Metal object is mutated concurrently across a suspension point;
- caches pipeline compilation per function name, so the second use of a kernel skips the expensive `makeComputePipelineState` step;
- delivers results through `withCheckedContinuation`, bridging Metal's completion-handler callback into `async`/`await`.

**Kernels.** High-level Swift wrappers in `HLShaders/` — `GrayScaleConvert`, `MeanValue`, `DotProduct`, `Subtraction` — pair with the raw Metal functions in `ComputeShaders/` (`ImgMath.metal`, `ArrayMath.metal`).

### Running a batch of kernels

A factory allocates the device buffers and builds each kernel; `MetalRunner` encodes the whole batch onto one command buffer, dispatches it, and fans the finished results back out to any subscribed observers.

![Running a batch of kernels](ImgLabs/Diagrams/Kernel/kernelRun.png)

*Source: [`kernelRun.puml`](ImgLabs/Diagrams/Kernel/kernelRun.puml).*

### Kernel Engine class diagram

The protocols and actors that make up the engine, and how they relate:

![Kernel Engine architecture](ImgLabs/Diagrams/Kernel/kernelSubsystem.png)

*Source: [`kernelSubsystem.puml`](ImgLabs/Diagrams/Kernel/kernelSubsystem.puml).*

---

## Project layout

```
ImgLabs/
├── ImgLabsApp.swift             App entry point
├── ContentView.swift            SwiftUI UI, app-state models, import/analyze flow
├── DuplicateView.swift          Renders duplicate clusters + the sensitivity slider
├── ImageData.swift              Wraps a CGImage; resamples to a canvas; exposes RGBA as an MTLBuffer
├── ImageError.swift             Image-related error types
├── ImageCompute/
│   ├── ImageCorrelation.swift   ZNCC similarity + all-pairs similarity matrix
│   └── DuplicateFinder.swift    Union-find clustering + medoid keeper selection
├── KernelEngine/
│   ├── HLObjects/               Core engine: context, runner, buffer cache, protocols, observers
│   ├── HLShaders/               Swift kernel wrappers (grayscale, mean, dot, subtraction)
│   └── ComputeShaders/          Raw Metal (.metal) kernel functions
└── Diagrams/                    Architecture & pipeline diagrams (PlantUML)
```

## Adding a new GPU operation

The engine is built to be extended without modifying the scheduler:

1. Write the kernel function in a `.metal` file under `ComputeShaders/`.
2. Add a `ComputeKernel` type in `HLShaders/` that returns the Metal function name and implements `encode()` (buffer binding + thread dispatch).
3. Add a matching `ComputeKernelCreatable` factory that allocates the required buffers.
4. Attach a `ResultObserver` and run it through `MetalRunner.runCompute(...)`.

The new operation composes with every existing kernel — its output (`MTBufable`) can feed straight into the next stage.
