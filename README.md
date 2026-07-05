# ImgLabs

**GPU-accelerated image analysis for macOS — built on Swift, SwiftUI, and Metal.**

ImgLabs measures how visually similar images are to one another using **Zero-Normalized Cross-Correlation (ZNCC)**, running the entire numerical pipeline — grayscale conversion, statistics, and correlation — as compute shaders on the GPU. Comparing two 12-megapixel images means tens of millions of floating-point operations; ImgLabs pushes that work off the CPU and onto Metal, where it runs in parallel across thousands of GPU threads.

Powering the app is the **Kernel Engine**: a reusable, thread-safe framework for authoring, batching, and dispatching Metal compute kernels with strongly-typed, `async`/`await`-friendly results. The engine is deliberately decoupled from ImgLabs itself and can be dropped into any Metal project that needs a clean abstraction over raw GPU plumbing.

> **Status:** Active development. The compute pipeline (ZNCC, similarity matrices) is functional and tested; the SwiftUI front-end is being wired up to visualize results.

---

## Highlights

- **End-to-end GPU pipeline** — every stage of ZNCC (grayscale, mean, mean-subtraction, squaring, summation, dot product) runs as a Metal kernel. No per-pixel work happens on the CPU.
- **Batched dispatch** — independent kernels are encoded onto a single command buffer and submitted together, minimizing GPU round-trips (e.g. both images in a pair are grayscaled in one pass).
- **Structured concurrency** — Metal objects aren't uniformly thread-safe, so the engine leans on Swift `actor`s to serialize access, caches compiled pipeline states, and confines command encoding to a single thread across `await` boundaries.
- **A reusable framework** — the Kernel Engine exposes a small set of protocols (`ComputeKernel`, `ComputeKernelCreatable`, `MTBufable`, `ResultObserver`) that make adding a new GPU operation possible without touching the scheduler.

## Requirements

- macOS with a Metal-capable GPU
- Xcode (uses recent SwiftUI APIs, including the Liquid Glass `glassEffect` material)
- Swift concurrency (`actor` / `async`/`await`)

## Building & running

```sh
open ImgLabs.xcodeproj
```

Build and run the `ImgLabs` scheme (⌘R), then import images from the native macOS file picker.

---

## How it works

ImgLabs is organized into three layers, each with a clear responsibility:

```
SwiftUI UI  ──►  Image Compute (ImageCorrelation)  ──►  Kernel Engine  ──►  Metal Shaders (.metal)
```

### The algorithm: ZNCC

Zero-Normalized Cross-Correlation is a lighting-invariant measure of similarity between two signals. For two images `A` and `B` it yields a value in `[-1, 1]`, where `1` means identical:

```
                       Σ (A - meanA)·(B - meanB)
ZNCC(A, B)  =  ────────────────────────────────────────
               √( Σ(A - meanA)²  ·  Σ(B - meanB)² )
```

Subtracting the mean removes overall brightness; normalizing by the standard deviations removes contrast — so ZNCC compares *structure*, not exposure. `ImageCorrelation` realizes this as a chain of GPU passes: **grayscale → mean → subtract-mean → square → sum → dot product**, each one a `ComputeKernel` run through the engine.

### The Kernel Engine

The engine separates *what* a kernel computes from *how* it is scheduled on the GPU. Its core types live in `KernelEngine/HLObjects/`:

| Type | Kind | Responsibility |
| --- | --- | --- |
| `MetalComputeContext` | `actor` | Owns the `MTLDevice`, command queue, and shader library. Caches compiled pipeline states (keyed by function name, compiled lazily via `Task`) and holds the kernel-factory registry. |
| `ComputeKernel` | `protocol` | Describes one kernel: its Metal function name and an `encode()` closure that binds buffers and configures thread dispatch. Is itself observable. |
| `ComputeKernelCreatable` | `protocol` | A factory that allocates the required `MTLBuffer`s and instantiates a `ComputeKernel`. |
| `MetalRunner` | `actor` | Batches kernels onto one command buffer, dispatches to the GPU, `await`s completion, then notifies observers. |
| `MTBufable` | `protocol` | Anything that can turn itself into an `MTLBuffer` — an `ImageData`, or an intermediate result array feeding the next stage. |
| `ObservableResult` / `ResultObserver` / `ObserverStore` | `protocol` / `actor` | A typed, `async` observer pattern for pulling results back off the GPU once a run completes. |

**Concurrency model.** Because Metal's mutable objects can't be freely shared across threads, the engine:

- serializes device/pipeline/registry access behind `MetalComputeContext` (an `actor`);
- pre-fetches every pipeline state and `encode()` closure *before* touching the command buffer, then performs the actual encoding on a single thread — so no Metal object is mutated concurrently across a suspension point;
- caches pipeline compilation per function name, so the second use of a kernel skips the expensive `makeComputePipelineState` step;
- delivers results through `withCheckedContinuation`, bridging Metal's completion-handler callback into `async`/`await`.

**Kernels.** High-level Swift wrappers in `HLShaders/` — `GrayScaleConvert`, `MeanValue`, `DotProduct`, `Subtraction` — pair with the raw Metal functions in `ComputeShaders/` (`ImgMath.metal`, `ArrayMath.metal`).

### Data flow for a single run

```
ImageData ──(MTBufable)──► Factory allocates buffers ──► ComputeKernel
                                                             │
       MetalRunner.runCompute([kernels]) ──► GPU dispatch ──┘
                                                             │
       commandBuffer completes ──► kernel.notifyObservers() ──► ResultObserver.update(with:)
```

### Architecture diagram

![Kernel Engine architecture](ImgLabs/Diagrams/Kernel/kernelSubsystem.png)

*The protocols and actors that make up the Kernel Engine, and how they relate. Source: [`kernelSubsystem.puml`](ImgLabs/Diagrams/Kernel/kernelSubsystem.puml).*

---

## Project layout

```
ImgLabs/
├── ImgLabsApp.swift             App entry point
├── ContentView.swift            SwiftUI UI + ImageModel
├── ImageData.swift              Wraps a CGImage, exposes raw RGBA pixels as an MTLBuffer
├── ImageError.swift             Image-related error types
├── ImageCompute/
│   └── ImageCorrelation.swift   ZNCC similarity + similarity matrix
├── KernelEngine/
│   ├── HLObjects/               Core engine: context, runner, protocols, observers
│   ├── HLShaders/               Swift kernel wrappers (grayscale, mean, dot, subtraction)
│   └── ComputeShaders/          Raw Metal (.metal) kernel functions
└── Diagrams/                    Architecture diagrams (PlantUML)
```

## Adding a new GPU operation

The engine is built to be extended without modifying the scheduler:

1. Write the kernel function in a `.metal` file under `ComputeShaders/`.
2. Add a `ComputeKernel` type in `HLShaders/` that returns the Metal function name and implements `encode()` (buffer binding + thread dispatch).
3. Add a matching `ComputeKernelCreatable` factory that allocates the required buffers.
4. Attach a `ResultObserver` and run it through `MetalRunner.runCompute(...)`.

The new operation composes with every existing kernel — its output (`MTBufable`) can feed straight into the next stage.
