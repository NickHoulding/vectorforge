# SIMD and AVX512: A Comprehensive Guide for VectorForge Optimization

**Author's Background Assumptions:**
- Comfortable with C programming fundamentals
- Strong understanding of cosine similarity and matrix operations from AI/ML
- Familiar with concepts like dot products, vector normalization
- New to C++ and CPU-level optimizations

**Learning Objectives:**
By the end of this document, you will be able to:
1. Explain what SIMD and AVX512 are and why they matter
2. Understand the performance benefits at a hardware level
3. Confidently implement SIMD optimizations in VectorForge
4. Explain these optimizations to others in technical discussions

---

## Table of Contents

1. [The Big Picture: Why SIMD Matters](#1-the-big-picture-why-simd-matters)
2. [Understanding Your CPU: The Hardware Foundation](#2-understanding-your-cpu-the-hardware-foundation)
3. [SIMD Explained: One Instruction, Multiple Data](#3-simd-explained-one-instruction-multiple-data)
4. [AVX, AVX2, and AVX512: Evolution of Intel's SIMD](#4-avx-avx2-and-avx512-evolution-of-intels-simd)
5. [Practical Example: Dot Product Optimization](#5-practical-example-dot-product-optimization)
6. [VectorForge-Specific: Optimizing Cosine Similarity](#6-vectorforge-specific-optimizing-cosine-similarity)
7. [Implementation Guide: Step-by-Step](#7-implementation-guide-step-by-step)
8. [Common Pitfalls and How to Avoid Them](#8-common-pitfalls-and-how-to-avoid-them)
9. [Benchmarking and Validation](#9-benchmarking-and-validation)
10. [Further Optimizations and Next Steps](#10-further-optimizations-and-next-steps)

---

## 1. The Big Picture: Why SIMD Matters

### The Performance Opportunity

In VectorForge, we currently achieve:
- **Raw C++ function:** 2.5x faster than Python
- **End-to-end search:** 1.3x faster than Python

With SIMD/AVX512 optimization, we can potentially achieve:
- **Raw C++ function:** 8-16x faster than Python
- **End-to-end search:** 3-5x faster than Python

### Where Does This Speedup Come From?

Let's use an analogy from your ML background. Imagine you're computing:

```python
# Regular code (scalar processing)
result = 0
for i in range(384):
    result += query[i] * doc[i]
```

Your CPU executes this as **384 separate multiply-add operations**, one at a time.

With SIMD/AVX512, you execute:
```
# SIMD code (vector processing)
result = 0
for i in range(0, 384, 16):  # Process 16 floats at once!
    result += simd_multiply_add(query[i:i+16], doc[i:i+16])
```

Your CPU now does **24 operations** (384÷16) instead of 384. That's **16x fewer instructions** to achieve the same result!

### The "Free Lunch" in Modern Hardware

This isn't theoretical—your CPU already has these capabilities built-in. Using SIMD is like discovering your car has a turbo button you never pressed. The hardware is already there; we just need to tell the CPU to use it.

---

## 2. Understanding Your CPU: The Hardware Foundation

### CPU Architecture Basics

Modern Intel/AMD CPUs have a hierarchical structure:

```
┌─────────────────────────────────────────────────────────────┐
│                         CPU Core                             │
├─────────────────────────────────────────────────────────────┤
│  Scalar Units          │  Vector Units (SIMD)               │
│  ┌─────────────────┐   │  ┌──────────────────────────────┐  │
│  │ Integer ALU     │   │  │ 512-bit Vector Register      │  │
│  │ (64-bit)        │   │  │ (16 × 32-bit floats)         │  │
│  ├─────────────────┤   │  ├──────────────────────────────┤  │
│  │ Floating Point  │   │  │ Vector ALU                   │  │
│  │ Unit (1 float)  │   │  │ (operate on all 16 at once)  │  │
│  └─────────────────┘   │  └──────────────────────────────┘  │
│                        │                                    │
│  Processes 1 value     │  Processes 16 values               │
│  at a time             │  simultaneously                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Parallelism Within a Single Core

This is **not** the same as multi-threading or multi-core processing:

| Concept | What It Does | Example |
|---------|-------------|---------|
| **Multi-core** | Run different tasks on different cores | Core 1 searches docs 0-999, Core 2 searches docs 1000-1999 |
| **Multi-threading** | Multiple instruction streams sharing cores | Thread 1 does I/O while Thread 2 computes |
| **SIMD (this guide)** | **One instruction** operates on **multiple data elements** within a single core | One multiply instruction processes 16 floats simultaneously |

**Analogy:**
- Multi-core: Having multiple workers
- Multi-threading: Workers can switch between different tasks
- SIMD: Each worker has 16 hands and can do 16 things at once

### Register Sizes: The Evolution

Your CPU has special registers (super-fast storage) for SIMD operations:

```
SSE (2001):     128-bit register = 4 × float32
                ████████████████

AVX (2011):     256-bit register = 8 × float32
                ████████████████████████████████

AVX2 (2013):    256-bit register = 8 × float32 (better integer ops)
                ████████████████████████████████

AVX512 (2017):  512-bit register = 16 × float32
                ████████████████████████████████████████████████████████████████
```

**For VectorForge:** Our embeddings are 384-dimensional `float32` arrays, so:
- **Without SIMD:** 384 operations
- **With SSE:** 384÷4 = 96 operations
- **With AVX/AVX2:** 384÷8 = 48 operations
- **With AVX512:** 384÷16 = **24 operations** ✨

---

## 3. SIMD Explained: One Instruction, Multiple Data

### The Core Concept

**SIMD = Single Instruction, Multiple Data**

Let's make this concrete with actual numbers from a dot product:

#### Scalar (Regular) Code:
```c
float query[4] = {1.0, 2.0, 3.0, 4.0};
float doc[4]   = {0.5, 1.5, 2.5, 3.5};

// CPU executes 4 separate multiply operations:
float p0 = query[0] * doc[0];  // Instruction 1: 1.0 * 0.5 = 0.5
float p1 = query[1] * doc[1];  // Instruction 2: 2.0 * 1.5 = 3.0
float p2 = query[2] * doc[2];  // Instruction 3: 3.0 * 2.5 = 7.5
float p3 = query[3] * doc[3];  // Instruction 4: 4.0 * 3.5 = 14.0

// Then 3 more add operations:
float sum = p0 + p1 + p2 + p3;  // = 25.0

// Total: 7 instructions executed
```

#### SIMD (Vector) Code:
```c
// Using AVX (256-bit = 8 floats, but showing 4 for simplicity)
__m128 query_vec = _mm_loadu_ps(query);  // Load 4 floats into one vector
__m128 doc_vec   = _mm_loadu_ps(doc);    // Load 4 floats into one vector

// ONE instruction multiplies ALL FOUR pairs simultaneously!
__m128 products = _mm_mul_ps(query_vec, doc_vec);
// products now contains {0.5, 3.0, 7.5, 14.0}

// Total: 1 multiply instruction (instead of 4)
```

### How the Hardware Actually Works

Inside the CPU, there are physical circuits that look like this:

```
Scalar Float Multiplier:
┌─────────────┐
│   Input A   │──┐
└─────────────┘  │
                 ├──► Multiply ──► Output
┌─────────────┐  │
│   Input B   │──┘
└─────────────┘

Processes: 1 multiplication per clock cycle


SIMD Vector Multiplier (4-wide):
┌─────────────┐      ┌──────────┐
│ A[0] A[1]   │──┬──►│  Mult 0  │──┬──► Result[0]
│ A[2] A[3]   │  │   ├──────────┤  │
└─────────────┘  │   │  Mult 1  │──┼──► Result[1]
                 │   ├──────────┤  │
┌─────────────┐  │   │  Mult 2  │──┼──► Result[2]
│ B[0] B[1]   │──┘   ├──────────┤  │
│ B[2] B[3]   │      │  Mult 3  │──┴──► Result[3]
└─────────────┘      └──────────┘

Processes: 4 multiplications per clock cycle
```

**Key Point:** The SIMD circuits exist in parallel on the chip. By using one SIMD instruction instead of four scalar instructions, we:
1. Use less power (one instruction fetch vs. four)
2. Process data faster (parallel execution)
3. Keep the pipeline fuller (better CPU utilization)

### Data Alignment: Why It Matters

SIMD operations work best when data is **aligned** to the register size:

```
Aligned (fast):
Memory address:  0x0000  0x0010  0x0020  0x0030
                 ┌──────┬──────┬──────┬──────┐
Data:            │ Vec0 │ Vec1 │ Vec2 │ Vec3 │
                 └──────┴──────┴──────┴──────┘
                    ↑ Each starts at 16-byte boundary
                    One memory fetch gets entire vector


Unaligned (slower):
Memory address:  0x0000  0x0010  0x0020  0x0030
                 ┌───┬──────┬──────┬──────┬──┐
Data:            │   │ Vec0 │ Vec1 │ Vec2 │  │
                 └───┴──────┴──────┴──────┴──┘
                      ↑ Spans two cache lines
                      Requires two memory fetches + merge
```

**For VectorForge:** NumPy arrays are typically aligned, but we'll verify this in our code.

---

## 4. AVX, AVX2, and AVX512: Evolution of Intel's SIMD

### Historical Timeline

```
2001: SSE2 ───► 128-bit registers (4 floats)
                First widely-used SIMD for floating point

2011: AVX ────► 256-bit registers (8 floats)
                Doubled register width
                New 3-operand instruction format

2013: AVX2 ───► 256-bit registers (8 floats)
                Added integer operations to AVX
                Fused Multiply-Add (FMA) instructions

2017: AVX512 ─► 512-bit registers (16 floats)
                Quadrupled width from SSE
                Masking for conditional operations
                Many specialized variants
```

### AVX512 Variants: What You Need to Know

AVX512 is actually a family of instruction sets:

| Variant | What It Adds | Availability |
|---------|-------------|--------------|
| **AVX512F** (Foundation) | Core 512-bit operations, basic for all | Xeon Phi, Skylake-X (2017+) |
| **AVX512DQ** | Additional conversions and operations | Skylake-X (2017+) |
| **AVX512VL** | Lets you use AVX512 ops on 128/256-bit vectors | Skylake-X (2017+) |
| **AVX512BW** | Byte/word operations | Skylake-X (2017+) |
| **AVX512VNNI** | Vector Neural Network Instructions | Cascade Lake (2019+) |

**For VectorForge:** We primarily need **AVX512F** (foundation), which handles `float32` operations perfectly.

### Checking Your CPU Support

Run this command to see what your CPU supports:
```bash
lscpu | grep -i avx
# Or on Linux:
cat /proc/cpuinfo | grep flags | head -1
# Look for: avx, avx2, avx512f
```

If you don't have AVX512, don't worry! The same concepts apply to AVX2 (8 floats at a time instead of 16).

### Key Features for Our Use Case

#### 1. Larger Registers (512 bits = 16 × float32)

```c
// Load 16 floats in one instruction
__m512 vec = _mm512_loadu_ps(array);  // array[0..15]
```

#### 2. Fused Multiply-Add (FMA)

This is **crucial** for dot products. Instead of separate multiply + add:

```c
// Old way (2 instructions):
result = a * b;
result = result + c;

// FMA way (1 instruction):
result = a * b + c;  // _mm512_fmadd_ps(a, b, c)
```

**For dot products:** This is perfect because we repeatedly do `sum += query[i] * doc[i]`.

#### 3. Horizontal Reductions

After processing vectors, we often need to sum all elements:

```c
__m512 vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// Need to compute: 1+2+3+...+16 = 136

// AVX512 provides efficient reduction:
float sum = _mm512_reduce_add_ps(vec);  // sum = 136
```

### Comparison: Scalar vs AVX vs AVX512

Let's compute a dot product of two 16-element vectors:

```c
float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
float b[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
// Expected result: 1+2+3+...+16 = 136
```

**Scalar Code (regular C):**
```c
float scalar_dot_product(float* a, float* b) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += a[i] * b[i];  // 16 multiply instructions
                              // 16 add instructions
    }
    return sum;  // Total: 32 instructions
}
```

**AVX Code (256-bit = 8 floats):**
```c
float avx_dot_product(float* a, float* b) {
    __m256 sum_vec = _mm256_setzero_ps();  // Initialize to 0

    for (int i = 0; i < 16; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);  // Load 8 floats
        __m256 b_vec = _mm256_loadu_ps(&b[i]);  // Load 8 floats
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);  // FMA
    }
    // 2 iterations × 1 FMA = 2 FMA instructions

    // Horizontal sum (add all 8 lanes together)
    // This takes a few more instructions, but still fast
    float result = horizontal_sum_avx(sum_vec);
    return result;  // Total: ~10 instructions
}
```

**AVX512 Code (512-bit = 16 floats):**
```c
float avx512_dot_product(float* a, float* b) {
    __m512 a_vec = _mm512_loadu_ps(a);      // Load all 16 floats
    __m512 b_vec = _mm512_loadu_ps(b);      // Load all 16 floats

    __m512 prod = _mm512_mul_ps(a_vec, b_vec);  // Multiply all 16
    float sum = _mm512_reduce_add_ps(prod);      // Sum all 16

    return sum;  // Total: 4 instructions!
}
```

**Instruction Count Summary:**
- Scalar: **32 instructions**
- AVX: **~10 instructions** (3.2x fewer)
- AVX512: **4 instructions** (8x fewer)

And remember, our vectors are **384 dimensions**, not 16, so the savings multiply!

---

## 5. Practical Example: Dot Product Optimization

Let's implement a real dot product with increasing optimization levels. This builds directly toward our cosine similarity function.

### Level 0: Naive C Code (Baseline)

```c
float dot_product_naive(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// For n=384: Executes 384 multiply + 384 add = 768 operations
```

### Level 1: Manual Loop Unrolling (Scalar Optimization)

```c
float dot_product_unrolled(const float* a, const float* b, int n) {
    float sum = 0.0f;
    int i;

    // Process 4 at a time
    for (i = 0; i < n - 3; i += 4) {
        sum += a[i]   * b[i];
        sum += a[i+1] * b[i+1];
        sum += a[i+2] * b[i+2];
        sum += a[i+3] * b[i+3];
    }

    // Handle remainder
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// Helps with CPU pipelining, ~1.2-1.5x faster than naive
// But still processes one float at a time
```

### Level 2: AVX2 Implementation (8 floats at once)

```c
#include <immintrin.h>  // AVX/AVX2 intrinsics

float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();  // 8 × float32 = 0.0
    int i;

    // Main loop: process 8 floats per iteration
    for (i = 0; i < n - 7; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);  // Load 8 floats from a
        __m256 b_vec = _mm256_loadu_ps(&b[i]);  // Load 8 floats from b

        // Fused multiply-add: sum_vec = (a_vec * b_vec) + sum_vec
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // Horizontal sum: add all 8 lanes together
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);  // Upper 4
    __m128 sum_low  = _mm256_castps256_ps128(sum_vec);    // Lower 4
    sum_low = _mm_add_ps(sum_low, sum_high);              // Combine

    sum_low = _mm_hadd_ps(sum_low, sum_low);  // Horizontal add
    sum_low = _mm_hadd_ps(sum_low, sum_low);  // Horizontal add again

    float sum = _mm_cvtss_f32(sum_low);

    // Handle remainder (if n not divisible by 8)
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// For n=384: 384÷8 = 48 vector operations
// ~6-8x faster than naive (depends on CPU, memory, etc.)
```

**Understanding the Code:**

1. **`__m256`**: Data type for 256-bit vector (8 floats)
2. **`_mm256_loadu_ps`**: Load 8 unaligned floats (u = unaligned, ps = packed single-precision)
3. **`_mm256_fmadd_ps`**: Fused multiply-add on 8 floats simultaneously
4. **Horizontal sum**: Collapse 8 values into 1 final sum

### Level 3: AVX512 Implementation (16 floats at once)

```c
#include <immintrin.h>  // AVX512 intrinsics

float dot_product_avx512(const float* a, const float* b, int n) {
    __m512 sum_vec = _mm512_setzero_ps();  // 16 × float32 = 0.0
    int i;

    // Main loop: process 16 floats per iteration
    for (i = 0; i < n - 15; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(&a[i]);  // Load 16 floats
        __m512 b_vec = _mm512_loadu_ps(&b[i]);  // Load 16 floats

        // Fused multiply-add: sum_vec = (a_vec * b_vec) + sum_vec
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // Horizontal sum: AVX512 provides a single instruction!
    float sum = _mm512_reduce_add_ps(sum_vec);

    // Handle remainder
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// For n=384: 384÷16 = 24 vector operations
// ~10-16x faster than naive
```

**Key Improvements over AVX2:**
- **Simpler code**: `_mm512_reduce_add_ps()` replaces complex horizontal sum
- **Fewer iterations**: 24 instead of 48
- **Better throughput**: CPU can execute more 512-bit ops/cycle on modern chips

### Level 4: AVX512 with Multiple Accumulators

This is a **pro optimization** that reduces dependency chains:

```c
float dot_product_avx512_optimized(const float* a, const float* b, int n) {
    // Use 4 independent accumulators to reduce dependency chains
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    int i;

    // Process 64 floats per iteration (4 × 16)
    for (i = 0; i < n - 63; i += 64) {
        __m512 a0 = _mm512_loadu_ps(&a[i]);
        __m512 b0 = _mm512_loadu_ps(&b[i]);
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);

        __m512 a1 = _mm512_loadu_ps(&a[i + 16]);
        __m512 b1 = _mm512_loadu_ps(&b[i + 16]);
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);

        __m512 a2 = _mm512_loadu_ps(&a[i + 32]);
        __m512 b2 = _mm512_loadu_ps(&b[i + 32]);
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);

        __m512 a3 = _mm512_loadu_ps(&a[i + 48]);
        __m512 b3 = _mm512_loadu_ps(&b[i + 48]);
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
    }

    // Combine accumulators
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    float sum = _mm512_reduce_add_ps(sum0);

    // Handle remainder
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// For n=384: 384÷64 = 6 iterations of 4 FMAs = 24 FMA ops (same)
// But better CPU pipeline utilization
// ~12-20x faster than naive on modern CPUs
```

**Why Multiple Accumulators Help:**

Each FMA instruction has **latency** (time to complete) vs **throughput** (how many can start per cycle):
- **Latency:** 4-6 cycles (result available after 4-6 cycles)
- **Throughput:** 2 per cycle (can start 2 new FMAs per cycle)

With one accumulator:
```
Cycle 0: FMA0 starts
Cycle 1: (waiting for FMA0)
Cycle 2: (waiting for FMA0)
Cycle 3: (waiting for FMA0)
Cycle 4: FMA0 done, FMA1 starts
```

With four accumulators:
```
Cycle 0: FMA0 starts (sum0)
Cycle 1: FMA1 starts (sum1) - independent!
Cycle 2: FMA2 starts (sum2) - independent!
Cycle 3: FMA3 starts (sum3) - independent!
Cycle 4: FMA4 starts (sum0) - FMA0 is done, reuse sum0
```

**Pipeline stays full** = better performance!

### Performance Comparison Table

For a 384-dimensional dot product:

| Implementation | Instructions | Relative Speed | Actual Time (estimate) |
|---------------|-------------|----------------|----------------------|
| Naive | 768 | 1.0x | 100 ns |
| Unrolled | 768 | 1.3x | 77 ns |
| AVX2 | ~56 | 7.0x | 14 ns |
| AVX512 | ~28 | 12.0x | 8 ns |
| AVX512 optimized | ~28 | 16.0x | 6 ns |

**Real-world note:** Actual speedups depend on:
- Memory bandwidth (can data be fed to SIMD units fast enough?)
- Cache locality (is data in L1/L2/L3 cache?)
- CPU model (newer = better SIMD execution units)
- Compiler optimizations (gcc/clang can auto-vectorize simple cases)

---

## 6. VectorForge-Specific: Optimizing Cosine Similarity

Now let's apply everything we've learned to VectorForge's actual use case.

### The Cosine Similarity Formula

From your ML background, you know:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product = Σ(A[i] × B[i])
- ||A|| = magnitude = sqrt(Σ(A[i]²))
```

**VectorForge simplification:** Our embeddings are already normalized (`||A|| = ||B|| = 1.0`), so:

```
cosine_similarity(A, B) = A · B  (when normalized)
```

This is **just a dot product**! Perfect for SIMD optimization.

### Current Implementation Analysis

Let's look at our current C++ code (from `cpp/src/similarity.cpp`):

```cpp
py::array_t<float> cosine_similarity_batch(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
) {
    auto q = query_embedding.unchecked<1>();
    auto docs = doc_embeddings.unchecked<2>();

    int n_docs = docs.shape(0);  // e.g., 5000
    int dim = docs.shape(1);     // 384

    auto result = py::array_t<float>(n_docs);
    auto r = result.mutable_unchecked<1>();

    // Compute cosine similarity for each document
    for (int i = 0; i < n_docs; i++) {
        float score = 0.0f;

        // Dot product: this is where we spend most time!
        for (int j = 0; j < dim; j++) {
            score += q(j) * docs(i, j);
        }

        r(i) = score;
    }

    return result;
}
```

**Analysis:**
- **Outer loop:** 5,000 iterations (one per document)
- **Inner loop:** 384 iterations (embedding dimension)
- **Total operations:** 5,000 × 384 = 1,920,000 multiplies + 1,920,000 adds

**Optimization opportunity:** The inner loop is a perfect candidate for SIMD!

### Optimization Strategy

We'll create a **hierarchy of implementations** with runtime CPU detection:

```cpp
// 1. Detect CPU capabilities at runtime
bool has_avx512 = check_avx512_support();
bool has_avx2 = check_avx2_support();

// 2. Choose best available implementation
if (has_avx512) {
    return cosine_similarity_batch_avx512(...);
} else if (has_avx2) {
    return cosine_similarity_batch_avx2(...);
} else {
    return cosine_similarity_batch_scalar(...);  // Fallback
}
```

This ensures:
- **Portability:** Works on all CPUs (scalar fallback)
- **Performance:** Uses best available instructions
- **No recompilation needed:** Single binary adapts to hardware

### Memory Layout Considerations

Our data comes from NumPy as:
```
query_embedding: shape (384,)       → Contiguous 1D array
doc_embeddings:  shape (5000, 384)  → 2D array

Memory layout:
[ doc0[0], doc0[1], ..., doc0[383], doc1[0], doc1[1], ..., doc4999[383] ]
```

**For SIMD efficiency:**
- ✅ Query is contiguous → perfect for vectorization
- ✅ Each document row is contiguous → perfect for vectorization
- ❌ Accessing same dimension across docs (column-wise) is not contiguous → avoid

**Implication:** We'll vectorize **within each dot product** (across dimensions), not across documents.

### Cache Optimization

Modern CPUs have a cache hierarchy:

```
L1 Cache:  32-64 KB   ~4 cycles latency    (per core)
L2 Cache:  256-512 KB ~12 cycles latency   (per core)
L3 Cache:  8-32 MB    ~40 cycles latency   (shared)
RAM:       16+ GB     ~200 cycles latency
```

**Our data sizes:**
- Query: 384 × 4 bytes = 1.5 KB → **fits in L1** ✅
- Single doc: 384 × 4 bytes = 1.5 KB → **fits in L1** ✅
- All docs: 5000 × 384 × 4 = 7.3 MB → **needs L3 or RAM**

**Optimization insight:**
- Query stays in L1 cache throughout all computations
- Each document is loaded once, computed with cached query
- Sequential access pattern is cache-friendly

No special cache optimizations needed beyond SIMD!

---

## 7. Implementation Guide: Step-by-Step

Let's build the AVX512 optimized cosine similarity function incrementally.

### Step 1: Set Up CPU Feature Detection

Create a helper function to check AVX512 support:

```cpp
// cpp/src/cpu_features.h
#ifndef VECTORFORGE_CPU_FEATURES_H
#define VECTORFORGE_CPU_FEATURES_H

#include <cpuid.h>
#include <stdbool.h>

// Check if CPU supports AVX512F (foundation)
inline bool has_avx512f() {
    unsigned int eax, ebx, ecx, edx;

    // Check CPUID function 7, subfunction 0
    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    // AVX512F is bit 16 of EBX
    return (ebx & (1 << 16)) != 0;
}

// Check if CPU supports AVX2
inline bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    // AVX2 is bit 5 of EBX
    return (ebx & (1 << 5)) != 0;
}

// Check if OS supports AVX512 (OS must save/restore 512-bit registers)
inline bool os_supports_avx512() {
    unsigned int eax, ebx, ecx, edx;

    __cpuid_count(1, 0, eax, ebx, ecx, edx);

    // Check OSXSAVE (bit 27 of ECX)
    if (!(ecx & (1 << 27))) {
        return false;
    }

    // Check XCR0 register (extended control register)
    unsigned int xcr0_lo, xcr0_hi;
    __asm__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));

    // Bits 5-7 must be set for AVX512 state
    return (xcr0_lo & 0xE6) == 0xE6;
}

inline bool cpu_supports_avx512() {
    return has_avx512f() && os_supports_avx512();
}

#endif // VECTORFORGE_CPU_FEATURES_H
```

**What this does:**
- **`__get_cpuid_max()`**: Checks which CPUID functions are available
- **`__cpuid_count()`**: Executes CPUID instruction to query CPU features
- **XCR0 check**: Ensures operating system is saving AVX512 state on context switches

### Step 2: Implement Scalar Fallback (Baseline)

This is our current implementation, cleaned up:

```cpp
// cpp/src/similarity_scalar.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> cosine_similarity_batch_scalar(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
) {
    auto q = query_embedding.unchecked<1>();
    auto docs = doc_embeddings.unchecked<2>();

    const int n_docs = docs.shape(0);
    const int dim = docs.shape(1);

    auto result = py::array_t<float>(n_docs);
    auto r = result.mutable_unchecked<1>();

    // Compute dot product for each document
    for (int i = 0; i < n_docs; i++) {
        float score = 0.0f;

        for (int j = 0; j < dim; j++) {
            score += q(j) * docs(i, j);
        }

        r(i) = score;
    }

    return result;
}
```

### Step 3: Implement AVX512 Dot Product Helper

Create a standalone function for the SIMD dot product:

```cpp
// cpp/src/similarity_avx512.cpp
#include <immintrin.h>  // AVX512 intrinsics
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// AVX512 dot product: compute sum of (a[i] * b[i]) for all i
inline float dot_product_avx512(const float* a, const float* b, int n) {
    // Use 4 accumulators for better pipeline utilization
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    int i = 0;

    // Main loop: process 64 floats per iteration (4 × 16)
    for (; i + 63 < n; i += 64) {
        __m512 a0 = _mm512_loadu_ps(&a[i]);
        __m512 b0 = _mm512_loadu_ps(&b[i]);
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);

        __m512 a1 = _mm512_loadu_ps(&a[i + 16]);
        __m512 b1 = _mm512_loadu_ps(&b[i + 16]);
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);

        __m512 a2 = _mm512_loadu_ps(&a[i + 32]);
        __m512 b2 = _mm512_loadu_ps(&b[i + 32]);
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);

        __m512 a3 = _mm512_loadu_ps(&a[i + 48]);
        __m512 b3 = _mm512_loadu_ps(&b[i + 48]);
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
    }

    // Process remaining 16-element chunks
    for (; i + 15 < n; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(&a[i]);
        __m512 b_vec = _mm512_loadu_ps(&b[i]);
        sum0 = _mm512_fmadd_ps(a_vec, b_vec, sum0);
    }

    // Combine all accumulators
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    // Reduce vector to scalar
    float sum = _mm512_reduce_add_ps(sum0);

    // Handle any remaining elements (< 16)
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}
```

**Code breakdown:**

1. **`__m512 sum0 = _mm512_setzero_ps()`**
   - Creates a 512-bit vector (16 floats) initialized to 0.0
   - Type: `__m512` = 512-bit SIMD type

2. **`_mm512_loadu_ps(&a[i])`**
   - Loads 16 consecutive floats from memory
   - `loadu` = load unaligned (works even if address not 64-byte aligned)
   - `ps` = packed single-precision (float32)

3. **`_mm512_fmadd_ps(a0, b0, sum0)`**
   - Fused multiply-add: `result = (a0 * b0) + sum0`
   - Performs 16 operations simultaneously
   - More accurate than separate multiply + add (one rounding instead of two)

4. **`_mm512_reduce_add_ps(sum0)`**
   - Horizontal reduction: adds all 16 lanes together
   - Returns a single float

5. **Remainder loop**
   - Handles cases where n is not a multiple of 16
   - For n=384: 384 = 24×16, so remainder = 0 (no scalar ops needed!)

### Step 4: Implement Batch Function with AVX512

Now use the dot product helper for the full batch operation:

```cpp
py::array_t<float> cosine_similarity_batch_avx512(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
) {
    auto q = query_embedding.unchecked<1>();
    auto docs = doc_embeddings.unchecked<2>();

    const int n_docs = docs.shape(0);
    const int dim = docs.shape(1);

    // Get raw pointers for fast access
    const float* q_data = q.data(0);

    auto result = py::array_t<float>(n_docs);
    auto r = result.mutable_unchecked<1>();

    // Compute dot product for each document using AVX512
    for (int i = 0; i < n_docs; i++) {
        const float* doc_data = docs.data(i, 0);
        r(i) = dot_product_avx512(q_data, doc_data, dim);
    }

    return result;
}
```

**Key differences from scalar version:**
- Use `q.data(0)` to get raw pointer (faster than `q(j)` indexing)
- Call AVX512 dot product instead of manual loop

### Step 5: Add Runtime Dispatch

Create the main entry point that chooses the best implementation:

```cpp
// cpp/src/similarity.cpp
#include "cpu_features.h"
#include "similarity_scalar.cpp"
#include "similarity_avx512.cpp"

py::array_t<float> cosine_similarity_batch(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
) {
    // Validate inputs
    if (query_embedding.ndim() != 1) {
        throw std::runtime_error("query_embedding must be 1-dimensional");
    }
    if (doc_embeddings.ndim() != 2) {
        throw std::runtime_error("doc_embeddings must be 2-dimensional");
    }
    if (query_embedding.shape(0) != doc_embeddings.shape(1)) {
        throw std::runtime_error("dimension mismatch");
    }

    // Runtime CPU detection and dispatch
    if (cpu_supports_avx512()) {
        return cosine_similarity_batch_avx512(query_embedding, doc_embeddings);
    } else {
        return cosine_similarity_batch_scalar(query_embedding, doc_embeddings);
    }
}

// Pybind11 module definition
PYBIND11_MODULE(vectorforge_cpp, m) {
    m.def("cosine_similarity_batch", &cosine_similarity_batch,
          "Compute cosine similarity between query and batch of documents",
          py::arg("query_embedding"),
          py::arg("doc_embeddings"));
}
```

### Step 6: Update CMakeLists.txt

Enable AVX512 compilation:

```cmake
# CMakeLists.txt (partial)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX512 instructions
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Check compiler
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # GCC and Clang
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mfma")
        message(STATUS "Enabled AVX512F and FMA instructions")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # Visual Studio
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        message(STATUS "Enabled AVX512 instructions (MSVC)")
    endif()
endif()

# Define the Python module
pybind11_add_module(vectorforge_cpp
    cpp/src/similarity.cpp
    # Add other source files as needed
)
```

**Compiler flags explained:**
- **`-mavx512f`**: Enable AVX512 Foundation instructions
- **`-mfma`**: Enable Fused Multiply-Add instructions
- **`/arch:AVX512`**: MSVC equivalent

### Step 7: Build and Test

```bash
cd /home/nick/projects/vectorforge
./build.sh  # Rebuild with AVX512 support

# Test
cd python
uv run python -c "
from vectorforge.vectorforge_cpp import cosine_similarity_batch
import numpy as np

q = np.random.rand(384).astype(np.float32)
docs = np.random.rand(100, 384).astype(np.float32)

scores = cosine_similarity_batch(q, docs)
print('Success! Scores shape:', scores.shape)
print('Sample scores:', scores[:5])
"
```

### Debugging Tips

If you get compilation errors:

1. **"error: '__m512' undeclared"**
   - Solution: Add `#include <immintrin.h>` at top of file

2. **"undefined reference to `_mm512_*`"**
   - Solution: Add `-mavx512f` flag to compiler (check CMakeLists.txt)

3. **"Illegal instruction" at runtime**
   - Your CPU doesn't support AVX512
   - Solution: Code should fall back to scalar (check `cpu_supports_avx512()`)

4. **Numerical differences from Python**
   - Expected! Floating point math is not associative
   - Solution: Use `np.allclose()` with tolerance (e.g., `rtol=1e-5`)

---

## 8. Common Pitfalls and How to Avoid Them

### Pitfall 1: Assuming Aligned Memory

**Problem:**
```cpp
// This can crash or be slow on unaligned data!
__m512 vec = _mm512_load_ps(data);  // Requires 64-byte alignment
```

**Solution:**
```cpp
// Use unaligned loads (slightly slower, but safe)
__m512 vec = _mm512_loadu_ps(data);  // Works with any alignment
```

**When to use each:**
- **Aligned (`_mm512_load_ps`)**: When you control allocation and ensure 64-byte alignment
- **Unaligned (`_mm512_loadu_ps`)**: With external data (NumPy arrays, user input)

**For VectorForge:** Always use `_mm512_loadu_ps()` since NumPy may not guarantee 64-byte alignment.

### Pitfall 2: Ignoring Remainder Elements

**Problem:**
```cpp
// Bug: if dim=384, this processes i=0,16,32,...,368 → last 16 elements missed!
for (int i = 0; i < dim; i += 16) {
    // ... SIMD code
}
// Missing: elements 368-383!
```

**Solution:**
```cpp
int i = 0;
// Process complete 16-element chunks
for (; i + 15 < dim; i += 16) {
    // SIMD code
}
// Handle remainder
for (; i < dim; i++) {
    // Scalar code
}
```

**For VectorForge:** dim=384 is divisible by 16, so no remainder. But be defensive!

### Pitfall 3: Data Dependencies Limiting Performance

**Problem:**
```cpp
// Accumulator has a dependency chain: each iteration waits for previous
__m512 sum = _mm512_setzero_ps();
for (int i = 0; i < n; i += 16) {
    __m512 a = _mm512_loadu_ps(&data[i]);
    sum = _mm512_add_ps(sum, a);  // Depends on previous sum!
}
```

**Solution:**
```cpp
// Multiple accumulators break dependency chain
__m512 sum0 = _mm512_setzero_ps();
__m512 sum1 = _mm512_setzero_ps();
__m512 sum2 = _mm512_setzero_ps();
__m512 sum3 = _mm512_setzero_ps();

for (int i = 0; i < n; i += 64) {
    sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(&data[i]));
    sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(&data[i+16]));
    sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(&data[i+32]));
    sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(&data[i+48]));
}

// Combine at end
sum0 = _mm512_add_ps(sum0, sum1);
sum2 = _mm512_add_ps(sum2, sum3);
sum0 = _mm512_add_ps(sum0, sum2);
```

### Pitfall 4: Mixing Data Types

**Problem:**
```cpp
double* data = ...;  // 64-bit doubles
__m512 vec = _mm512_loadu_ps(data);  // Wrong! Treats doubles as floats
```

**Solution:**
```cpp
// For float32: use _ps (packed single)
float* float_data = ...;
__m512 vec = _mm512_loadu_ps(float_data);

// For float64: use _pd (packed double)
double* double_data = ...;
__m512d vec = _mm512_loadu_pd(double_data);  // Note: __m512d and _pd
```

**For VectorForge:** We use `float32`, so always use `_ps` variants.

### Pitfall 5: Not Testing Numerical Accuracy

**Problem:**
Floating point operations aren't perfectly associative:
```python
# Python/NumPy (left-to-right addition)
sum = 0
for x in arr:
    sum += x

# SIMD (parallel reduction, different order)
# May give slightly different result!
```

**Solution:**
```python
# In tests, use tolerance
np.allclose(simd_result, numpy_result, rtol=1e-5, atol=1e-7)

# Not this:
simd_result == numpy_result  # Will often fail!
```

**For VectorForge:** Already handled in our tests with `np.allclose()`.

### Pitfall 6: Compiling for Wrong CPU

**Problem:**
```bash
# Compile on modern CPU with AVX512
g++ -mavx512f code.cpp -o program

# Run on older CPU without AVX512
./program
# → Illegal instruction (core dumped)
```

**Solution: Runtime Detection**
```cpp
if (cpu_supports_avx512()) {
    use_avx512_code();
} else if (cpu_supports_avx2()) {
    use_avx2_code();
} else {
    use_scalar_code();
}
```

**For VectorForge:** We implement this in Step 5 above.

### Pitfall 7: False Sharing in Multi-threaded Code

**Problem (future consideration):**
```cpp
// If we parallelize across documents:
#pragma omp parallel for
for (int i = 0; i < n_docs; i++) {
    results[i] = compute(i);  // results array elements on same cache line
}
```

Cache lines are 64 bytes. If `results[i]` and `results[i+1]` are on the same cache line and different threads write them, you get **cache thrashing**.

**Solution:**
```cpp
// Pad to cache line size (64 bytes = 16 floats)
struct alignas(64) PaddedResult {
    float value;
    char padding[60];  // Total 64 bytes
};

PaddedResult results[n_docs];
```

**For VectorForge:** Not needed yet (we're not multi-threaded), but good to know for future!

---

## 9. Benchmarking and Validation

### Creating a Comprehensive Benchmark

Create `cpp/benchmark/benchmark_simd.cpp`:

```cpp
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>

#include "../src/cpu_features.h"
#include "../src/similarity_scalar.cpp"
#include "../src/similarity_avx512.cpp"

using namespace std;
using namespace std::chrono;

// Benchmark harness
template<typename Func>
double benchmark(Func f, int iterations) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        f();
    }

    auto end = high_resolution_clock::now();
    duration<double, milli> elapsed = end - start;

    return elapsed.count() / iterations;  // ms per iteration
}

int main() {
    const int dim = 384;
    const int n_docs = 5000;
    const int iterations = 100;

    // Generate random data
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    vector<float> query(dim);
    vector<float> docs(n_docs * dim);

    for (auto& x : query) x = dis(gen);
    for (auto& x : docs) x = dis(gen);

    cout << "=====================================" << endl;
    cout << "VectorForge SIMD Benchmark" << endl;
    cout << "=====================================" << endl;
    cout << "Dimensions: " << dim << endl;
    cout << "Documents: " << n_docs << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "=====================================" << endl << endl;

    // Benchmark scalar implementation
    vector<float> results_scalar(n_docs);
    auto scalar_bench = [&]() {
        for (int i = 0; i < n_docs; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                sum += query[j] * docs[i * dim + j];
            }
            results_scalar[i] = sum;
        }
    };

    double scalar_time = benchmark(scalar_bench, iterations);
    cout << "Scalar implementation: " << fixed << setprecision(3)
         << scalar_time << " ms" << endl;

    // Benchmark AVX512 implementation (if available)
    if (cpu_supports_avx512()) {
        vector<float> results_avx512(n_docs);

        auto avx512_bench = [&]() {
            for (int i = 0; i < n_docs; i++) {
                results_avx512[i] = dot_product_avx512(
                    query.data(),
                    &docs[i * dim],
                    dim
                );
            }
        };

        double avx512_time = benchmark(avx512_bench, iterations);
        cout << "AVX512 implementation: " << fixed << setprecision(3)
             << avx512_time << " ms" << endl;

        double speedup = scalar_time / avx512_time;
        cout << "Speedup: " << fixed << setprecision(2)
             << speedup << "x" << endl;

        // Validate numerical accuracy
        double max_diff = 0.0;
        for (int i = 0; i < n_docs; i++) {
            double diff = abs(results_scalar[i] - results_avx512[i]);
            max_diff = max(max_diff, diff);
        }
        cout << "Max difference: " << scientific << setprecision(2)
             << max_diff << endl;

        if (max_diff < 1e-4) {
            cout << "✓ Numerical validation passed" << endl;
        } else {
            cout << "✗ Warning: Large numerical difference!" << endl;
        }
    } else {
        cout << "AVX512 not supported on this CPU" << endl;
    }

    cout << endl;

    return 0;
}
```

**Compile and run:**
```bash
g++ -std=c++17 -mavx512f -O3 cpp/benchmark/benchmark_simd.cpp -o benchmark_simd
./benchmark_simd
```

**Expected output:**
```
=====================================
VectorForge SIMD Benchmark
=====================================
Dimensions: 384
Documents: 5000
Iterations: 100
=====================================

Scalar implementation: 8.234 ms
AVX512 implementation: 0.612 ms
Speedup: 13.45x
Max difference: 2.38e-06
✓ Numerical validation passed
```

### Understanding the Results

**Why not exactly 16x speedup?**

Even though AVX512 processes 16 floats at once, real-world speedup is less due to:

1. **Memory bandwidth** (10-20% overhead)
   - CPU can compute faster than memory can deliver data
   - Mitigated by cache, but still a factor

2. **Loop overhead** (5-10% overhead)
   - Loop counter increment, branch prediction
   - Less significant with longer loops (our case)

3. **Horizontal reduction** (5-10% overhead)
   - Summing 16 vector lanes into one scalar
   - AVX512's `_mm512_reduce_add_ps` is fast, but not free

4. **Remainder handling** (negligible for us)
   - Scalar code for elements that don't fit in vectors
   - Zero for dim=384 (exactly 24×16)

**Typical speedups:**
- **Theoretical max:** 16x (perfect vectorization)
- **Realistic best case:** 12-14x (our target)
- **Typical real-world:** 8-12x (with memory bottlenecks)

### Validating Correctness

Create `python/tests/test_simd_accuracy.py`:

```python
import numpy as np
import pytest
from vectorforge.vectorforge_cpp import cosine_similarity_batch


def test_avx512_matches_numpy():
    """Verify AVX512 implementation matches NumPy reference."""
    np.random.seed(42)

    dim = 384
    n_docs = 1000

    query = np.random.randn(dim).astype(np.float32)
    docs = np.random.randn(n_docs, dim).astype(np.float32)

    # Normalize (as VectorForge does)
    query = query / np.linalg.norm(query)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    # C++ implementation
    cpp_scores = cosine_similarity_batch(query, docs)

    # NumPy reference
    numpy_scores = np.dot(docs, query)

    # Validate
    assert np.allclose(cpp_scores, numpy_scores, rtol=1e-5, atol=1e-7), \
        f"Max diff: {np.max(np.abs(cpp_scores - numpy_scores))}"


def test_edge_cases():
    """Test edge cases: zeros, ones, negatives."""

    # Test 1: Zero vector (edge case)
    query = np.zeros(384, dtype=np.float32)
    docs = np.random.randn(10, 384).astype(np.float32)

    scores = cosine_similarity_batch(query, docs)
    assert np.allclose(scores, 0.0), "Zero query should give zero scores"

    # Test 2: Identical vectors (should give 1.0 when normalized)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)

    docs = np.tile(query, (5, 1))  # 5 copies of query

    scores = cosine_similarity_batch(query, docs)
    assert np.allclose(scores, 1.0, atol=1e-6), "Identical vectors should give 1.0"

    # Test 3: Opposite vectors (should give -1.0)
    docs = np.tile(-query, (5, 1))

    scores = cosine_similarity_batch(query, docs)
    assert np.allclose(scores, -1.0, atol=1e-6), "Opposite vectors should give -1.0"


def test_large_scale():
    """Test with realistic VectorForge workload."""
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)

    docs = np.random.randn(10000, 384).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    cpp_scores = cosine_similarity_batch(query, docs)
    numpy_scores = np.dot(docs, query)

    assert cpp_scores.shape == (10000,)
    assert np.allclose(cpp_scores, numpy_scores, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Run validation:**
```bash
cd python
uv run pytest tests/test_simd_accuracy.py -v
```

---

## 10. Further Optimizations and Next Steps

You've now mastered SIMD/AVX512 basics! Here are advanced topics for further optimization:

### 10.1 Cache Blocking (Tiling)

For very large datasets that don't fit in cache:

```cpp
// Instead of processing all docs sequentially:
for (int i = 0; i < n_docs; i++) {
    compute_similarity(query, docs[i]);
}

// Process in blocks that fit in L3 cache:
const int BLOCK_SIZE = 1024;  // Tune based on cache size
for (int block = 0; block < n_docs; block += BLOCK_SIZE) {
    int block_end = min(block + BLOCK_SIZE, n_docs);
    for (int i = block; i < block_end; i++) {
        compute_similarity(query, docs[i]);
    }
}
```

**When useful:** n_docs > 10,000 and docs don't fit in L3 cache

### 10.2 Multi-threading with OpenMP

Parallelize across documents:

```cpp
#include <omp.h>

py::array_t<float> cosine_similarity_batch_parallel(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
) {
    // ... setup code ...

    #pragma omp parallel for
    for (int i = 0; i < n_docs; i++) {
        r(i) = dot_product_avx512(q_data, docs.data(i, 0), dim);
    }

    return result;
}
```

**Speedup potential:** 4-8x on 8-core CPU (on top of SIMD speedup!)

**Total speedup:** 12x (SIMD) × 6x (threading) = 72x vs naive Python

### 10.3 Prefetching

Hint to CPU to load next document while computing current one:

```cpp
for (int i = 0; i < n_docs; i++) {
    // Prefetch next document (if exists)
    if (i + 1 < n_docs) {
        __builtin_prefetch(docs.data(i + 1, 0), 0, 3);
        // 0 = read, 3 = high temporal locality
    }

    r(i) = dot_product_avx512(q_data, docs.data(i, 0), dim);
}
```

**Speedup:** 5-15% improvement in memory-bound scenarios

### 10.4 AVX512 Masking for Remainder Handling

Use AVX512 masks instead of scalar loop for remainder:

```cpp
inline float dot_product_avx512_masked(const float* a, const float* b, int n) {
    __m512 sum = _mm512_setzero_ps();

    int i = 0;
    // Process complete vectors
    for (; i + 15 < n; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(&a[i]);
        __m512 b_vec = _mm512_loadu_ps(&b[i]);
        sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
    }

    // Handle remainder with mask (no scalar loop!)
    if (i < n) {
        int remaining = n - i;
        __mmask16 mask = (1 << remaining) - 1;  // e.g., 0b0000111 for 3 elements

        __m512 a_vec = _mm512_maskz_loadu_ps(mask, &a[i]);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[i]);
        sum = _mm512_mask_fmadd_ps(sum, mask, a_vec, b_vec);
    }

    return _mm512_reduce_add_ps(sum);
}
```

**Benefit:** Slightly cleaner code, marginal performance gain

### 10.5 Quantization to INT8

For extreme performance, quantize float32 → int8:

```cpp
// AVX512 has VNNI (Vector Neural Network Instructions)
// Can process 64 int8 values at once (vs 16 float32)

// Quantize: float32 → int8
int8_t quantize(float x, float scale) {
    return static_cast<int8_t>(x * scale);
}

// Compute with int8, then dequantize result
```

**Speedup potential:** 2-4x faster than float32 SIMD

**Tradeoff:** Slight accuracy loss (usually acceptable for search ranking)

**When useful:** Extreme-scale deployments (millions of docs), GPU competition

### 10.6 GPU Acceleration (Future Direction)

For even larger scale, consider CUDA:

```cuda
__global__ void cosine_similarity_kernel(
    const float* query,
    const float* docs,
    float* results,
    int n_docs,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_docs) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += query[j] * docs[i * dim + j];
        }
        results[i] = sum;
    }
}
```

**Speedup potential:** 50-100x vs CPU (on high-end GPUs)

**When useful:** n_docs > 100,000, batch search workloads

---

## Summary and Learning Checklist

### Concepts Mastered ✓

- [x] **SIMD fundamentals**: One instruction, multiple data
- [x] **CPU architecture**: Vector units, registers, cache hierarchy
- [x] **AVX512 specifics**: 512-bit registers, 16 float32 operations
- [x] **Intrinsics**: `__m512`, `_mm512_loadu_ps`, `_mm512_fmadd_ps`, etc.
- [x] **Optimization techniques**: Multiple accumulators, FMA, reduction
- [x] **Runtime dispatch**: CPU feature detection, fallback code
- [x] **Numerical validation**: Tolerance-based testing

### Implementation Skills ✓

- [x] Write AVX512 dot product from scratch
- [x] Integrate SIMD into VectorForge cosine similarity
- [x] Handle edge cases (alignment, remainder elements)
- [x] Benchmark and validate performance
- [x] Debug common SIMD issues

### Can You Explain These to Someone Else? ✓

1. **Why is SIMD faster than regular code?**
   > "Instead of processing one number at a time, SIMD processes 16 numbers simultaneously with a single CPU instruction. It's like having 16 calculators working in parallel instead of one."

2. **What's the difference between AVX2 and AVX512?**
   > "AVX2 uses 256-bit registers (8 floats at once), while AVX512 uses 512-bit registers (16 floats at once). AVX512 also adds features like masking and better reduction instructions."

3. **How much speedup should we expect?**
   > "Theoretically 16x for AVX512 vs scalar, but realistically 10-14x due to memory bandwidth, loop overhead, and reduction costs. For VectorForge, we target 12x for raw dot products and 3-5x end-to-end."

4. **Why do we need runtime CPU detection?**
   > "Not all CPUs support AVX512 (especially older ones). Runtime detection ensures our code runs everywhere—fast on modern CPUs, correct on older ones."

5. **What's FMA and why does it matter?**
   > "Fused Multiply-Add computes `a*b+c` in one instruction instead of two. For dot products (which are all multiply-add operations), this nearly doubles throughput and improves numerical accuracy."

### Next Steps for VectorForge

**Immediate (this optimization):**
1. ✅ Implement AVX512 cosine similarity
2. ✅ Add comprehensive tests
3. ✅ Benchmark against Python baseline
4. ✅ Document performance gains

**Short-term (follow-up optimizations):**
1. Add AVX2 fallback (for CPUs without AVX512)
2. Add OpenMP multi-threading (4-8x additional speedup)
3. Profile with real workloads (identify any remaining bottlenecks)

**Long-term (scaling up):**
1. Consider GPU acceleration (for >100k documents)
2. Explore quantization (int8) for extreme performance
3. Implement approximate nearest neighbor (for >1M documents)

---

## Recommended Reading

**Books:**
- *Computer Organization and Design* (Patterson & Hennessy) - Chapter on parallelism
- *Software Optimization Resources* by Agner Fog (free PDFs) - Advanced SIMD techniques

**Online Resources:**
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Agner Fog's optimization manuals: https://www.agner.org/optimize/
- "SIMD for C++ Developers" (blog series): https://www.johndcook.com/blog/simd/

**Practice:**
- Implement other operations: matrix multiply, convolution, sorting
- Benchmark on your specific CPU and understand its characteristics
- Read VectorForge production code after optimization

---

## Glossary

| Term | Definition |
|------|------------|
| **SIMD** | Single Instruction, Multiple Data - parallel processing paradigm |
| **AVX** | Advanced Vector Extensions - Intel's SIMD instruction set |
| **AVX512** | 512-bit version of AVX (16 × float32) |
| **Intrinsic** | C/C++ function that maps directly to assembly instruction |
| **FMA** | Fused Multiply-Add - computes a×b+c in one instruction |
| **Lane** | One element position in a SIMD vector (0-15 for AVX512) |
| **Scalar** | Operating on single values (opposite of vector/SIMD) |
| **Alignment** | Memory address divisible by specific size (e.g., 64 bytes) |
| **Reduction** | Combining vector elements into scalar (e.g., sum all lanes) |
| **Throughput** | How many operations can start per cycle |
| **Latency** | How many cycles until operation completes |
| **Cache line** | 64-byte unit of cache storage |
| **Unrolling** | Manually expanding loop iterations for better performance |

---

**You're now ready to implement AVX512 optimization in VectorForge!**

Good luck, and remember: measure, optimize, validate. Always benchmark before and after to prove your optimization actually helps!

---

*End of Guide - Total Reading Time: ~45-60 minutes*
