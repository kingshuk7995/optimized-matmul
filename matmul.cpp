// // matmul.cpp
// // Optimized matrix multiplication for 12th Gen Intel Core i3-1215U
// // Compile with: g++ -O3 -march=native -ffast-math -fopenmp -shared
// -fPIC matmul.cpp -o matmul.so

#include <algorithm>   //for std::min
#include <cstring>     // For memset
#include <immintrin.h> // For AVX2 intrinsics
#include <omp.h>       // For OpenMP parallelization

#ifndef N_THREADS
#define N_THREADS 8
#endif

// Function to perform optimized matrix multiplication C = A * B
// A: m x k matrix
// B: k x n matrix
// C: m x n matrix (result)
extern "C" void matmul(const float *A, const float *B, float *C, int m, int k,
                       int n) {
  // Clear the output matrix first
  memset(C, 0, m * n * sizeof(float));

  // Set the number of threads to match your CPU (8 threads for i3-1215U)
  omp_set_num_threads(N_THREADS);

// Use OpenMP to parallelize the outer loop
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    // For each row of A
    for (int j = 0; j < n; j += 8) {
      // Process 8 columns of B at once when possible (using AVX2)
      if (j + 8 <= n) {
        for (int l = 0; l < k; l++) {
          // Broadcast single element from A
          __m256 a_val = _mm256_set1_ps(A[i * k + l]);

          // Load 8 elements from B
          __m256 b_vals = _mm256_loadu_ps(&B[l * n + j]);

          // Load current result
          __m256 c_vals = _mm256_loadu_ps(&C[i * n + j]);

          // Multiply and add (C += A * B)
          c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);

          // Store back the result
          _mm256_storeu_ps(&C[i * n + j], c_vals);
        }
      } else {
        // Handle remaining columns (less than 8) with scalar operations
        for (int jj = j; jj < n; jj++) {
          for (int l = 0; l < k; l++) {
            C[i * n + jj] += A[i * k + l] * B[l * n + jj];
          }
        }
        break; // Exit the j loop since we've handled the remainder
      }
    }
  }
}

// Function to transpose matrix B for better memory access patterns
// This can improve performance for certain matrix sizes
extern "C" void transpose_matrix(const float *src, float *dst, int rows,
                                 int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}

// Alternative implementation using matrix B transposition for better cache
// utilization
extern "C" void matmul_transposed(const float *A, const float *B, float *C,
                                  int m, int k, int n) {
  // Create a transposed copy of B
  // for better cache locality
  float *B_transposed = new float[k * n];
  transpose_matrix(B, B_transposed, k, n);

  // Clear the output matrix
  memset(C, 0, m * n * sizeof(float));

  // Set the number of threads to
  // match your CPU
  omp_set_num_threads(N_THREADS);

  // Use OpenMP to parallelize the
  // computation
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      __m256 sum = _mm256_setzero_ps();

      // Process 8 elements at a
      // time
      for (int l = 0; l < k - 7; l += 8) {
        __m256 a_vals = _mm256_loadu_ps(&A[i * k + l]);
        __m256 b_vals = _mm256_loadu_ps(&B_transposed[j * k + l]);

        // Use FMA instruction for
        // better performance
        sum = _mm256_fmadd_ps(a_vals, b_vals, sum);
      }

      // Horizontal sum of the 8
      // partial results
      float temp[8];
      _mm256_storeu_ps(temp, sum);
      float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                     temp[6] + temp[7];

      // Handle remaining elements
      for (int l = (k / 8) * 8; l < k; l++) {
        result += A[i * k + l] * B_transposed[j * k + l];
      }

      C[i * n + j] = result;
    }
  }

  delete[] B_transposed;
}

// Tiled matrix multiplication for
// better cache utilization
extern "C" void matmul_tiled(const float *A, const float *B, float *C, int m,
                             int k, int n) {
  // Clear the output matrix
  memset(C, 0, m * n * sizeof(float));

  // Define tile sizes based on
  // cache size L1 data cache on
  // i3-1215U is 48KB per core
  const int TILE_SIZE_M = 32;
  const int TILE_SIZE_N = 32;
  const int TILE_SIZE_K = 32;

  // Set the number of threads
  omp_set_num_threads(N_THREADS);

  // Use OpenMP to parallelize the
  // tiled computation
#pragma omp parallel for collapse(2)
  for (int i0 = 0; i0 < m; i0 += TILE_SIZE_M) {
    for (int j0 = 0; j0 < n; j0 += TILE_SIZE_N) {
      for (int k0 = 0; k0 < k; k0 += TILE_SIZE_K) {
        // Determine actual tile
        // sizes (handle
        // boundaries)
        int i_end = std::min(i0 + TILE_SIZE_M, m);
        int j_end = std::min(j0 + TILE_SIZE_N, n);
        int k_end = std::min(k0 + TILE_SIZE_K, k);

        // Process current tile
        for (int i = i0; i < i_end; i++) {
          for (int j = j0; j < j_end; j++) {
            float sum = 0.0f;
            for (int l = k0; l < k_end; l++) {
              sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] += sum;
          }
        }
      }
    }
  }
}

// by kingshuk7995
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

static constexpr int MC = 256;
static constexpr int NC = 256;
static constexpr int KC = 256;

#ifndef N_THREADS
#define N_THREADS omp_get_max_threads()
#endif

static inline int round_up8(int x) { return (x + 7) & ~7; }

static inline float *alloc_aligned_floats(std::size_t count) {
  void *p = nullptr;
  if (posix_memalign(&p, 64, count * sizeof(float)) != 0)
    return nullptr;
  return static_cast<float *>(p);
}

static inline void pack_A_8(const float *__restrict__ A, float *__restrict__ Ap,
                            int lda, int mc, int kc) {
  for (int i = 0; i < mc; i += 8) {
    float *Ablock = Ap + (std::size_t)i * kc;
    for (int p = 0; p < kc; ++p) {
      for (int ii = 0; ii < 8; ++ii) {
        Ablock[p * 8 + ii] = (i + ii < mc) ? A[(i + ii) * lda + p] : 0.0f;
      }
    }
  }
}

static inline void pack_B_8(const float *__restrict__ B, float *__restrict__ Bp,
                            int ldb, int kc, int nc) {
  for (int j = 0; j < nc; j += 8) {
    float *Bblock = Bp + (std::size_t)j * kc;
    for (int p = 0; p < kc; ++p) {
      for (int jj = 0; jj < 8; ++jj) {
        Bblock[p * 8 + jj] = (j + jj < nc) ? B[p * ldb + j + jj] : 0.0f;
      }
    }
  }
}

static inline void kernel_8x8(int kc, const float *__restrict__ Ap,
                              const float *__restrict__ Bp,
                              float *__restrict__ C, int ldc, int mc_actual,
                              int nc_actual) {
  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps();
  __m256 c5 = _mm256_setzero_ps();
  __m256 c6 = _mm256_setzero_ps();
  __m256 c7 = _mm256_setzero_ps();

  const float *a_ptr = Ap;
  const float *b_ptr = Bp;

#define KERNEL_CORE                                                            \
  {                                                                            \
    __m256 b0 = _mm256_load_ps(b_ptr);                                         \
    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), b0, c0);              \
    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 1), b0, c1);              \
    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 2), b0, c2);              \
    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 3), b0, c3);              \
    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 4), b0, c4);              \
    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 5), b0, c5);              \
    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 6), b0, c6);              \
    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 7), b0, c7);              \
    a_ptr += 8;                                                                \
    b_ptr += 8;                                                                \
  }

  int p = 0;
  for (; p + 3 < kc; p += 4) {
    KERNEL_CORE
    KERNEL_CORE
    KERNEL_CORE
    KERNEL_CORE
  }
  for (; p < kc; ++p) {
    KERNEL_CORE
  }

#undef KERNEL_CORE

  if (mc_actual == 8 && nc_actual == 8) {
    _mm256_storeu_ps(C + 0 * ldc,
                     _mm256_add_ps(c0, _mm256_loadu_ps(C + 0 * ldc)));
    _mm256_storeu_ps(C + 1 * ldc,
                     _mm256_add_ps(c1, _mm256_loadu_ps(C + 1 * ldc)));
    _mm256_storeu_ps(C + 2 * ldc,
                     _mm256_add_ps(c2, _mm256_loadu_ps(C + 2 * ldc)));
    _mm256_storeu_ps(C + 3 * ldc,
                     _mm256_add_ps(c3, _mm256_loadu_ps(C + 3 * ldc)));
    _mm256_storeu_ps(C + 4 * ldc,
                     _mm256_add_ps(c4, _mm256_loadu_ps(C + 4 * ldc)));
    _mm256_storeu_ps(C + 5 * ldc,
                     _mm256_add_ps(c5, _mm256_loadu_ps(C + 5 * ldc)));
    _mm256_storeu_ps(C + 6 * ldc,
                     _mm256_add_ps(c6, _mm256_loadu_ps(C + 6 * ldc)));
    _mm256_storeu_ps(C + 7 * ldc,
                     _mm256_add_ps(c7, _mm256_loadu_ps(C + 7 * ldc)));
  } else {
    alignas(32) float tmp[64];
    _mm256_store_ps(tmp + 0 * 8, c0);
    _mm256_store_ps(tmp + 1 * 8, c1);
    _mm256_store_ps(tmp + 2 * 8, c2);
    _mm256_store_ps(tmp + 3 * 8, c3);
    _mm256_store_ps(tmp + 4 * 8, c4);
    _mm256_store_ps(tmp + 5 * 8, c5);
    _mm256_store_ps(tmp + 6 * 8, c6);
    _mm256_store_ps(tmp + 7 * 8, c7);

    for (int i = 0; i < mc_actual; ++i) {
      for (int j = 0; j < nc_actual; ++j) {
        C[i * ldc + j] += tmp[i * 8 + j];
      }
    }
  }
}

extern "C" void matmul_kp(const float *A, const float *B, float *C, int m,
                          int k, int n) {
  if (m <= 0 || k <= 0 || n <= 0)
    return;

  std::memset(C, 0, (std::size_t)m * n * sizeof(float));

  omp_set_dynamic(0);
  omp_set_num_threads(N_THREADS);

  const int NC_PAD = round_up8(n);

  float *Bp = alloc_aligned_floats((std::size_t)KC * NC_PAD);
  if (!Bp)
    return;

  std::atomic<int> failed{0};

#pragma omp parallel shared(Bp, failed)
  {
    const int MC_PAD = round_up8(MC);
    float *Ap = alloc_aligned_floats((std::size_t)MC_PAD * KC);
    if (!Ap) {
      failed.store(1, std::memory_order_relaxed);
    }

    for (int jc = 0; jc < n; jc += NC) {
      const int nc = std::min(NC, n - jc);

      for (int kk = 0; kk < k; kk += KC) {
        const int kc = std::min(KC, k - kk);

#pragma omp single
        {
          if (!failed.load(std::memory_order_relaxed)) {
            pack_B_8(B + (std::size_t)kk * n + jc, Bp, n, kc, nc);
          }
        }

#pragma omp for schedule(static)
        for (int ic = 0; ic < m; ic += MC) {
          if (failed.load(std::memory_order_relaxed) || !Ap)
            continue;

          const int mc = std::min(MC, m - ic);
          pack_A_8(A + (std::size_t)ic * k + kk, Ap, k, mc, kc);

          for (int i = 0; i < mc; i += 8) {
            const float *Ablock = Ap + (std::size_t)i * kc;
            int mc_actual = std::min(8, mc - i);

            for (int j = 0; j < nc; j += 8) {
              const float *Bblock = Bp + (std::size_t)j * kc;
              int nc_actual = std::min(8, nc - j);

              kernel_8x8(kc, Ablock, Bblock,
                         C + (std::size_t)(ic + i) * n + jc + j, n, mc_actual,
                         nc_actual);
            }
          }
        }
      }
    }

    if (Ap)
      free(Ap);
  }

  free(Bp);
}