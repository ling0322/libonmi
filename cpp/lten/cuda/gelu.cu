// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cuda_fp16.h>

#include "lten/cuda/common.h"
#include "lten/cuda/gelu.h"

namespace lten {
namespace op {
namespace cuda {

__device__ constexpr float Sqrt2 = 1.4142136f;

__global__ void geluKernel3D(PackedSubtensor<const half, 3> A, PackedSubtensor<half, 3> C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (z >= C.getShape(0) || y >= C.getShape(1) || x >= C.getShape(2)) return;

  float a = __half2float(A[z][y][x]);
  float c = a * 0.5f * (1.0f + erf(a / Sqrt2));

  C[z][y][x] = __float2half(c);
}

Tensor gelu3D(const Tensor &tensor) {
  Tensor C = createCudaTensorHalf(tensor.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.z = C.getShape(0);
  d.y = C.getShape(1);
  d.x = (C.getShape(2) + blockSize - 1) / blockSize;

  geluKernel3D<<<d, blockSize>>>(tensor, C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor gelu(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);

  if (tensor.getDim() == 3) return gelu3D(tensor);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace lten
