// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "lten/cpu/lookup.h"

#include "lten/cpu/accessor.h"
#include "lten/cpu/common.h"
#include "lten/cpu/copy.h"
#include "lten/cpu/kernel/interface.h"
#include "lten/cpu/print.h"
#include "lten/cpu/tensor.h"

namespace lten {
namespace op {
namespace cpu {

template<typename T>
Tensor lookupKernel2D(const Tensor &table, const Tensor &indices) {
  CHECK(table.getDim() == 2 && indices.getDim() == 2);

  int vocabSize = table.getShape(0);
  int d0 = indices.getShape(0);
  int d1 = indices.getShape(1);
  int embdDim = table.getShape(1);
  Tensor xC = tensor(lut::makeConstSpan({d0, d1, embdDim}), DType::getType<T>());

  TensorAccessor<const T, 2> A = table;
  TensorAccessor<const LongType, 2> B = indices;
  TensorAccessor<T, 3> C = xC;

  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      int64_t index = B[i][j];
      CHECK(index < vocabSize) << "indices out of range";

      copyVector(C[i][j], A[index]);
    }
  }

  return xC;
}

template<typename SrcT, typename DestT>
Tensor lookupQuantizedKernel2D(const Tensor &table, const Tensor &indices) {
  CHECK(table.getDim() == 2 && table.getShape(1) % DType::getType<SrcT>().getGroupSize() == 0);
  const TensorData *embdData = table.getDataObject();

  int vocabSize = table.getShape(0);
  int d0 = indices.getShape(0);
  int d1 = indices.getShape(1);
  int embdDim = table.getShape(1);
  Tensor xC = tensor(lut::makeConstSpan({d0, d1, embdDim}), DType::getType<DestT>());

  TensorAccessor<const LongType, 2> B = indices;
  TensorAccessor<DestT, 3> C = xC;

  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      int64_t index = B[i][j];
      CHECK(index < vocabSize) << "indices out of range";

      applyDequant(embdDim * index, embdDim, embdData, C[i][j].getData());
    }
  }

  return xC;
}

Tensor lookup(const Tensor &table, const Tensor &indices) {
  if (table.getDType() == DType::kFloat) return lookupKernel2D<float>(table, indices);
  if (table.getDType() == DType::kQInt4x32)
    return lookupQuantizedKernel2D<QInt4x32, DefaultFloatType>(table, indices);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (table.getDType() == DType::kFloat16) return lookupKernel2D<Float16>(table, indices);
#endif
  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
