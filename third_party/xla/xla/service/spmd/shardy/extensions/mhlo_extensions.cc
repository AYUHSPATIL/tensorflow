/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/shardy/extensions/mhlo_extensions.h"

#include <cstdint>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

using ::mlir::ArrayRef;
using ::mlir::sdy::kNullDim;
namespace mhlo = ::mlir::mhlo;

namespace xla {
namespace sdy {

namespace {

struct RaggedDotShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          RaggedDotShardingRuleOpInterface, mhlo::RaggedDotOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::RaggedDotOp raggedDot = cast<mhlo::RaggedDotOp>(op);
    mhlo::RaggedDotDimensionNumbersAttr raggedDotDimNumbers =
        raggedDot.getRaggedDotDimensionNumbers();
    mhlo::DotDimensionNumbersAttr dotDimNumbers =
        raggedDotDimNumbers.getDotDimensionNumbers();

    ArrayRef<int64_t> lhsBatchingDims =
        dotDimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims =
        dotDimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dotDimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotDimNumbers.getRhsContractingDimensions();

    ArrayRef<int64_t> lhsRaggedDims =
        raggedDotDimNumbers.getLhsRaggedDimensions();
    CHECK_EQ(lhsRaggedDims.size(), 1);
    int64_t lhsRaggedDim = lhsRaggedDims[0];
    ArrayRef<int64_t> rhsGroupDims =
        raggedDotDimNumbers.getRhsGroupDimensions();

    mlir::sdy::OpShardingRuleBuilder builder(raggedDot);

    mlir::RankedTensorType lhsType = raggedDot.getLhs().getType();
    mlir::RankedTensorType rhsType = raggedDot.getRhs().getType();
    mlir::RankedTensorType resultType = raggedDot.getResult().getType();
    const int64_t lhsRank = lhsType.getRank();
    const int64_t rhsRank = rhsType.getRank();

    int64_t outputDim =
        llvm::is_contained(lhsContractingDims, lhsRaggedDim) ? 1 : 0;

    // batching dimensions
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsBatchingDims, rhsBatchingDims)) {
      builder.addFactor({lhsDim, rhsDim, lhsDim == lhsRaggedDim ? kNullDim : 0},
                        outputDim++, lhsType.getDimSize(lhsDim));
    }

    // lhs non-contracting dimensions
    for (int64_t i = 0; i < lhsRank; i++) {
      if (!llvm::is_contained(lhsContractingDims, i) &&
          !llvm::is_contained(lhsBatchingDims, i)) {
        builder.addFactor({i, kNullDim, kNullDim}, outputDim++,
                          lhsType.getDimSize(i));
      }
    }

    // rhs non-contracting dimensions
    for (int64_t i = 0; i < rhsRank; i++) {
      if (!llvm::is_contained(rhsContractingDims, i) &&
          !llvm::is_contained(rhsBatchingDims, i) &&
          !llvm::is_contained(rhsGroupDims, i)) {
        builder.addFactor({kNullDim, i, kNullDim}, outputDim++,
                          rhsType.getDimSize(i));
      }
    }

    // contracting dimensions
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsContractingDims, rhsContractingDims)) {
      builder.addFactor({lhsDim, rhsDim, kNullDim}, kNullDim,
                        lhsType.getDimSize(lhsDim),
                        mlir::sdy::FactorType::kReduction);
    }

    // group dimension
    if (llvm::is_contained(lhsContractingDims, lhsRaggedDim)) {
      builder.addFactor({kNullDim, kNullDim, 1}, 0, resultType.getDimSize(0));
    } else if (!llvm::is_contained(lhsBatchingDims, lhsRaggedDim)) {
      // the ragged dimension is an lhs non-contracting dim
      CHECK_EQ(rhsGroupDims.size(), 1);
      int64_t rhsGroupDim = rhsGroupDims[0];
      builder.addFactor({kNullDim, rhsGroupDim, 1}, kNullDim,
                        rhsType.getDimSize(rhsGroupDim));
    }

    return builder.build();
  }
};

}  // namespace

void registerMhloExtensions(mlir::MLIRContext* ctx) {
  // Ensure dialect is loaded before attaching interfaces.
  ctx->loadDialect<mhlo::MhloDialect>();
  mhlo::RaggedDotOp::attachInterface<RaggedDotShardingRuleOpInterface>(*ctx);
}

}  // namespace sdy
}  // namespace xla
