/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/unbatch_and_batch_dataset_op.h"

#include <atomic>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace data {
namespace experimental {

constexpr char kDatasetName[] = "UnbatchAndBatch";

UnbatchAndBatchDatasetOp::UnbatchAndBatchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
}

class UnbatchAndBatchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int64 batch_size,
          bool drop_remainder,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        batch_size_(batch_size),
        drop_remainder_(drop_remainder),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetName)});
  }

  const DataTypeVector& output_dtypes() const override {
    return output_types_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return "UnbatchAndBatchDatasetOp::Dataset";
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* batch_size_node;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
    Node* drop_remainder_node;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder_node));

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, batch_size_node, drop_remainder_node}, output));
    return Status::OK();
  }

  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params)
        , current_index_(0)
        , current_batch_size_(0) {}

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (!input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }
      *end_of_sequence = false;

      int64 chunk_read = 0;

      out_tensors->clear();
      std::vector<Tensor> elements;

      while (!*end_of_sequence) {
        if (current_index_ < current_batch_size_) {
          // If out_tensors->size() == 0, then this is the first time
          // we arrive here for each batched output tensors. We initialize
          // the out_tensors to pre-allocate the storage. Since batched
          // output tensors may not be filled by one input tensors, we may
          // re-enter here again but We will not re-initialize (until
          // next batched output tensors is needed).
          if (out_tensors->size() == 0) {

            out_tensors->reserve(in_tensors_.size());
            elements.reserve(in_tensors_.size());
            for (size_t i = 0; i < in_tensors_.size(); ++i) {
              TensorShape shape = in_tensors_[i].shape();

              shape.RemoveDim(0);
              elements.emplace_back(ctx->allocator({}), in_tensors_[i].dtype(), shape);

              shape.InsertDim(0, dataset()->batch_size_);
              out_tensors->emplace_back(ctx->allocator({}), in_tensors_[i].dtype(), shape);
            }
          }

          if (out_tensors->size() != in_tensors_.size()) {
            return errors::InvalidArgument("number tensors should match previous one, ", in_tensors_.size(), " vs. ", out_tensors->size());
          }

          int64 chunk_to_read = (current_batch_size_ - current_index_) < (dataset()->batch_size_ - chunk_read) ? (current_batch_size_ - current_index_) : (dataset()->batch_size_ - chunk_read);
          for (int i = 0; i < in_tensors_.size(); ++i) {
            // TODO: concurrent copy?
            for (int64 r = 0; r < chunk_to_read; ++r) {
              TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                  &in_tensors_[i], &elements[i], current_index_ + r));
              TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
                  elements[i], &(*out_tensors)[i], chunk_read + r));
            }
          }

          chunk_read += chunk_to_read;
          current_index_ += chunk_to_read;
          if (chunk_read == dataset()->batch_size_) {
            *end_of_sequence = false;
            return Status::OK();
          }
        }

        // If we get here, the current input tensors don't have enough available
        // slices to construct a full output batch, so we pull more input.
        current_index_ = 0;
        current_batch_size_ = 0;
        in_tensors_.clear();
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &in_tensors_, end_of_sequence));
        if (!*end_of_sequence) {
          for (size_t i = 0; i < in_tensors_.size(); ++i) {
            if (in_tensors_[i].dims() == 0) {
              return errors::InvalidArgument(
                  "Input element must have a non-scalar value in each "
                  "component.");
            }
            if (in_tensors_[i].dim_size(0) != in_tensors_[0].dim_size(0)) {
              return errors::InvalidArgument(
                  "Input element must have the same batch size in each "
                  "component. Component 0 had size ",
                  in_tensors_[0].dim_size(0), " but component ", i,
                  " had size, ", in_tensors_[i].dim_size(0), ".");
            }
          }
          current_batch_size_ = in_tensors_[0].dim_size(0);
        }
      }

      // Before we return with Status::OK() and *end_of_sequence = false,
      // we need to separate several situations:
      // 1) chunk_read == 0:
      //    We hit the dataset end and no data copied to out_tensors.
      //    Nothing need to be done other than out_tensors might have been
      //    allocated (because we don't know at the time of allocation),
      //    so we need to call out_tensors->clear();
      // 2) chunk_read > 0 and chunk_read < dataset()->batch_size_:
      //    We hit the dataset end and out_tensors are not filled to
      //    `batch size`. Depending on the mode:
      //    a) drop_remainder = True: clear out the out_tensors, and return.
      //    b) drop_remainder = False: leave as is, except we have
      //       to change the shape of out_tensors to `[chunk_read, ...]`.
      //       because, out_tensors was created  as `[batch_size, ...]`.
      // 3) chunk_read > 0 and chunk_read == dataset()->batch_size_:
      //    Nothing needs to be done, continue.
      if (chunk_read > 0) {
        // shape may need to be adjusted if:
        // chunk_read < dataset()->batch_size_  (and drop_remainder is False)
        if (chunk_read < dataset()->batch_size_) {
          // No need to resieze with drop_reminder = True
          if (dataset()->drop_remainder_) {
            out_tensors->clear();
            input_impl_.reset();
            *end_of_sequence = true;
            return Status::OK();
          }
          for (int i = 0; i < out_tensors->size(); ++i) {
            TensorShape shape = (*out_tensors)[i].shape();
            shape.set_dim(0, chunk_read);
            Tensor value_tensor;
            value_tensor.CopyFrom((*out_tensors)[i], shape);
            (*out_tensors)[i] = std::move(value_tensor);
          }
        }
        *end_of_sequence = false;
        return Status::OK();
      }
      out_tensors->clear();
      input_impl_.reset();
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      // Unbatch assumes that all input components have the same leading
      // dimension. If it is statically known for any component, we model the
      // transformation using `KnownRatio`. Otherwise, we use `UnknownRatio`.
      for (auto& shape : dataset()->input_->output_shapes()) {
        if (shape.dims() > 0 && shape.dim_size(0) > 0) {
          return model::MakeKnownRatioNode(
              std::move(args), 1.0 / static_cast<double>(shape.dim_size(0)) * dataset()->batch_size_);
        }
      }
      return model::MakeUnknownRatioNode(std::move(args));
    }

   private:
    mutex mu_;
    // Next slice to read of the current input batch of tensors.
    int64 current_index_ GUARDED_BY(mu_);
    // The batch size associated with the input tensors (in_tensors_).
    int64 current_batch_size_ GUARDED_BY(mu_);
    // The input batch of tensors, saved here as unbatch().batch(n)
    // might be cross the boundary.
    std::vector<Tensor> in_tensors_ GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
  };

  const DatasetBase* const input_;
  const int64 batch_size_;
  const bool drop_remainder_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

void UnbatchAndBatchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                 DatasetBase** output) {
  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));
  OP_REQUIRES(
      ctx, batch_size > 0,
      errors::InvalidArgument("batch_size must be greater than zero."));

  bool drop_remainder;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument(ctx, "drop_remainder", &drop_remainder));

  *output = new Dataset(ctx, input, batch_size, drop_remainder,
                        output_types_, output_shapes_);
}

REGISTER_KERNEL_BUILDER(
    Name("UnbatchAndBatchDataset").Device(DEVICE_CPU),
    UnbatchAndBatchDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("UnbatchAndBatchDataset");

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
