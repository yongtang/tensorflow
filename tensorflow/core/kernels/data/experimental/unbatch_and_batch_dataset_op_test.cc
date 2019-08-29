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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "unbatch_and_batch_dataset";

class UnbatchAndBatchDatasetParams : public DatasetParams {
 public:
  UnbatchAndBatchDatasetParams(
      RangeDatasetParams range_dataset_params,
      std::vector<Tensor> other_arguments, int64 batch_size,
      int64 num_parallel_calls, bool drop_remainder,
      FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      bool preserve_cardinality, DataTypeVector output_dtypes,
      std::vector<PartialTensorShape> output_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        input_dataset_params(std::move(range_dataset_params)),
        other_arguments(std::move(other_arguments)),
        batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
        num_parallel_calls(
            CreateTensor<int64>(TensorShape({}), {num_parallel_calls})),
        drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        func(std::move(func)),
        func_lib(std::move(func_lib)),
        type_arguments(std::move(type_arguments)),
        preserve_cardinality(preserve_cardinality) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (!IsDatasetTensor(input_dataset)) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset)};
    for (auto& argument : other_arguments) {
      inputs->emplace_back(TensorValue(&argument));
    }
    inputs->insert(inputs->end(),
                   {TensorValue(&batch_size), TensorValue(&num_parallel_calls),
                    TensorValue(&drop_remainder)});
    return Status::OK();
  }

  RangeDatasetParams input_dataset_params;
  Tensor input_dataset;
  std::vector<Tensor> other_arguments;
  Tensor batch_size;
  Tensor num_parallel_calls;
  Tensor drop_remainder;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  DataTypeVector type_arguments;
  bool preserve_cardinality;
};

class UnbatchAndBatchDatasetOpTest
    : public DatasetOpsTestBaseV2<UnbatchAndBatchDatasetParams> {
 public:
  Status Initialize(
      UnbatchAndBatchDatasetParams* map_and_batch_dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime(
        map_and_batch_dataset_params->func_lib, cpu_num_));

    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*map_and_batch_dataset_params, &dataset_kernel_));
    TF_RETURN_IF_ERROR(
        MakeRangeDataset(map_and_batch_dataset_params->input_dataset_params,
                         &map_and_batch_dataset_params->input_dataset));
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(map_and_batch_dataset_params->MakeInputs(&inputs));
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(
        iterator_ctx_.get(), map_and_batch_dataset_params->iterator_prefix,
        &iterator_));
    return Status::OK();
  }

 protected:
  Status MakeDatasetOpKernel(
      const UnbatchAndBatchDatasetParams& map_and_batch_dataset_params,
      std::unique_ptr<OpKernel>* map_and_batch_kernel) override {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(UnbatchAndBatchDatasetOp::kDatasetType),
        {UnbatchAndBatchDatasetOp::kInputDataset, UnbatchAndBatchDatasetOp::kBatchSize,
         UnbatchAndBatchDatasetOp::kNumParallelCalls,
         UnbatchAndBatchDatasetOp::kDropRemainder},
        {{UnbatchAndBatchDatasetOp::kFunc, map_and_batch_dataset_params.func},
         {UnbatchAndBatchDatasetOp::kTarguments,
          map_and_batch_dataset_params.type_arguments},
         {UnbatchAndBatchDatasetOp::kOutputTypes,
          map_and_batch_dataset_params.output_dtypes},
         {UnbatchAndBatchDatasetOp::kOutputShapes,
          map_and_batch_dataset_params.output_shapes},
         {UnbatchAndBatchDatasetOp::kPreserveCardinality,
          map_and_batch_dataset_params.preserve_cardinality}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, map_and_batch_kernel));
    return Status::OK();
  }
};

FunctionDefHelper::AttrValueWrapper UnbatchFunc(const string& func_name,
                                            const DataType& dtype) {
  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

// test case 1: num_parallel_calls = 1, drop_remainder = true,
// preserve_cardinality = false, UnbatchFunc = XTimesTwo
UnbatchAndBatchDatasetParams UnbatchAndBatchDatasetParams1() {
  return {/*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
          /*other_arguments=*/{},
          /*batch_size=*/2,
          /*num_parallel_calls=*/1,
          /*drop_remainder=*/true,
          /*func=*/UnbatchFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*type_arguments*/ {},
          /*preserve_cardinality=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({2})},
          /*node_name=*/kNodeName};
}

// test case 2: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = true, UnbatchFunc = XTimesTwo
UnbatchAndBatchDatasetParams UnbatchAndBatchDatasetParams2() {
  return {/*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
          /*other_arguments=*/{},
          /*batch_size=*/2,
          /*num_parallel_calls=*/2,
          /*drop_remainder=*/true,
          /*func=*/UnbatchFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*type_arguments*/ {},
          /*preserve_cardinality=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({2})},
          /*node_name=*/kNodeName};
}

// test case 3: num_parallel_calls = 3, drop_remainder = false,
// preserve_cardinality = true, UnbatchFunc = XTimesFour
UnbatchAndBatchDatasetParams UnbatchAndBatchDatasetParams3() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/3,
      /*drop_remainder=*/false,
      /*func=*/UnbatchFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName};
}

// test case 4: num_parallel_calls = 4, drop_remainder = true,
// preserve_cardinality = false, UnbatchFunc = XTimesTwo
UnbatchAndBatchDatasetParams UnbatchAndBatchDatasetParams4() {
  return {/*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
          /*other_arguments=*/{},
          /*batch_size=*/2,
          /*num_parallel_calls=*/4,
          /*drop_remainder=*/true,
          /*func=*/UnbatchFunc("XTimesTwo", DT_INT64),
          /*func_lib=*/{test::function::XTimesTwo()},
          /*type_arguments*/ {},
          /*preserve_cardinality=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({2})},
          /*node_name=*/kNodeName};
}

// test case 5: num_parallel_calls = kAutotune, drop_remainder = true,
// preserve_cardinality = true, UnbatchFunc = XTimesTwo
UnbatchAndBatchDatasetParams UnbatchAndBatchDatasetParams5() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/model::kAutotune,
      /*drop_remainder=*/true,
      /*func=*/UnbatchFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName};
}

// test case 6: num_parallel_calls = 4, drop_remainder = false,
// preserve_cardinality = true, UnbatchFunc = XTimesFour
UnbatchAndBatchDatasetParams UnbatchAndBatchDatasetParams6() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/4,
      /*drop_remainder=*/false,
      /*func=*/UnbatchFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName};
}

UnbatchAndBatchDatasetParams InvalidNumParallelCallsUnbatchAndBatchDatasetParams() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
      /*other_arguments=*/{},
      /*batch_size=*/2,
      /*num_parallel_calls=*/-4,
      /*drop_remainder=*/false,
      /*func=*/UnbatchFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName};
}

UnbatchAndBatchDatasetParams InvalidBatchSizeUnbatchAndBatchDatasetParams() {
  return {
      /*range_dataset_params=*/{/*start=*/0, /*stop=*/10, /*step=*/2},
      /*other_arguments=*/{},
      /*batch_size=*/-2,
      /*num_parallel_calls=*/2,
      /*drop_remainder=*/false,
      /*func=*/UnbatchFunc("XTimesFour", DT_INT64),
      /*func_lib=*/{test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments*/ {},
      /*preserve_cardinality=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<UnbatchAndBatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/UnbatchAndBatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams5(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 8}, {16, 24}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams6(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}}};
}

ITERATOR_GET_NEXT_TEST_P(UnbatchAndBatchDatasetOpTest, UnbatchAndBatchDatasetParams,
                         GetNextTestCases())

TEST_F(UnbatchAndBatchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = UnbatchAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(UnbatchAndBatchDatasetOp::kDatasetType)));
}

TEST_F(UnbatchAndBatchDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = UnbatchAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<UnbatchAndBatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/UnbatchAndBatchDatasetParams1(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams2(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams3(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams4(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams5(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams6(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(UnbatchAndBatchDatasetOpTest, UnbatchAndBatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<UnbatchAndBatchDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/UnbatchAndBatchDatasetParams1(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams2(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams3(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams4(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams5(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams6(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(UnbatchAndBatchDatasetOpTest, UnbatchAndBatchDatasetParams,
                           CardinalityTestCases())

TEST_F(UnbatchAndBatchDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = UnbatchAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<UnbatchAndBatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/UnbatchAndBatchDatasetParams1(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams2(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams3(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams4(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams5(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams6(),
           /*expected_output_shapes=*/
           {PartialTensorShape({2})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(UnbatchAndBatchDatasetOpTest,
                              UnbatchAndBatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(UnbatchAndBatchDatasetOpTest, IteratorPrefix) {
  auto dataset_params = UnbatchAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      UnbatchAndBatchDatasetOp::kDatasetType, dataset_params.iterator_prefix)));
}

std::vector<IteratorSaveAndRestoreTestCase<UnbatchAndBatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/UnbatchAndBatchDatasetParams1(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams2(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams3(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams4(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 4}, {8, 12}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams5(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({2}), {{0, 8}, {16, 24}})},
          {/*dataset_params=*/UnbatchAndBatchDatasetParams6(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({2}), {0, 8}),
            CreateTensor<int64>(TensorShape({2}), {16, 24}),
            CreateTensor<int64>(TensorShape({1}), {32})}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(UnbatchAndBatchDatasetOpTest,
                                 UnbatchAndBatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(UnbatchAndBatchDatasetOpTest, InvalidBatchSize) {
  auto dataset_params = InvalidBatchSizeUnbatchAndBatchDatasetParams();
  EXPECT_EQ(Initialize(&dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(UnbatchAndBatchDatasetOpTest, InvalidNumParallel) {
  auto dataset_params = InvalidNumParallelCallsUnbatchAndBatchDatasetParams();
  EXPECT_EQ(Initialize(&dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
