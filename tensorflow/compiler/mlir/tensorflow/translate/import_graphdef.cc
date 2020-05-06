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

#include "tensorflow/compiler/mlir/tensorflow/translate/import_graphdef.h"

#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_base.h"

namespace tensorflow {
using mlir::TensorType;
using mlir::TF::VarHandleOp;
using mlir::tf_saved_model::GlobalTensorOp;
using stream_executor::port::StatusOr;

StatusOr<mlir::OwningModuleRef> GraphDefImporter::Convert(
    mlir::MLIRContext* context, const Graph& graph,
    const GraphDebugInfo& debug_info, const FunctionLibraryDefinition& flib_def,
    const GraphImportConfig& specs, llvm::StringRef func_name) {
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  NameUniquifier function_name_uniquifier(flib_def);

  GraphDefImporter importer(flib_def, debug_info, specs, module.get(),
                            &tf_name_to_mlir_name, &function_name_uniquifier);

  TF_RETURN_IF_ERROR(importer.PrepareConvert(graph));

  mlir::FunctionType func_type;
  absl::InlinedVector<OutputTensor, 4> arg_nodes;
  absl::InlinedVector<OutputTensor, 4> ret_nodes;
  absl::InlinedVector<Node*, 4> control_ret_nodes;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 4> resource_arg_unique_ids;
  llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
  if (specs.graph_as_function) {
    if (specs.prune_unused_nodes || !specs.inputs.empty() ||
        !specs.outputs.empty())
      return errors::InvalidArgument(
          "Pruning of graph is currently unsupported when the main graph is "
          "converted to a function.");

    TF_ASSIGN_OR_RETURN(
        func_type,
        importer.GetArgsRetsAndTypesFromFunctionGraph(
            context, &arg_nodes, &ret_nodes, &resource_arg_unique_ids));

    TF_RETURN_IF_ERROR(importer.GetControlRetsFromGraph(specs.control_outputs,
                                                        &control_ret_nodes));

    if (!arg_nodes.empty() || !ret_nodes.empty() ||
        !control_ret_nodes.empty()) {
      mlir::Builder b(context);
      std::string s;
      llvm::raw_string_ostream ss(s);
      auto node_name = [&](const OutputTensor& tensor) {
        ss << tensor.node->name();
      };
      llvm::interleave(arg_nodes, ss, node_name, ",");
      auto inputs = b.getNamedAttr("inputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(ret_nodes, ss, node_name, ",");
      auto outputs = b.getNamedAttr("outputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.control_outputs, ss, ",");
      auto control_outputs =
          b.getNamedAttr("control_outputs", b.getStringAttr(ss.str()));

      attrs.push_back(b.getNamedAttr(
          "tf.entry_function",
          b.getDictionaryAttr({inputs, outputs, control_outputs})));
    }
  } else {
    // Collects the argument and return nodes by looking up the node names
    // specified by the user.
    TF_ASSIGN_OR_RETURN(func_type, importer.InferMainFunctionType(
                                       specs, context, &arg_nodes, &ret_nodes));

    TF_RETURN_IF_ERROR(importer.GetControlRetsFromGraph(specs.control_outputs,
                                                        &control_ret_nodes));

    // TODO(prakalps): Refactor to keep tf.entry_function attribute encoding and
    // decoding in a centralized place.
    // Record the input and output mapping.
    if (!specs.inputs.empty() || !specs.outputs.empty() ||
        !specs.control_outputs.empty()) {
      mlir::Builder b(context);
      std::string s;
      llvm::raw_string_ostream ss(s);
      llvm::interleave(
          specs.inputs, ss,
          [&](const std::pair<std::string, ArrayInfo>& v) { ss << v.first; },
          ",");
      auto inputs = b.getNamedAttr("inputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.outputs, ss, ",");
      auto outputs = b.getNamedAttr("outputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.control_outputs, ss, ",");
      auto control_outputs =
          b.getNamedAttr("control_outputs", b.getStringAttr(ss.str()));

      attrs.push_back(b.getNamedAttr(
          "tf.entry_function",
          b.getDictionaryAttr({inputs, outputs, control_outputs})));
    }
  }

  // Record version info.
  PopulateTfVersions(module.get(), graph.versions());

  TF_RETURN_IF_ERROR(importer.ImporterBase::Convert(
      func_name, func_type, arg_nodes, ret_nodes, control_ret_nodes, attrs,
      resource_arg_unique_ids));
  return module;
}

StatusOr<mlir::FunctionType> GraphDefImporter::InferMainFunctionType(
    const GraphImportConfig& specs, mlir::MLIRContext* context,
    absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes) {
  // Find all the input nodes and output nodes.
  // Feeds have been remapped to single output nodes (Placeholder), so an exact
  // name match is sufficient.
  absl::flat_hash_map<absl::string_view, int> inputs;
  for (auto input_and_idx : llvm::enumerate(specs.inputs)) {
    TensorId tensor = ParseTensorName(input_and_idx.value().first);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      inputs.insert({remapped_it->second, input_and_idx.index()});
    } else {
      inputs.insert({tensor.node(), input_and_idx.index()});
    }
  }

  absl::flat_hash_set<absl::string_view> output_node_names;
  std::vector<TensorId> outputs;
  output_node_names.reserve(specs.outputs.size());
  for (const auto& output : specs.outputs) {
    TensorId tensor = ParseTensorName(output);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      output_node_names.insert(remapped_it->second);
      outputs.push_back({remapped_it->second, 0});
    } else {
      output_node_names.insert(tensor.node());
      outputs.push_back(tensor);
    }
  }

  if (!inputs.empty() || !outputs.empty()) {
    arg_nodes->resize(inputs.size());
    ret_nodes->resize(outputs.size());

    for (Node* n : GetOrderedNodes()) {
      // Handle inputs/arguments.
      auto input_it = inputs.find(n->name());
      if (input_it != inputs.end()) {
        (*arg_nodes)[input_it->second] = {n, 0};
      }

      // Handle outputs/returns.
      if (output_node_names.contains(n->name())) {
        for (int i = 0, e = outputs.size(); i != e; ++i) {
          TensorId tensor = outputs[i];
          if (n->name() != tensor.node()) continue;
          (*ret_nodes)[i] = {n, tensor.index()};
        }
      }
    }
  }

  // Starts to construct the function type.
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  arg_types.reserve(specs.inputs.size());
  int i = 0;
  for (const auto& it : specs.inputs) {
    Node* arg_node = arg_nodes->at(i).node;
    if (arg_node == nullptr) {
      return errors::InvalidArgument("Input ", it.first,
                                     " was not found in graph");
    }
    mlir::Type element_type;
    const auto& node_info = it.second;
    DataType imported_dtype = node_info.imported_dtype;
    // Uses the existing output type of the arg node if the data type of the
    // the node isn't specified through the import configuration.
    if (imported_dtype == DT_INVALID) {
      imported_dtype = arg_node->output_type(0);
      if (imported_dtype == DT_INVALID) {
        return errors::InvalidArgument("Input ", i, "has invalid data type");
      }
    }
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(imported_dtype, builder, &element_type));
    if (node_info.shape.unknown_rank()) {
      arg_types.push_back(mlir::UnrankedTensorType::get(element_type));
    } else {
      llvm::SmallVector<int64_t, 4> shape;
      TF_RETURN_IF_ERROR(ConvertToMlirShape(node_info.shape, &shape));
      arg_types.push_back(mlir::RankedTensorType::get(shape, element_type));
    }
    i++;
  }

  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(specs.outputs.size());
  for (int i = 0, e = specs.outputs.size(); i != e; ++i) {
    if (ret_nodes->at(i).node == nullptr) {
      return errors::InvalidArgument("Output ", specs.outputs[i],
                                     " was not found in graph");
    }
  }
  for (const auto& ret : *ret_nodes) {
    if (ret.node->num_outputs() <= ret.index) {
      return errors::InvalidArgument("Invalid output index ", ret.index,
                                     " specified for node: ", ret.node->name());
    }
    TF_ASSIGN_OR_RETURN(auto type,
                        InferOutputType(*ret.node, ret.index, builder));
    ret_types.push_back(type);
  }

  return builder.getFunctionType(arg_types, ret_types);
}

StatusOr<mlir::FunctionType>
GraphDefImporter::GetArgsRetsAndTypesFromFunctionGraph(
    mlir::MLIRContext* context, absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes,
    absl::InlinedVector<std::pair<int64_t, int64_t>, 4>*
        resource_arg_unique_ids) {
  auto add_node = [](Node* node, absl::InlinedVector<OutputTensor, 4>* nodes) {
    auto* attr = node->attrs().Find("index");
    if (!attr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(),
                                     "' is missing attribute 'index'");

    auto index = attr->i();
    if (nodes->size() < index + 1) nodes->resize(index + 1);

    if ((*nodes)[index].node != nullptr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(), "' has attribute 'index' ",
                                     index, " that conflicts with node '",
                                     (*nodes)[index].node->name(), "'");
    (*nodes)[index] = {node, 0};

    return Status::OK();
  };

  // Collect arg and ret nodes from graph.
  for (auto* node : GetOrderedNodes())
    if (node->IsArg())
      TF_RETURN_IF_ERROR(add_node(node, arg_nodes));
    else if (node->IsRetval())
      TF_RETURN_IF_ERROR(add_node(node, ret_nodes));

  // Collect arg and ret types and create function type.
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  arg_types.reserve(arg_nodes->size());
  for (auto arg_node_and_idx : llvm::enumerate(*arg_nodes)) {
    auto& arg_node = arg_node_and_idx.value();
    if (arg_node.node == nullptr)
      return errors::InvalidArgument("Graph missing _Arg at index ",
                                     arg_node_and_idx.index());

    TF_ASSIGN_OR_RETURN(auto type,
                        InferOutputType(*arg_node.node, /*idx=*/0, builder));
    arg_types.push_back(type);
    tensorflow::int64 resource_arg_unique_id;
    if (TryGetNodeAttr(arg_node.node->attrs(), "_resource_arg_unique_id",
                       &resource_arg_unique_id)) {
      resource_arg_unique_ids->emplace_back(arg_node_and_idx.index(),
                                            resource_arg_unique_id);
    }
  }

  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(ret_nodes->size());
  for (auto ret_node_and_idx : llvm::enumerate(*ret_nodes)) {
    auto& ret_node = ret_node_and_idx.value();
    if (ret_node.node == nullptr)
      return errors::InvalidArgument("Graph missing _Retval at index ",
                                     ret_node_and_idx.index());

    TF_ASSIGN_OR_RETURN(auto type,
                        InferInputType(*ret_node.node, /*idx=*/0, builder));
    ret_types.push_back(type);
  }

  return builder.getFunctionType(arg_types, ret_types);
}

Status GraphDefImporter::GetControlRetsFromGraph(
    llvm::ArrayRef<std::string> control_outputs,
    absl::InlinedVector<Node*, 4>* control_ret_nodes) {
  if (control_outputs.empty()) return Status::OK();

  llvm::SmallDenseMap<llvm::StringRef, int32_t> controls_to_idx;
  for (auto control_and_idx : llvm::enumerate(control_outputs))
    controls_to_idx.insert({control_and_idx.value(), control_and_idx.index()});

  if (controls_to_idx.size() != control_outputs.size())
    return errors::InvalidArgument("Control outputs must be unique");

  control_ret_nodes->resize(controls_to_idx.size());

  for (auto* node : GetOrderedNodes()) {
    auto it = controls_to_idx.find(node->name());
    if (it != controls_to_idx.end()) (*control_ret_nodes)[it->second] = node;
  }

  for (auto node_and_name : llvm::zip(*control_ret_nodes, control_outputs))
    if (std::get<0>(node_and_name) == nullptr)
      return errors::InvalidArgument(
          "Control output '", std::get<1>(node_and_name), "' is missing");

  return Status::OK();
}

Status UpgradeLegacyGraph(Graph* graph, FunctionLibraryDefinition* flib_def) {
  return FunctionalizeControlFlow(graph, flib_def);
}

StatusOr<mlir::OwningModuleRef> ConvertGraphdefToMlir(
    const GraphDef& graphdef, const GraphDebugInfo& debug_info,
    const GraphImportConfig& specs, mlir::MLIRContext* context,
    bool add_default_attributes) {
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = add_default_attributes;
  Graph graph(OpRegistry::Global());

  GraphDef preprocessed_graphdef(graphdef);
  if (add_default_attributes) {
    TF_RETURN_IF_ERROR(PreprocessGraphDef(&specs, &preprocessed_graphdef));
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));
  return ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                            context);
}

StatusOr<mlir::OwningModuleRef> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context) {
  // TODO(jpienaar): Remove need to const_cast.
  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(
        UpgradeLegacyGraph(const_cast<Graph*>(&graph),
                           const_cast<FunctionLibraryDefinition*>(&flib_def)));
  }
  return GraphDefImporter::Convert(context, graph, debug_info, flib_def, specs,
                                   /*func_name=*/"main");
}

}  // namespace tensorflow
