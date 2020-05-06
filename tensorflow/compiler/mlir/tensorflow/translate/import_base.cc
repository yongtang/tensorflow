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
//#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
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

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

namespace tensorflow {
using mlir::TensorType;
using mlir::TF::VarHandleOp;
using mlir::tf_saved_model::GlobalTensorOp;
using stream_executor::port::StatusOr;

namespace {

bool IsDisableCallShapeInferenceAttribute(const AttrValue& attr_value,
                                          llvm::StringRef attr_name) {
  return attr_name.compare("_disable_call_shape_inference") == 0 &&
         attr_value.value_case() == AttrValue::kB;
}

bool IsOutputShapesAttribute(const AttrValue& attr_value,
                             llvm::StringRef attr_name) {
  return attr_name.compare("_output_shapes") == 0 &&
         attr_value.value_case() == AttrValue::kList;
}

// Returns true if the node with given name has a non primary output that is
// used by some other node as an input. Returns false if no outputs are in use
// or only the first output is in use.
bool HasNonPrimaryOutputInUse(const GraphDef& graph_def,
                              const std::string& node) {
  for (const auto& node_def : graph_def.node()) {
    for (const auto& input : node_def.input()) {
      if (absl::StartsWith(input, node + ":") && input != node + ":0") {
        return true;
      }
    }
  }
  return false;
}

// Updates the given LegacyFedInput node with Placeholder node if it is one of
// the inputs. Returns an error if non primary output of the LegacyFedInput node
// is in use and therefore can not be replaced by the Placeholder node that only
// has a single output.
Status UpdateLegacyFedInputNode(const GraphDef& graph_def,
                                const GraphImportConfig::InputArrays& inputs,
                                NodeDef* node) {
  const std::string& node_name = node->name();
  auto it = inputs.find(node_name);

  // Node is not an input.
  if (it == inputs.end()) return Status::OK();

  if (HasNonPrimaryOutputInUse(graph_def, node_name)) {
    return errors::InvalidArgument(
        "LegacyFedInput node ", node->name(),
        " has non primary output in use and can not be replaced with "
        "Placeholder node");
  }

  DataType dtype = it->second.imported_dtype;
  // Uses the existing output type if it isn't specified by the user.
  if (dtype == DT_INVALID) {
    dtype = node->attr().at("output_types").list().type(0);
  }
  // Update op name, drop inputs and set attributes required by the Placeholder
  // op.
  *node->mutable_op() = "Placeholder";
  node->clear_attr();
  node->clear_input();
  AddNodeAttr("dtype", dtype, node);
  AddNodeAttr("shape", it->second.shape, node);
  return Status::OK();
}

// Mapping from node name to feed (index and ArrayInfo). Node name must outlive
// this map.
using FeedsByNode = absl::flat_hash_map<
    absl::string_view,
    absl::flat_hash_map<int, const std::pair<std::string, ArrayInfo>*>>;

// Creates from a `GraphImportConfig::InputArrays` a mapping from a feeds output
// tensor name to index and ArrayInfo. Keys and values are backed by
// `GraphImportConfig::InputArrays`.
StatusOr<FeedsByNode> GetFeedsByNode(
    const GraphImportConfig::InputArrays& inputs) {
  FeedsByNode feeds_by_node;
  feeds_by_node.reserve(inputs.size());

  for (const auto& input : inputs) {
    TensorId tensor = ParseTensorName(input.first);
    if (tensor.index() < 0)
      return errors::FailedPrecondition(
          "Feed output tensor must be a data output '", tensor.ToString(), "'");

    auto& node = feeds_by_node[tensor.node()];
    if (!node.insert({tensor.index(), &input}).second)
      return errors::FailedPrecondition(
          "Multiple feeds for the same output tensor '", tensor.ToString(),
          "'");
  }

  return feeds_by_node;
}

// Creates a unique name for a node that will be replacing a feed output tensor.
std::string GetUniqueNodeName(
    absl::string_view node_name, int index,
    const std::unordered_map<string, Node*>& node_name_map) {
  std::string new_node_name_base = absl::StrCat(node_name, "_", index);
  int count = 0;
  std::string new_node_name = new_node_name_base;
  while (node_name_map.find(new_node_name) != node_name_map.end()) {
    new_node_name = absl::StrCat(new_node_name_base, "_", count++);
  }
  return new_node_name;
}

}  // namespace


// Preprocesses GraphDef before it can be converted to Graph by,
// - Adding the default attributes to each node def if they are missing from
//   the GraphDef.
// - Replacing LegacyFedInput nodes with Placeholder nodes if
//   convert_legacy_fed_inputs option is enabled.
Status PreprocessGraphDef(const GraphImportConfig* specs, GraphDef* graph_def) {
  for (auto& node_def : *graph_def->mutable_node()) {
    // TODO(hinsu): Completely deprecate support for LegacyFedInput ops. One
    // solution could be have a tool to let users upgrade old serialized graphs.
    if (specs && specs->convert_legacy_fed_inputs &&
        node_def.op() == "LegacyFedInput") {
      TF_RETURN_IF_ERROR(
          UpdateLegacyFedInputNode(*graph_def, specs->inputs, &node_def));
    }

    const tensorflow::OpRegistrationData* op_reg_data =
        tensorflow::OpRegistry::Global()->LookUp(node_def.op());
    if (!op_reg_data) {
      // This is likely a function call node, so we should continue.
      continue;
    }
    ::tensorflow::AddDefaultsToNodeDef(op_reg_data->op_def, &node_def);
  }
  return Status::OK();
}

Status ImporterBase::RemoveBackedges(const Graph& graph) {
  // TODO(fengliuai): Converting to GraphDef and back is the easiest way to
  // clone a graph.
  // TODO(fengliuai): clone the graph without going to graph_def first.
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  graph_ = absl::make_unique<Graph>(graph.flib_def());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.add_default_attributes = false;
  TF_RETURN_IF_ERROR(::tensorflow::ConvertGraphDefToGraph(
      opts, std::move(graph_def), graph_.get()));

  // Remove all the backedges. So the nodes can be added to the shape refiner.
  TF_RETURN_IF_ERROR(back_edge_helper_.Remove(graph_.get()));
  VLOG(1) << "Found " << (back_edge_helper_.RemovedEdges().size())
          << " backedges.";

  // Creates a map for quickly identifying whether a node output is a backedge.
  for (const auto& edge : back_edge_helper_.RemovedEdges()) {
    if (back_edge_node_output_.find(edge.src) != back_edge_node_output_.end() &&
        back_edge_node_output_[edge.src] != edge.src_output) {
      return errors::FailedPrecondition(
          "More than one of the src node outputs are backedges!");
    }
    back_edge_node_output_[edge.src] = edge.src_output;
    // We expect a merge to receive a single backedge (multiple NextIteration
    // nodes feeding into the same merge is unexpected here).
    DCHECK(!back_edge_dst_inputs_.contains(edge.dst));
    back_edge_dst_inputs_[edge.dst] = edge;
  }

  // Obtains a RPO ordering, using node names as a tiebreak for stable sorting.
  GetReversePostOrder(
      *graph_, &ordered_nodes_,
      [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });

  return Status::OK();
}

StatusOr<std::pair<Node*, bool>> ImporterBase::CreatePlaceholderNodeForFeed(
    const TensorShapeProto& shape, DataType dtype, Node* node, int index,
    const std::unordered_map<string, Node*>& node_name_map) {
  DCHECK_LT(index, node->num_outputs());
  const bool update_inplace = node->num_outputs() == 1 && index == 0;
  std::string new_node_name =
      update_inplace ? node->name()
                     : GetUniqueNodeName(node->name(), index, node_name_map);

  Node* placeholder_node;
  NodeBuilder builder(new_node_name, "Placeholder");
  builder.Attr("shape", shape);
  builder.Attr("dtype", dtype);
  TF_RETURN_IF_ERROR(builder.Finalize(graph_.get(), &placeholder_node));

  // Update edges from original feed with Placeholder node.
  std::vector<const Edge*> data_edges;
  std::vector<const Edge*> control_edges;
  for (const tensorflow::Edge* edge : node->out_edges()) {
    if (edge->src_output() == index) {
      data_edges.push_back(edge);
    } else if (update_inplace && edge->IsControlEdge()) {
      control_edges.push_back(edge);
    }
  }

  for (const auto* edge : data_edges) {
    TF_RETURN_IF_ERROR(graph_->UpdateEdge(placeholder_node, 0, edge->dst(),
                                          edge->dst_input()));
  }

  // TODO(lyandy): Preserve control dependencies properly by not forwarding
  // control dependencies to data outputs and not removing single output nodes.
  // When a data output is replaced as a feed, unless there is another non feed
  // data output or an explicit control output used by the same node, transitive
  // control dependencies are not to be executed. For single output nodes,
  // Placeholders can be converted to a NoOp if there are no uses, and
  // PlaceholderWithDefault can be converted to an Identity.
  for (const auto* edge : control_edges) {
    graph_->AddControlEdge(placeholder_node, edge->dst());
    graph_->RemoveControlEdge(edge);
  }

  if (update_inplace) {
    graph_->RemoveNode(node);
  }

  return std::pair<Node*, bool>(placeholder_node, update_inplace);
}

Status ImporterBase::GetInputOutputNodes(
    const std::unordered_map<string, Node*>& node_name_map,
    std::unordered_set<const Node*>* nodes) {
  auto add_node = [&](absl::string_view name) {
    auto it = node_name_map.find(std::string(name));
    if (it == node_name_map.end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", name));
    }
    nodes->insert(it->second);
    return Status::OK();
  };

  // Remap feeds and fetches to newly created Placeholder nodes.
  for (const auto& input : specs_.inputs) {
    TensorId tensor = ParseTensorName(input.first);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      TF_RETURN_IF_ERROR(add_node(remapped_it->second));
    } else {
      TF_RETURN_IF_ERROR(add_node(tensor.node()));
    }
  }

  for (const auto& output : specs_.outputs) {
    TensorId tensor = ParseTensorName(output);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      TF_RETURN_IF_ERROR(add_node(remapped_it->second));
    } else {
      TF_RETURN_IF_ERROR(add_node(tensor.node()));
    }
  }

  for (const auto& control_output : specs_.control_outputs)
    TF_RETURN_IF_ERROR(add_node(control_output));

  return Status::OK();
}

// TODO(jpienaar): Remove this post shape inference on import flag is removed.
Status ImporterBase::AddNodesToShapeRefiner(
    std::unordered_map<string, Node*>* node_name_map) {
  shape_refiner_ = absl::make_unique<ShapeRefiner>(graph_->versions(),
                                                   graph_->op_registry());
  // Some operations (for example "TPUExecute") don't have shape inference
  // function defined, so we should set this to false for adding nodes with
  // these types of operations.
  shape_refiner_->set_require_shape_inference_fns(false);
  shape_refiner_->set_function_library_for_shape_inference(&graph_flib_);

  TF_ASSIGN_OR_RETURN(auto feeds_by_node, GetFeedsByNode(specs_.inputs));

  // First add all nodes to the refiner.
  for (Node* node : ordered_nodes_) {
    // We need to use a TensorFlow node to teach the shape refiner that user
    // specifies certain data type and shape for the inputs in the `specs_`.
    // This node shouldn't have any inputs, only have one output and its
    // output type/shape is only determined by its "named" attributes. (The
    // attributes should have fixed names so we can use the info from `specs_`
    // to set the value of them.) `Placeholder` satisfies these constraints.
    //
    // Therefore, if the input node isn't a `Placeholder`, we create one and use
    // it to replace the original input node, so the shape refiner can
    // successfully propagate the user's input type and shape to the rest of the
    // graph.
    bool node_added_to_shape_refiner = false;
    auto it = feeds_by_node.find(node->name());
    if (it != feeds_by_node.end()) {
      auto op_name = node->op_def().name();
      if (op_name != "Placeholder" && op_name != "LegacyFedInput" &&
          op_name != FunctionLibraryDefinition::kArgOp) {
        for (const auto& output_tensor : it->second) {
          const int index = output_tensor.first;
          const ArrayInfo& array_info = output_tensor.second->second;

          DataType dtype = array_info.imported_dtype;
          // Uses the existing output type if it isn't specified by the user.
          if (dtype == DT_INVALID) {
            dtype = node->output_type(index);
          }

          TF_ASSIGN_OR_RETURN(
              auto placeholder_node_and_removed,
              CreatePlaceholderNodeForFeed(array_info.shape, dtype, node, index,
                                           *node_name_map));

          Node* placeholder_node = placeholder_node_and_removed.first;
          if (placeholder_node_and_removed.second) {
            // Original node has been removed from the graph.
            node = placeholder_node;
            node_added_to_shape_refiner = true;
          }
          remapped_feeds_[{it->first, index}] = placeholder_node->name();
          (*node_name_map)[placeholder_node->name()] = placeholder_node;
          // Add the new placeholder node to the shape refiner.
          Status status = shape_refiner_->AddNode(placeholder_node);
          if (!status.ok()) {
            return EmitErrorWithLocationStr(*placeholder_node, status);
          }
        }
      } else {
        auto index_it = it->second.find(0);
        if (index_it == it->second.end()) {
          return errors::FailedPrecondition(
              "Missing feed output tensor at index 0 for node '", node->name(),
              "'");
        }
        node->AddAttr("shape", index_it->second->second.shape);
        DataType dtype = index_it->second->second.imported_dtype;
        // Uses the existing output type if it isn't specified by the user.
        if (dtype == DT_INVALID) {
          dtype = node->output_type(0);
        }
        node->AddAttr("dtype", dtype);
      }
    }
    if (!node_added_to_shape_refiner) {
      // Add the node to the shape refiner if the node hasn't been removed.
      Status status = shape_refiner_->AddNode(node);
      if (!status.ok()) {
        return EmitErrorWithLocationStr(*node, status);
      }
    }

    auto set_shape_from_list_attr = [&](const AttrValue* attr) {
      auto& list = attr->list();
      for (auto shape : llvm::enumerate(list.shape())) {
        auto* node_context = shape_refiner_->GetContext(node);
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(shape.value(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(shape.index(), handle);
      }
      return Status::OK();
    };

    // We currently have no other way to get shapes from ReadVariableOp's.
    // Some graphs seem to have _output_shapes attributes on them, so use that
    // if possible.
    // TODO(silvasean): Ideally, we would do this in a separate shape inference
    // pass to avoid adding complexity to the importer. But right now, we don't
    // have an MLIR-native shape inference pass, so we need to do this while we
    // still have the Graph around, i.e. here, in the importer.
    if (node->op_def().name() == "ReadVariableOp") {
      // TODO(silvasean): In some graphs, this seems to be annotated on every
      // node. Why and by whom?
      // TODO(b/140588338): We should ideally incorporate that information for
      // all nodes, but right now, this can result in e.g. an Identity node with
      // signature such as
      // `(tensor<?x?xf32>) -> tensor<?x9216xf32>` which fails the verifier
      // (which checks for exact type equality; _output_shapes results in
      // us shoehorning in the more-precise type on the output).
      if (const AttrValue* attr = node->attrs().Find("_output_shapes"))
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
    }

    // If it is the argument node, the shape handle is set explicitly, so it
    // can be propagated to the body nodes of the function.
    if (StringPiece(node->type_string()) == FunctionLibraryDefinition::kArgOp) {
      auto* node_context = shape_refiner_->GetContext(node);
      DCHECK(node_context != nullptr);
      if (const AttrValue* attr = node->attrs().Find("shape")) {
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(attr->shape(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(0, handle);
      } else if (const AttrValue* attr = node->attrs().Find("_output_shapes")) {
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
      } else {
        node_context->set_output(0, node_context->UnknownShape());
      }
    }
  }

  // Since we might have inserted and removed nodes from the graph, fix
  // source/sink edges and reconstruct the RPO ordering of nodes
  FixupSourceAndSinkEdges(graph_.get());

  // Prune nodes in the graph that are not reachable from the output.
  if (specs_.prune_unused_nodes) {
    std::unordered_set<const Node*> prune_start;
    TF_RETURN_IF_ERROR(GetInputOutputNodes(*node_name_map, &prune_start));
    if (!prune_start.empty()) {
      if (PruneForReverseReachability(graph_.get(), prune_start)) {
        VLOG(1) << "Pruned unused nodes in graphdef";
      } else {
        VLOG(1) << "No unused nodes in graphdef to prune";
      }
    } else {
      VLOG(1) << "No output nodes specified, skipping pruning";
    }
  } else {
    VLOG(1) << "Pruning unused nodes in graphdef is disabled";
  }

  // Re-initialize ordered_nodes_ since we might have modified the graph.
  GetReversePostOrder(
      *graph_, &ordered_nodes_,
      [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });

  VLOG(1) << "Inferring graph shapes to fixpoint";

  // The "changed" information from UpdateNode can give false positives, so we
  // create a dedicated method to verify the shapes are not changed before and
  // after the shape refine.
  auto same_inferred_shape = [](shape_inference::InferenceContext* c,
                                shape_inference::ShapeHandle s0,
                                shape_inference::ShapeHandle s1) -> bool {
    if (s0.SameHandle(s1) || (!c->RankKnown(s0) && !c->RankKnown(s1))) {
      return true;
    }
    if (c->Rank(s0) != c->Rank(s1)) {
      return false;
    }
    for (int i = 0; i < c->Rank(s0); ++i) {
      if (!c->Dim(s0, i).SameHandle(c->Dim(s1, i))) {
        int64 val0 = c->Value(c->Dim(s0, i));
        int64 val1 = c->Value(c->Dim(s1, i));
        // Negative value is treated as unknown so all negative values indicate
        // the same dimension.
        if (val0 >= 0 && val1 >= 0 && val0 != val1) return false;
      }
    }
    return true;
  };

  bool changed = true;
  int i = 0;
  const int kMaxIterationCount = 2;
  while (changed && i != kMaxIterationCount) {
    changed = false;
    for (const Node* node : ordered_nodes_) {
      auto* shape_context = shape_refiner_->GetContext(node);
      DCHECK(shape_context != nullptr);
      absl::InlinedVector<shape_inference::ShapeHandle, 4> existing;
      existing.reserve(shape_context->num_outputs());
      for (int o = 0; o < shape_context->num_outputs(); ++o) {
        existing.push_back(shape_context->output(o));
      }
      bool inferred = false;
      shape_inference::ShapeHandle handle;
      Status status =
          shape_refiner_->UpdateNode(node, /*relax=*/false, &inferred);
      if (!status.ok()) {
        return EmitErrorWithLocationStr(*node, status);
      }
      for (int o = 0; o < shape_context->num_outputs(); ++o) {
        if (!same_inferred_shape(shape_context, shape_context->output(o),
                                 existing[o])) {
          changed = true;
          break;
        }
      }
    }
    ++i;
  }
  if (i >= kMaxIterationCount) {
    LOG(WARNING) << "Graph shapes did not converge to a fixpoint within "
                 << kMaxIterationCount
                 << " iterations. Graph shapes may be conservative.";
  }
  VLOG(1) << "Graph shapes were inferred with " << (i - 1)
          << " extra rounds of analysis to reach a fixpoint.";
  return Status::OK();
}

StatusOr<mlir::Type> ImporterBase::InferInputType(const Node& node, int idx,
                                                  mlir::Builder builder) {
  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove this if shape inference on import flag is removed.
    ExtendedInferenceContext* shape_context =
        shape_refiner_->GetExtendedContext(&node);
    DataType dtype = shape_context->input_type(idx);
    auto* context = shape_context->get_context();
    return ConvertDataTypeAndShape(dtype, context->input(idx),
                                   context->input_handle_shapes_and_types(idx),
                                   context, builder);
  }
  DataType dtype = node.properties()->input_types[idx];
  mlir::Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
  return mlir::UnrankedTensorType::get(element_type);
}

StatusOr<mlir::Type> ImporterBase::InferOutputType(const Node& node, int idx,
                                                   mlir::Builder builder) {
  DataType dtype = node.properties()->output_types[idx];

  // Returns output type given inference context.
  auto shape_ic = [&](shape_inference::InferenceContext* c) {
    return ConvertDataTypeAndShape(dtype, c->output(idx),
                                   c->output_handle_shapes_and_types(idx), c,
                                   builder);
  };

  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove this if shape inference on import flag is removed.
    ExtendedInferenceContext* shape_context =
        shape_refiner_->GetExtendedContext(&node);
    return shape_ic(shape_context->get_context());
  }

  // Treat TensorList init ops specially here as the op requires knowing its
  // element dtype.
  // TODO(jpienaar): Reconsider post refactoring shape functions.
  if (node.type_string() == "TensorListReserve" ||
      node.type_string() == "EmptyTensorList") {
    mlir::Type etype;
    if (auto element_dtype = node.attrs().Find("element_dtype")) {
      TF_RETURN_IF_ERROR(
          ConvertDataType(element_dtype->type(), builder, &etype));
    }
    return mlir::RankedTensorType::get(
        {}, mlir::TF::VariantType::get({mlir::UnrankedTensorType::get(etype)},
                                       etype.getContext()));
  }

  // Returns a simple, more conservative unranked tensor type.
  auto default_type = [&]() -> StatusOr<mlir::Type> {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
    return mlir::UnrankedTensorType::get(element_type);
  };

  // Below we only try and do some shape inference for "source" ops which have
  // no inputs.
  if (node.num_inputs() > 0) return default_type();

  // Do some simply inference here to get the function arguments correct for
  // this common case.
  // TODO(jpienaar): Reconsider post refactoring shape functions.
  if (node.IsArg()) {
    if (dtype == DT_RESOURCE) {
      const AttrValue* dtype_attr = node.attrs().Find("_handle_dtypes");
      const AttrValue* shape_attr = node.attrs().Find("_handle_shapes");
      LOG(INFO) << dtype_attr << " " << shape_attr;
      if (dtype_attr && shape_attr) {
        if (dtype_attr->list().type().empty()) {
          return errors::InvalidArgument(
              "Invalid \"_handle_dtypes\" attribute value for _Arg node: ",
              shape_attr->DebugString());
        }
        if (shape_attr->list().shape().empty()) {
          return errors::InvalidArgument(
              "Invalid \"_handle_shapes\" attribute value for _Arg node: ",
              shape_attr->DebugString());
        }
        DataType dtype = dtype_attr->list().type(0);
        const TensorShapeProto& shape_proto = shape_attr->list().shape(0);
        TF_ASSIGN_OR_RETURN(
            auto etype, ConvertToMlirTensorType(shape_proto, dtype, &builder));
        return mlir::UnrankedTensorType::get(mlir::TF::ResourceType::get(
            {etype.cast<TensorType>()}, builder.getContext()));
      } else {
        return mlir::UnrankedTensorType::get(
            mlir::TF::ResourceType::get(builder.getContext()));
      }
    } else if (auto shape = node.attrs().Find("_output_shapes")) {
      if (shape->has_list() && shape->list().shape_size() == 1) {
        return ConvertToMlirTensorType(shape->list().shape().at(0), dtype,
                                       &builder);
      }
    }
  }

  const tensorflow::OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(
      graph_->op_registry()->LookUp(node.type_string(), &op_reg_data));
  if (!op_reg_data) {
    DVLOG(1) << "Skipping inference for unregistered op " << node.type_string();
    return default_type();
  }
  if (op_reg_data->shape_inference_fn == nullptr) {
    DVLOG(1) << "Skipping inference for op without shape function "
             << node.type_string();
    return default_type();
  }
  shape_inference::InferenceContext c(graph_->versions().producer(),
                                      node.attrs(), op_reg_data->op_def,
                                      std::vector<PartialTensorShape>{}, {},
                                      /*input_tensors_as_shapes=*/{}, {});
  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));
  return shape_ic(&c);
}

StatusOr<TensorType> ImporterBase::ConvertDataTypeAndShape(
    DataType dtype, const shape_inference::ShapeHandle& handle,
    const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  TF_ASSIGN_OR_RETURN(auto subtypes,
                      ConvertSubtypes(handle_subtypes, context, builder));

  mlir::Type element_type;
  if (dtype == DT_VARIANT)
    element_type = mlir::TF::VariantType::get(subtypes, context_);
  else if (dtype == DT_RESOURCE)
    element_type = mlir::TF::ResourceType::get(subtypes, context_);
  else
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(dtype, builder, &element_type));

  return ConvertElementTypeAndShape(element_type, handle, context, builder);
}

StatusOr<TensorType> ImporterBase::ConvertElementTypeAndShape(
    mlir::Type element_type, const shape_inference::ShapeHandle& handle,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  if (!context->RankKnown(handle)) {
    return mlir::UnrankedTensorType::get(element_type);
  }

  // Sentinel for an unknown dimension size. getTensorType interprets any
  // negative value as an unknown dimension.
  // TODO(jmolloy): Ideally this shouldn't be a local sentinel.
  const int64_t kUnknownDim = -1;

  absl::InlinedVector<int64_t, 4> dimensions;
  int32 rank = context->Rank(handle);
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto dim_handle = context->Dim(handle, i);
    if (!context->ValueKnown(dim_handle))
      dimensions.push_back(kUnknownDim);
    else
      dimensions.push_back(context->Value(dim_handle));
  }

  return mlir::RankedTensorType::get(
      llvm::makeArrayRef(dimensions.begin(), dimensions.end()), element_type);
}

StatusOr<ImporterBase::ElementSubtypes> ImporterBase::ConvertSubtypes(
    const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  ElementSubtypes subtypes;
  if (!handle_subtypes) return subtypes;

  subtypes.reserve(handle_subtypes->size());
  for (const auto& subtype : *handle_subtypes) {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(subtype.dtype, builder, &element_type));
    TF_ASSIGN_OR_RETURN(TensorType type,
                        ConvertElementTypeAndShape(element_type, subtype.shape,
                                                   context, builder));
    subtypes.push_back(type);
  }
  return subtypes;
}

Status ImporterBase::ConvertFunctionCallAttribute(
    const std::string& base_name, const AttrValue& value,
    llvm::SmallVector<mlir::NamedAttribute, 4>* attributes) {
  TF_ASSIGN_OR_RETURN(auto func_attr,
                      ConvertFunctionCallName(value.func().name()));
  attributes->push_back(builder_.getNamedAttr(base_name, func_attr));

  for (const auto& it : value.func().attr()) {
    auto name = absl::StrCat(base_name, ".", it.first);
    TF_ASSIGN_OR_RETURN(auto value, ConvertAttributeValue(it.second));
    attributes->push_back(builder_.getNamedAttr(name, value));
  }
  return Status::OK();
}

StatusOr<mlir::FlatSymbolRefAttr> ImporterBase::ConvertFunctionCallName(
    const std::string& func_name) {
  TF_RETURN_IF_ERROR(ConvertLibFunction(func_name));
  auto mlir_func_name = (*tf_name_to_mlir_name_)[func_name];
  auto func = module_.lookupSymbol<mlir::FuncOp>(mlir_func_name);
  return builder_.getSymbolRefAttr(func);
}

StatusOr<mlir::Attribute> ImporterBase::ConvertAttributeValue(
    const AttrValue& value) {
  switch (value.value_case()) {
    case AttrValue::kI:
      return builder_.getI64IntegerAttr(value.i());
    case AttrValue::kS:
      return builder_.getStringAttr(value.s());
    case AttrValue::kF:
      return builder_.getFloatAttr(builder_.getF32Type(), value.f());
    case AttrValue::kB:
      return builder_.getBoolAttr(value.b());
    case AttrValue::kType: {
      mlir::Type type;
      TF_RETURN_IF_ERROR(ConvertDataType(value.type(), builder_, &type));
      return mlir::TypeAttr::get(type);
    }
    case AttrValue::kShape:
      return ConvertTensorShapeProto(value.shape());
    case AttrValue::kTensor:
      return ConvertTensorProto(value.tensor());
    case AttrValue::kList: {
      absl::InlinedVector<mlir::Attribute, 8> attrs;
      for (const auto& item : value.list().i())
        attrs.push_back(builder_.getI64IntegerAttr(item));
      for (const auto& item : value.list().s())
        attrs.push_back(builder_.getStringAttr(item));
      for (const auto& item : value.list().f())
        attrs.push_back(builder_.getFloatAttr(builder_.getF32Type(), item));
      for (const auto& item : value.list().b())
        attrs.push_back(builder_.getBoolAttr(item));
      for (const auto& item : value.list().type()) {
        mlir::Type type;
        TF_RETURN_IF_ERROR(ConvertDataType(DataType(item), builder_, &type));
        attrs.push_back(mlir::TypeAttr::get(type));
      }
      for (const auto& item : value.list().shape()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertTensorShapeProto(item));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().tensor()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertTensorProto(item));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().func()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertFunctionCallName(item.name()));
        if (item.attr_size() != 0)
          return errors::Unimplemented(
              "func attributes with non-zero attr.size()");
        attrs.push_back(attr);
      }
      return builder_.getArrayAttr(
          llvm::makeArrayRef(attrs.begin(), attrs.end()));
    }
    case AttrValue::kFunc:
      return errors::Unknown("kFunc type should be handled separately!");
    case AttrValue::VALUE_NOT_SET:
      return builder_.getUnitAttr();
    // kPlaceholder is not implemented.
    default:
      return errors::Unimplemented(
          absl::StrCat("Attribute ", value.DebugString()));
  }
}

void ImporterBase::GetArgsAndRetsFromFunctionBody(
    const FunctionBody& fbody, absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes,
    absl::InlinedVector<Node*, 4>* control_ret_nodes,
    absl::InlinedVector<std::pair<int64_t, int64_t>, 4>*
        resource_arg_unique_ids) {
  arg_nodes->reserve(fbody.arg_nodes.size());
  ret_nodes->reserve(fbody.ret_nodes.size());
  for (auto arg : fbody.arg_nodes) {
    arg_nodes->emplace_back(arg, 0);
  }
  for (auto ret : fbody.ret_nodes) {
    ret_nodes->emplace_back(ret, 0);
  }
  for (const auto& entry : fbody.fdef.resource_arg_unique_id()) {
    resource_arg_unique_ids->push_back(entry);
  }
  *control_ret_nodes = fbody.control_ret_nodes;
}

Status ImporterBase::ConvertLibFunction(llvm::StringRef func_name) {
  // If the library function has been converted already, nothing needs to be
  // done.
  if (tf_name_to_mlir_name_->find(std::string(func_name)) !=
      tf_name_to_mlir_name_->end())
    return Status::OK();

  std::string mlir_func_name(
      function_name_uniquifier_->GetUniqueName(func_name));
  (*tf_name_to_mlir_name_)[std::string(func_name)] = mlir_func_name;

  const auto& func_lib = graph_flib_;
  const auto* func_def = func_lib.Find(std::string(func_name));
  if (func_def == nullptr) {
    return errors::FailedPrecondition(
        absl::StrCat("Failed to find function '", StringRefToView(func_name),
                     "'. The imported TensorFlow GraphDef is ill-formed."));
  }

  // Converts the function definition to a graph.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*func_def, AttrSlice(), &func_lib, &fbody));

  // Converts the argument and return types to MLIR types.
  absl::InlinedVector<mlir::NamedAttribute, 8> attributes;
  attributes.reserve(func_def->attr_size());
  for (const auto& name_and_value : func_def->attr()) {
    // This is a function definition attribute, so it shouldn't contain
    // kFunc attribute and it is treated as normal one.
    TF_ASSIGN_OR_RETURN(auto attr,
                        ConvertAttributeValue(name_and_value.second));
    std::string attr_name =
        mangling_util::MangleAttributeName(name_and_value.first);
    attributes.push_back(builder_.getNamedAttr(attr_name, attr));
  }

  // Checks opdef stateful attribute and import that as Function Attribute
  if (func_def->signature().is_stateful()) {
    auto stateful_str = mlir::TF::TensorFlowDialect::GetStatefulAttrName();
    attributes.push_back(
        builder_.getNamedAttr(stateful_str, builder_.getUnitAttr()));
  }

  // Checks for an associated custom gradient function. Adds it to the attribute
  // list of this function.
  auto grad_func_name = func_lib.FindGradient(std::string(func_name));
  if (!grad_func_name.empty()) {
    TF_RETURN_IF_ERROR(ConvertLibFunction(grad_func_name));
    auto mlir_grad_func_name = (*tf_name_to_mlir_name_)[grad_func_name];
    auto grad_func = module_.lookupSymbol<mlir::FuncOp>(mlir_grad_func_name);
    auto gradient_attr = builder_.getSymbolRefAttr(grad_func);
    auto grad_string = mlir::TF::TensorFlowDialect::GetGradientAttrName();
    attributes.push_back(builder_.getNamedAttr(grad_string, gradient_attr));
  }

  // Converts the graph to an MLIR function and adds it to the module.
  // We populate the NodeSpec so that all the _Arg ops get their shape
  // added correctly.
  GraphImportConfig specs;
  specs.enable_shape_inference = specs_.enable_shape_inference;
  for (const auto& name_and_value : func_def->attr()) {
    if (name_and_value.first == "_input_shapes") {
      auto& list = name_and_value.second.list();
      auto& signature = func_def->signature();
      if (list.shape_size() != signature.input_arg_size()) {
        return errors::FailedPrecondition(
            "Number of input arguments must be equal to the length of "
            "_input_shapes attribute in function '",
            StringRefToView(func_name), "'.");
      }
      for (int i = 0; i < list.shape_size(); i++) {
        auto& input_arg = signature.input_arg(i);
        auto& array_info = specs.inputs[input_arg.name()];
        array_info.imported_dtype = input_arg.type();
        array_info.shape = list.shape(i);
      }
    }
  }

  ImporterBase child_importer(graph_flib_, debug_info_, specs, module_,
                              tf_name_to_mlir_name_, function_name_uniquifier_,
                              func_name);
  TF_RETURN_IF_ERROR(child_importer.PrepareConvert(*fbody->graph));

  TF_ASSIGN_OR_RETURN(auto func_type,
                      child_importer.InferLibFunctionType(*fbody));

  absl::InlinedVector<OutputTensor, 4> arg_nodes;
  absl::InlinedVector<OutputTensor, 4> ret_nodes;
  absl::InlinedVector<Node*, 4> control_ret_nodes;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 4> resource_arg_unique_ids;
  GetArgsAndRetsFromFunctionBody(*fbody, &arg_nodes, &ret_nodes,
                                 &control_ret_nodes, &resource_arg_unique_ids);

  TF_RETURN_IF_ERROR(child_importer.Convert(
      mlir_func_name, func_type, arg_nodes, ret_nodes, control_ret_nodes,
      llvm::makeArrayRef(attributes.begin(), attributes.end()),
      resource_arg_unique_ids));
  return Status::OK();
}

Status ImporterBase::PruneUnreachableNodes(
    std::unordered_map<string, Node*>* node_name_map) {
  std::unordered_set<const Node*> prune_start;
  TF_RETURN_IF_ERROR(GetInputOutputNodes(*node_name_map, &prune_start));

  if (!prune_start.empty()) {
    if (PruneForReverseReachability(graph_.get(), prune_start)) {
      VLOG(1) << "Pruned unused nodes in graphdef";
    } else {
      VLOG(1) << "No unused nodes in graphdef to prune";
    }
  } else {
    VLOG(1) << "No output nodes specified, skipping pruning";
  }
  return Status::OK();
}

Status ImporterBase::ConvertFeedsToPlaceholders(
    std::unordered_map<string, Node*>* node_name_map) {
  // Feeds (edges) are converted into single-output placeholder nodes to
  // simplify the conversion process.
  TF_ASSIGN_OR_RETURN(auto feeds_by_node, GetFeedsByNode(specs_.inputs));
  for (const auto& it : feeds_by_node) {
    TensorId tensor = ParseTensorName(it.first);
    auto jt = node_name_map->find(std::string(tensor.node()));
    if (jt == node_name_map->end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", tensor.node()));
    }

    Node* node = jt->second;
    auto op_name = node->op_def().name();
    if (op_name != "Placeholder" && op_name != "LegacyFedInput" &&
        op_name != FunctionLibraryDefinition::kArgOp) {
      for (const auto& output_tensor : it.second) {
        const int index = output_tensor.first;
        const ArrayInfo& array_info = output_tensor.second->second;

        DataType dtype = array_info.imported_dtype;
        // Uses the existing output type if it isn't specified by the user.
        if (dtype == DT_INVALID) {
          dtype = node->output_type(index);
        }

        TF_ASSIGN_OR_RETURN(
            auto placeholder_node_and_removed,
            CreatePlaceholderNodeForFeed(array_info.shape, dtype, node, index,
                                         *node_name_map));

        Node* placeholder_node = placeholder_node_and_removed.first;
        if (placeholder_node->in_edges().empty()) {
          graph_->AddControlEdge(graph_->source_node(), placeholder_node,
                                 true /* skip test for duplicates */);
        }
        if (placeholder_node->out_edges().empty()) {
          graph_->AddControlEdge(placeholder_node, graph_->sink_node(),
                                 true /* skip test for duplicates */);
        }
        remapped_feeds_[{it.first, index}] = placeholder_node->name();
        (*node_name_map)[placeholder_node->name()] = placeholder_node;
      }
    }
  }
  return Status::OK();
}

Status ImporterBase::PrepareConvert(const Graph& graph) {
  TF_RETURN_IF_ERROR(RemoveBackedges(graph));

  auto node_name_map = graph_->BuildNodeNameIndex();

  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove once infer shapes on import flag is removed.
    TF_RETURN_IF_ERROR(AddNodesToShapeRefiner(&node_name_map));
  } else {
    TF_RETURN_IF_ERROR(ConvertFeedsToPlaceholders(&node_name_map));
  }

  // Prune nodes in the graph that are not reachable from the output.
  if (specs_.prune_unused_nodes) {
    TF_RETURN_IF_ERROR(PruneUnreachableNodes(&node_name_map));
  }

  if (!specs_.enable_shape_inference) {
    // Re-initialize ordered_nodes_ since we might have modified the graph.
    GetReversePostOrder(
        *graph_, &ordered_nodes_,
        [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });
  }

  return Status::OK();
}

Status ImporterBase::Convert(
    llvm::StringRef func_name, mlir::FunctionType func_type,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes,
    llvm::ArrayRef<mlir::NamedAttribute> attrs,
    const absl::InlinedVector<std::pair<int64_t, int64_t>, 4>&
        resource_arg_unique_ids) {
  // TODO(b/122040776): Uses debug info for FunctionDef.
  auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(context_),
                                       func_name, func_type, attrs);

  module_.push_back(function);
  // Seeds the builder with an initial block.
  function.addEntryBlock();
  builder_ = mlir::OpBuilder(function.getBody());

  // Create the graph operation in which we will convert the individual nodes.
  auto graph = builder_.create<mlir::tf_executor::GraphOp>(
      function.getLoc(), func_type.getResults());
  builder_.createBlock(&graph.body());

  for (const Node* node : ordered_nodes_) {
    TF_RETURN_IF_ERROR(ConvertNode(*node));
  }

  // Adds the backedges back to the function by creating the source and sink
  // pairs.
  TF_RETURN_IF_ERROR(AddBackedges());

  TF_RETURN_IF_ERROR(ConvertFunctionArgAndRets(function, graph,
                                               func_type.getInputs(), arg_nodes,
                                               ret_nodes, control_ret_nodes));
  for (const auto& entry : resource_arg_unique_ids) {
    function.setArgAttr(entry.first, "tf.resource_arg_unique_id",
                        builder_.getI64IntegerAttr(entry.second));
  }

  // TODO(jpienaar): Update post removing shape_refinier_.
  if (!specs_.enable_shape_inference) {
    // Refine graph's type given more precise fetch.
    auto fetch = graph.GetFetch();
    bool all_equal = true;
    for (auto it :
         llvm::zip_first(graph.getResults(), fetch.getOperandTypes())) {
      auto rt = std::get<1>(it);
      if (rt == std::get<0>(it).getType()) continue;
      std::get<0>(it).setType(rt);
      all_equal = false;
    }
    if (!all_equal) {
      function.setType(mlir::FunctionType::get(func_type.getInputs(),
                                               graph.getResultTypes(),
                                               function.getContext()));
    }
  }

  return Status::OK();
}

Status ImporterBase::ConvertFunctionArgAndRets(
    mlir::FuncOp func, mlir::tf_executor::GraphOp graph_op,
    llvm::ArrayRef<mlir::Type> arg_types,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes) {
  auto* bb = &func.front();
  llvm::SmallDenseMap<std::pair<Node*, int>, mlir::Value, 4>
      arg_nodes_to_values;
  for (int i = 0, e = arg_types.size(); i < e; ++i) {
    auto& arg_node = arg_nodes[i];
    // The lookup can't fail here: otherwise some nodes in the function haven't
    // be converted to mlir operations and don't have a mapping.
    mlir::Operation* island = node_values_.find(arg_node.node->id())->second;

    auto bb_arg = bb->getArgument(i);
    mlir::Value arg_def = bb_arg;

    if (island->getNumResults() != 2)
      return errors::InvalidArgument(
          "Only feed output tensors of single output nodes are supported");

    // Collect mapping of OutputTensor to associated block arg.
    arg_nodes_to_values.try_emplace({arg_node.node, arg_node.index}, arg_def);
    island->getResult(0).replaceAllUsesWith(arg_def);
    // Erase control outputs from feed.
    auto control_uses = island->getResult(1).getUses();
    for (auto& control_use : llvm::make_early_inc_range(control_uses))
      control_use.getOwner()->eraseOperand(control_use.getOperandNumber());

    if (!arg_node.node->requested_device().empty())
      func.setArgAttr(
          i, "tf.device",
          builder_.getStringAttr(arg_node.node->requested_device()));

    island->dropAllReferences();
    island->erase();
  }

  llvm::SmallVector<mlir::Value, 8> inst_to_return;
  for (const auto& ret : ret_nodes) {
    auto* inst = node_values_[ret.node->id()];
    auto op = absl::string_view(ret.node->type_string());
    if (op == FunctionLibraryDefinition::kRetOp ||
        op == FunctionLibraryDefinition::kDeviceRetOp) {
      // Lookup the instruction inside the island
      auto island_op = llvm::cast<mlir::tf_executor::IslandOp>(inst);
      mlir::Operation* inner_op = &island_op.GetBody().front();
      // Remove kRetOp or kDeviceRetOp operation and return its operand.
      // kRetOp and kDeviceRetOp should have just one operand unless they have
      // control dependencies.
      if (inner_op->getNumOperands() != 1)
        return errors::Unimplemented("Return node with multiple inputs.");
      inst_to_return.push_back(inner_op->getOperand(0));
      inst->dropAllReferences();
      inst->erase();
    } else {
      // Lookup and use block arg if fetch is a feed.
      auto it = arg_nodes_to_values.find({ret.node, ret.index});
      if (it != arg_nodes_to_values.end())
        inst_to_return.push_back(it->second);
      else
        inst_to_return.push_back(inst->getResult(ret.index));
    }
  }

  for (Node* control_ret : control_ret_nodes) {
    auto* inst = node_values_[control_ret->id()];
    inst_to_return.push_back(*std::prev(inst->result_end()));
  }

  // Terminate the function by adding a Fetch operation to terminate the graph
  // and a return operation to return the Graph results.
  builder_.setInsertionPointToEnd(&graph_op.body().front());
  builder_.create<mlir::tf_executor::FetchOp>(graph_op.getLoc(),
                                              inst_to_return);
  builder_.setInsertionPointToEnd(bb);
  builder_.create<mlir::ReturnOp>(mlir::UnknownLoc::get(context_),
                                  graph_op.getResults());
  return Status::OK();
}

mlir::Location ImporterBase::GetLocation(const NodeDef& node_def) {
  // TODO(b/142400497): What is the semantic contract for locations?
  const auto& debug_info = debug_info_.traces();

  // Create a location for node `name` in function `function_name`.
  auto create_location = [&](llvm::StringRef name,
                             llvm::StringRef function_name) -> mlir::Location {
    // Use the catenation of function and node names as the lookup key into the
    // debug info. This matches the way that the key is formed on the python
    // side.
    //
    // We also use this as the name for the NameLoc for ops in function, since
    // otherwise our names could collide across functions.
    // For ops in the main graph, we omit the "@function_name" (which, would be
    // just "@" since function_name would be empty) because some code seems to
    // depend on the name being this way for correctness.
    std::string debug_info_key = (name + "@" + function_name).str();
    std::string name_for_name_loc =
        function_name.empty() ? name.str() : (name + "@" + function_name).str();
    auto name_loc_id = mlir::Identifier::get(name_for_name_loc, context_);
    const auto location_it = debug_info.find(debug_info_key);
    if (location_it == debug_info.end()) {
      return mlir::NameLoc::get(name_loc_id, context_);
    }

    // Convert the stack trace to a chain of mlir::CallSiteLocs.
    const auto& trace = location_it->second;
    llvm::SmallVector<mlir::Location, 4> locations;
    locations.reserve(trace.file_line_cols_size());
    for (const auto& location : trace.file_line_cols()) {
      const auto& file = debug_info_.files(location.file_index());
      auto file_name = mlir::Identifier::get(file, context_);
      auto file_line_loc = mlir::FileLineColLoc::get(file_name, location.line(),
                                                     location.col(), context_);
      locations.push_back(file_line_loc);
    }

    // If there are no locations in the stack trace, fall back to just a
    // NameLoc with no child.
    if (locations.empty()) return mlir::NameLoc::get(name_loc_id, context_);

    // Use the front FileLineColLoc to generate a NameLoc.
    mlir::Location node_name_loc =
        mlir::NameLoc::get(name_loc_id, locations.front());

    // If there are more locations then generate a stack trace, otherwise just
    // return the name loc.
    auto callsite_locs = llvm::makeArrayRef(locations).drop_front();
    return callsite_locs.empty()
               ? node_name_loc
               : mlir::CallSiteLoc::get(node_name_loc, callsite_locs);
  };

  // For NextIteration nodes, location is used to pair source and sink nodes.
  // Hence, we use node name as location to keep it unique.
  // TODO(prakalps): In future the plan is to use tokens to pair source/sink
  // nodes. Then NextIteration nodes would not need to be handled separately.
  if (node_def.op() == "NextIteration")
    return create_location(node_def.name(), function_name_for_debug_info_);

  auto original_nodes =
      node_def.experimental_debug_info().original_node_names();
  auto original_funcs =
      node_def.experimental_debug_info().original_func_names();

  if (original_nodes.empty()) {
    return create_location(node_def.name(), function_name_for_debug_info_);
  } else {
    // If the original nodes are defined, then we use them to get a list of
    // call sites, and then fuse them to a single fused location, with the name
    // of the node_def.
    llvm::SmallVector<mlir::Location, 4> node_locations;
    node_locations.reserve(original_nodes.size() + 1);

    // store the names in the experimental_debug_info
    for (int i = 0, e = original_nodes.size(); i != e; ++i) {
      auto node_name = original_nodes[i];
      auto func_name = (i < original_funcs.size()) ? original_funcs[i] : "";
      node_locations.push_back(create_location(node_name, func_name));
    }
    // store the name of the node_def
    node_locations.push_back(
        create_location(node_def.name(), function_name_for_debug_info_));
    return mlir::FusedLoc::get(node_locations, context_);
  }
}

Status ImporterBase::EmitErrorWithLocationStr(const Node& node,
                                              const Status& error_status) {
  const mlir::Location location = GetLocation(node.def());
  mlir::emitError(location);
  return error_handler_.Combine(error_status);
}

mlir::Operation* ImporterBase::CreateOperation(
    const Node& node, llvm::StringRef node_type_name,
    const mlir::OperationState& result,
    const llvm::SmallVectorImpl<mlir::Value>& control_operands,
    bool convert_to_legacy_call) {
  // For the tf.executor specific operations (not wrapped in an island), we
  // have an extra returned value for the control result, and we concatenate
  // control and non-control operands.
  mlir::SmallVector<mlir::Type, 4> types(result.types);
  types.push_back(mlir::tf_executor::ControlType::get(builder_.getContext()));
  mlir::SmallVector<mlir::Value, 4> operands(result.operands);
  operands.append(control_operands.begin(), control_operands.end());

  auto loc = result.location;
  // Dispatch based on the name and create the appropriate operation.
  if (node.IsSwitch()) {
    // Switch and _SwitchN both are in switch class, differentiate based on
    // op name.
    if (node.op_def().name() == "_SwitchN") {
      return builder_.create<mlir::tf_executor::SwitchNOp>(loc, types, operands,
                                                           result.attributes);
    }
    return builder_.create<mlir::tf_executor::SwitchOp>(loc, types, operands,
                                                        result.attributes);
  }
  if (node.IsMerge()) {
    return builder_.create<mlir::tf_executor::MergeOp>(loc, types, operands,
                                                       result.attributes);
  }
  if (node.IsNextIteration()) {
    // NextIteration is a bit special, we create a pair of operations that are
    // linked together through a token returned by the source.
    // We make use of a separate builder to insert the source at the top of
    // the block.
    mlir::OpBuilder builder_at_begin(builder_.getBlock(),
                                     builder_.getBlock()->begin());
    auto source_op =
        builder_at_begin.create<mlir::tf_executor::NextIterationSourceOp>(
            loc, operands[0].getType(), result.attributes);
    return builder_.create<mlir::tf_executor::NextIterationSinkOp>(
        loc, source_op.token(), operands, result.attributes);
  }
  if (node.IsLoopCond()) {
    return builder_.create<mlir::tf_executor::LoopCondOp>(loc, types, operands,
                                                          result.attributes);
  }
  if (node.IsEnter()) {
    return builder_.create<mlir::tf_executor::EnterOp>(loc, types, operands,
                                                       result.attributes);
  }
  if (node.IsExit()) {
    return builder_.create<mlir::tf_executor::ExitOp>(loc, types, operands,
                                                      result.attributes);
  }
  if (node.IsControlTrigger()) {
    return builder_.create<mlir::tf_executor::ControlTriggerOp>(
        loc, operands, result.attributes);
  }
  // Regular TensorFlow operation are wrapped in a tf_executor.island.
  auto island = builder_.create<mlir::tf_executor::IslandOp>(
      result.location, types, control_operands,
      mlir::ArrayRef<mlir::NamedAttribute>{});
  island.body().push_back(new mlir::Block);
  mlir::OpBuilder island_builder =
      mlir::OpBuilder::atBlockEnd(&island.GetBody());

  // Create the operation inside the island now.
  mlir::Operation* inner_op;
  if (convert_to_legacy_call) {
    bool disable_call_shape_inference = false;
    for (const auto& name_and_value : node.attrs()) {
      const auto& attr_name = name_and_value.first;
      const AttrValue& attr_value = name_and_value.second;
      if (IsDisableCallShapeInferenceAttribute(attr_value, attr_name)) {
        disable_call_shape_inference = attr_value.b();
      }
    }

    mlir::BoolAttr attribute =
        builder_.getBoolAttr(disable_call_shape_inference);
    inner_op = island_builder.create<mlir::TF::LegacyCallOp>(
        result.location, result.types, result.operands,
        island_builder.getSymbolRefAttr(node_type_name), attribute);
  } else {
    inner_op = island_builder.createOperation(result);
  }

  // Sets operand_segment_sizes or result_segment_sizes attribute to the op.
  const auto set_segment_sizes_attr =
      [&](const NameRangeMap& arg_ranges,
          const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
          llvm::StringRef attr_name) {
        std::vector<mlir::Attribute> values;
        values.reserve(args.size());
        for (const auto& arg : args) {
          auto range = arg_ranges.at(arg.name());
          values.push_back(
              island_builder.getI32IntegerAttr(range.second - range.first));
        }
        auto attr_type =
            mlir::VectorType::get(args.size(), builder_.getIntegerType(32));
        auto attr_value = mlir::DenseElementsAttr::get(attr_type, values);
        inner_op->setAttr(attr_name, attr_value);
      };

  if (inner_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>() ||
      inner_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
    // The op has multiple variadic operands or results.
    // Calculate operand and result segment sizes using the OpDef.
    NameRangeMap input_ranges, output_ranges;
    // This will fail only if the OpDef is syntactically invalid.
    // TODO(jpienaar): Convert this CHECK into a properly propagated error.
    TF_CHECK_OK(
        NameRangesForNode(node, node.op_def(), &input_ranges, &output_ranges));
    if (inner_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
      // Add derived "operand_segment_sizes" attr to the created operation.
      // TODO(b/146937733): Don't use <void> here.
      set_segment_sizes_attr(input_ranges, node.op_def().input_arg(),
                             mlir::OpTrait::AttrSizedOperandSegments<
                                 void>::getOperandSegmentSizeAttr());
    }

    if (inner_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
      // Add derived "result_segment_sizes" attr to the created operation.
      // TODO(b/146937733): Don't use <void> here.
      set_segment_sizes_attr(output_ranges, node.op_def().output_arg(),
                             mlir::OpTrait::AttrSizedResultSegments<
                                 void>::getResultSegmentSizeAttr());
    }
  }

  // Add the terminator for the island
  island_builder.create<mlir::tf_executor::YieldOp>(result.location,
                                                    inner_op->getResults());
  return island.getOperation();
}

Status ImporterBase::ConvertNode(const Node& node) {
  if (!node.IsOp()) {
    // Don't import the pseudo-nodes _SOURCE or _SINK. These are added by
    // Graph and don't exist in GraphDef.
    return Status::OK();
  }

  // If it is a custom OP, its definition should be found in the library. We
  // create the MLIR function and insert it to the module if it doesn't exist.
  std::string node_type_name = node.type_string();
  const auto* func_def = graph_flib_.Find(node_type_name);
  bool convert_to_legacy_call = false;
  if (func_def) {
    TF_RETURN_IF_ERROR(ConvertLibFunction(node_type_name));
    node_type_name = (*tf_name_to_mlir_name_)[node_type_name];
    convert_to_legacy_call = true;
  }

  auto get_full_op_name = [&](const std::string& op_name) {
    const char* kTfPrefix = "tf.";
    return kTfPrefix + op_name;
  };

  std::string op_name = get_full_op_name(node_type_name);
  if (back_edge_node_output_.contains(&node)) {
    op_name = op_name + ".sink";
  }

  const auto& node_def = node.def();
  mlir::OperationState result(GetLocation(node_def), op_name);
  for (int i = 0; i < node.num_outputs(); ++i) {
    // The backedge has been removed, so we shouldn't count the corresponding
    // output from the src node when converting to an operation.
    if (back_edge_node_output_.contains(&node) &&
        back_edge_node_output_[&node] == i) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto type, InferOutputType(node, i, builder_));
    result.types.push_back(type);
  }

  // Surprisingly input edges can be nondeterministically ordered. This
  // particularly seems to be the case for the control edges between _SOURCE
  // and _SINK that the Graph constructor inserts. Copy the input edges and
  // sort the edges, but only the control edges, not data edges!
  // TODO(jmolloy): We should probably just ignore _SOURCE and _SINK nodes.
  // They'll break roundtripping anyway unless we strip them when converting
  // back to graphdef.
  absl::InlinedVector<const Edge*, 8> in_edges(node.in_edges().size());
  absl::c_copy(node.in_edges(), in_edges.begin());
  absl::c_stable_sort(in_edges, [](const Edge* e1, const Edge* e2) {
    if (e1->IsControlEdge() && !e2->IsControlEdge()) return false;
    if (!e1->IsControlEdge() && e2->IsControlEdge()) return true;
    if (e1->IsControlEdge() && e2->IsControlEdge())
      return e1->src()->id() < e2->src()->id();
    return e1->dst_input() < e2->dst_input();
  });

  result.operands.reserve(in_edges.size());

  // Collect the control operands separately, they will be held by the island.
  mlir::SmallVector<mlir::Value, 8> control_operands;

  for (const auto* input_edge : in_edges) {
    const Node& input_node = *input_edge->src();
    if (input_node.IsSource()) {
      if (in_edges.size() != 1) {
        return errors::FailedPrecondition(
            "The node has other inputs besides the _Source node");
      }
      // We don't import the _SOURCE node.
      continue;
    }
    if (input_node.IsArg() && input_edge->IsControlEdge()) {
      // Currently we have not reached consensus as to what TF function
      // semantics are (b/133509504). Here we assume that all arguments to a
      // function should be available before we start execution of any internal
      // node. This makes the control dependencies between function arguments
      // and internal nodes redundant, and so we do not import them. The TF
      // inliner however assumes no such dependency between function args and
      // internal nodes exists, unless explicitly stated. Since we drop control
      // dependencies here, it leads to loss of information. If the function is
      // inlined later, the inliner would not know of these explicit control
      // dependencies present in the original graph.
      continue;
    }
    if (node_values_.find(input_node.id()) == node_values_.end())
      return errors::FailedPrecondition(
          "Graph not traversed in reverse post order; use seen before def!");
    mlir::Operation* inst = node_values_[input_node.id()];
    if (input_edge->IsControlEdge())
      control_operands.push_back(inst->getResult(inst->getNumResults() - 1));
    else
      result.operands.push_back(inst->getResult(input_edge->src_output()));
  }

  using FuncPairType = std::pair<const std::string*, const AttrValue*>;
  std::vector<FuncPairType> funcs;
  result.attributes.reserve(node.attrs().size() + 2);
  auto abstract_op = result.name.getAbstractOperation();
  auto derived_op =
      abstract_op
          ? abstract_op->getInterface<mlir::DerivedAttributeOpInterface>()
          : nullptr;
  for (const auto& name_and_value : node.attrs()) {
    const auto& attr_name = name_and_value.first;
    // Skip adding derived attributes to the generated op.
    if (derived_op && derived_op->isDerivedAttribute(attr_name)) continue;
    const AttrValue& attr_value = name_and_value.second;

    // Remove _output_shapes attribute that will be added by the exporter.
    if (IsOutputShapesAttribute(attr_value, attr_name)) continue;

    // We represent the _diable_call_shape_inference attribute and remove
    // the _output_shapes attribute for LegacyCall. If a call has other
    // attributes, we can't convert it to LegacyCall.
    if (convert_to_legacy_call &&
        !IsDisableCallShapeInferenceAttribute(attr_value, attr_name)) {
      convert_to_legacy_call = false;
    }
    if (attr_value.value_case() == AttrValue::kFunc) {
      // Attribute iteration order is not defined for protocol buffer Map.
      // Process function attributes separately in the lexicographical order to
      // have deterministic order of functions in the constructed IR.
      funcs.emplace_back(&attr_name, &attr_value);
    } else {
      TF_ASSIGN_OR_RETURN(auto attr, ConvertAttributeValue(attr_value));
      result.attributes.push_back(builder_.getNamedAttr(attr_name, attr));
    }
  }

  auto comparator = [](const FuncPairType& a, const FuncPairType& b) {
    return *a.first < *b.first;
  };
  std::sort(funcs.begin(), funcs.end(), comparator);
  for (const auto& func : funcs) {
    TF_RETURN_IF_ERROR(ConvertFunctionCallAttribute(*func.first, *func.second,
                                                    &result.attributes));
  }

  result.attributes.push_back(builder_.getNamedAttr(
      "device", builder_.getStringAttr(std::string(node_def.device()))));

  // Map If and StatelessIf op in TensorFlow to the common If op in MLIR and add
  // the differentiating attribute.
  if (node.IsIfNode()) {
    result.name = mlir::OperationName(get_full_op_name("If"), context_);
    mlir::BoolAttr val = builder_.getBoolAttr(node_type_name == "StatelessIf");
    result.attributes.push_back(builder_.getNamedAttr("is_stateless", val));
  }

  // Map While and StatelessWhile op in TensorFlow to the common While op in
  // MLIR and add the differentiating attribute.
  if (node.IsWhileNode()) {
    result.name = mlir::OperationName(get_full_op_name("While"), context_);
    mlir::BoolAttr val =
        builder_.getBoolAttr(node_type_name == "StatelessWhile");
    result.attributes.push_back(builder_.getNamedAttr("is_stateless", val));
  }

  // Register the mapping between the TF node and the newly created operation.
  node_values_[node.id()] = CreateOperation(
      node, node_type_name, result, control_operands, convert_to_legacy_call);
  return Status::OK();
}

// Add the backedges to the CFG. Given a backedge, we replace the original
// source and destination operations by two new operations. Most of the
// fields of the replacements are copied from the original operations.
// However,
// - for the src operation, one output is inserted to the front of the output
//   list. The type of the output is set to the type of the non-control result
//   of the dst operation, and
// - for the dst operation, one operand is inserted to the front of the
//   operand list. This operand is using the first result of the src
//   operation.
// TODO(fengliuai): Preserve the order of the results and operands if
// necessary.
Status ImporterBase::AddBackedges() {
  for (auto it : back_edge_dst_inputs_) {
    BackEdge& edge = it.second;
    if (!edge.src->IsNextIteration() || !edge.dst->IsMerge()) {
      return errors::FailedPrecondition(
          "Invalid backedge; should be from NextIteration to Merge!");
    }
    auto* sink = node_values_[edge.src->id()];
    auto* dst = node_values_[edge.dst->id()];
    TF_RETURN_IF_ERROR(AddBackedge(sink, dst, edge.dst_input));
  }
  return Status::OK();
}

Status ImporterBase::AddBackedge(mlir::Operation* sink, mlir::Operation* dst,
                                 int dst_input) {
  // Get the NextIteration.Source operation from the token operand of the sink.
  mlir::Operation* source = sink->getOperand(0).getDefiningOp();

  // Adds the "source" to the operands of the dst by creating a new dst
  // operation.
  mlir::OperationState state(dst->getLoc(), dst->getName());
  auto num_operands = dst->getNumOperands();
  state.operands.reserve(num_operands + 1);
  for (int input = 0, e = num_operands + 1; input != e; ++input) {
    if (input < dst_input) {
      state.operands.push_back(dst->getOperand(input));
    } else if (input == dst_input) {
      state.operands.push_back(source->getResult(0));
    } else {
      state.operands.push_back(dst->getOperand(input - 1));
    }
  }
  state.attributes.assign(dst->getAttrs().begin(), dst->getAttrs().end());
  state.types.assign(dst->getResultTypes().begin(),
                     dst->getResultTypes().end());
  builder_.setInsertionPoint(dst);
  auto* new_dst = builder_.createOperation(state);

  // Replaces the output uses of the old operation by the corresponding
  // result of the new operation, and deletes the old operation.
  for (unsigned i = 0, e = dst->getNumResults(); i != e; ++i) {
    auto new_output = new_dst->getResult(i);
    dst->getResult(i).replaceAllUsesWith(new_output);
  }
  dst->dropAllReferences();
  dst->erase();
  return Status::OK();
}

StatusOr<mlir::FunctionType> ImporterBase::InferLibFunctionType(
    const FunctionBody& fbody) {
  mlir::Builder builder(context_);

  // The FunctionBody contains a graph with a single-output _Arg node for each
  // function argument and a single-input _Retval node for each function return
  // value.
  //
  // We already populated the ShapeRefiner with all the information about the
  // shapes of these graph edges, so we just query it to build the corresponding
  // MLIR function type signature.

  llvm::SmallVector<mlir::Type, 4> arg_types;
  if (specs_.inputs.empty()) {
    arg_types.reserve(fbody.arg_types.size());
    for (auto arg : fbody.arg_nodes) {
      // Find node in the graph using the node id instead of using `arg`
      // directly because the graph has been cloned.
      auto* node = graph_->FindNodeId(arg->id());
      TF_ASSIGN_OR_RETURN(auto type,
                          InferOutputType(*node, /*idx=*/0, builder));
      arg_types.push_back(type);
    }
  } else {
    arg_types.reserve(fbody.arg_types.size());
    for (const auto& it : llvm::enumerate(specs_.inputs)) {
      mlir::Type element_type;
      const auto& node_info = it.value().second;
      DataType dtype = node_info.imported_dtype;
      // Uses the existing output type of the arg node if the data type of the
      // the node isn't specified through the import configuration.
      if (dtype == DT_INVALID) {
        auto arg = fbody.arg_nodes[it.index()];
        auto* node = graph_->FindNodeId(arg->id());
        dtype = node->output_type(0);
        if (dtype == DT_INVALID) {
          return errors::InvalidArgument("Input ", it.index(),
                                         "has invalid data type");
        }
      }
      TF_RETURN_IF_ERROR(
          ::tensorflow::ConvertDataType(dtype, builder, &element_type));
      if (node_info.shape.unknown_rank()) {
        arg_types.push_back(mlir::UnrankedTensorType::get(element_type));
      } else {
        llvm::SmallVector<int64_t, 4> shape;
        TF_RETURN_IF_ERROR(ConvertToMlirShape(node_info.shape, &shape));
        arg_types.push_back(mlir::RankedTensorType::get(shape, element_type));
      }
    }
  }

  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(fbody.ret_types.size());
  for (auto ret : fbody.ret_nodes) {
    // Find node in the graph using the node id instead of using `ret` directly
    // because the graph has been cloned.
    auto* node = graph_->FindNodeId(ret->id());
    TF_ASSIGN_OR_RETURN(auto type, InferInputType(*node, /*idx=*/0, builder));
    ret_types.push_back(type);
  }

  return builder.getFunctionType(arg_types, ret_types);
}
  
}  // namespace tensorflow
