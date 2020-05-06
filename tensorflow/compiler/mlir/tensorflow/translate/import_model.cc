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

#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"

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

namespace {

// Stateful helper class to import a TensorFlow model expressed in SavedModel
// into an MLIR Module.
class SavedModelObjectGraphImporter : public ImporterBase {
 public:
  // Main entry point: converts all functions in the given meta graph to an MLIR
  // Module.
  static StatusOr<mlir::OwningModuleRef> Convert(
      SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
      absl::Span<std::string> exported_names, bool add_default_attributes);

 private:
  explicit SavedModelObjectGraphImporter(
      const FunctionLibraryDefinition& flib, const GraphDebugInfo& debug_info,
      const GraphImportConfig& specs, mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      NameUniquifier* function_name_uniquifier)
      : ImporterBase(flib, debug_info, specs, module, tf_name_to_mlir_name,
                     function_name_uniquifier) {}
};

// Determines the names used to reference objects in the SavedObjectGraph.
class ObjectNames {
 public:
  explicit ObjectNames(const SavedObjectGraph& object_graph,
                       absl::Span<std::string> exported_names);

  // Gets the names that external users of the SavedModel can use to refer to
  // this node.
  llvm::ArrayRef<llvm::StringRef> GetExportedNames(int node_id) const;

  // Gets the name in the module symbol table for this node.
  // This name is only used for internal IR references.
  llvm::StringRef GetSymbolTableName(int node_id) const;

 private:
  // In the absence of any other information, use this name as the symbol table
  // name for this node.
  std::string GetDefaultSymbolTableName(int node_id) const;
  // Determines if a name is exported.
  bool IsExported(const std::string& name);
  // Main object graph traversal function.
  void RecursivelyVisitObjectGraph(int node_id);
  // Gets a stable StringRef from a std::string.
  llvm::StringRef SaveString(const std::string& s) const;

  // The object graph we are traversing.
  const SavedObjectGraph& object_graph_;
  // The set of names to export. Empty means "export all".
  std::unordered_set<std::string> names_to_export_;

  // When we recursively follow the object graph tree structure from the root,
  // we track its path in the object graph by pushing and popping from here
  // during traversal.
  llvm::SmallVector<std::string, 8> path_segments_;
  // The set of node_id's that are on the current DFS stack.
  // For cyclic object graphs, this prevents infinite recursion.
  std::unordered_set<int> on_stack_nodes_;

  // Key: node_id.
  // Value: all object names that node_id appears as.
  // Each object name corresponds to a unique path from the root of the object
  // graph.
  // The common intuitive case is when there is only one name for a given
  // object, which corresponds to the object graph being a tree.
  //
  // But, there cases where the object graph is a general graph. For
  // example, this happens commonly in Keras models, where `foo.bar` is
  // also reachable via the name `keras_api.foo.bar`.
  // Cycles are possible too.
  absl::flat_hash_map<int, std::vector<std::string>> object_names_;

  // Key: node_id
  // Value: all names that this object is exported as
  absl::flat_hash_map<int, llvm::SmallVector<llvm::StringRef, 1>>
      exported_names_;
  // Key: node_id
  // Value: pretty symbol table name to use for internal references to this
  // object.
  absl::flat_hash_map<int, llvm::StringRef> pretty_symbol_table_name_;

  // Stable strings we can take StringRef's into. Used only by the SaveString
  // method.
  mutable std::unordered_set<std::string> saved_strings_;
};

ObjectNames::ObjectNames(const SavedObjectGraph& object_graph,
                         absl::Span<std::string> exported_names)
    : object_graph_(object_graph),
      names_to_export_(exported_names.begin(), exported_names.end()) {
  // Visit all reachable nodes from the root of the object graph.
  // This builds up object_names_ to contain all names like `foo.bar` that a
  // particular node in the graph can be reached from.
  RecursivelyVisitObjectGraph(/*node_id=*/0);

  // Populate the exported_names_ map.
  // TODO(silvasean): Diagnose typos in exported names?
  for (auto& kv : object_names_) {
    // Make object names map independent of our particular choice of object
    // graph traversal.
    std::sort(kv.second.begin(), kv.second.end(),
              [](absl::string_view a, absl::string_view b) {
                // The sort order here influences the "pretty name" we assign
                // below. We want the most debuggable name to be first.
                //
                // Debuggability heuristics:
                // 1. Names that end in digits are likely to be internal aliases
                // to the "real" names.
                // 2. Longer names are more likely to be internal aliases.
                //
                // Example set of object names created by Keras for the weight
                // matrix of a fully connected layer on a trivial FC mnist
                // model:
                // - `model.layer-1.kernel` (this is the "best" name)
                // - `model.keras_api.layers.1.kernel`
                // - `model.variables.0`
                // - `model.keras_api.layers.1.keras_api.trainable_variables.0`
                // - ... 10 more long aliases ending in digits ...
                return std::make_tuple(isdigit(a.back()), a.size(), a) <
                       std::make_tuple(isdigit(b.back()), b.size(), b);
              });
    for (const std::string& name : kv.second) {
      if (IsExported(name)) {
        exported_names_[kv.first].push_back(SaveString(name));
      }
    }
  }
  // Create "pretty" symbol table names for nodes where that is applicable.
  // We could make all symbol table names use the default, which is basically
  // just the node id. But for debugging purposes, it's nicer if we can mix in
  // a recognizable object name if we have the information to do so.
  for (auto& kv : object_names_) {
    int node_id = kv.first;
    std::string internal_name =
        absl::StrCat(GetDefaultSymbolTableName(node_id), "__");
    // If the object has an exported name, we prefer that since it is probably
    // the most recognizable. Otherwise, we grab some non-exported name of the
    // object.
    if (exported_names_.find(node_id) != exported_names_.end()) {
      internal_name += exported_names_[node_id][0].str();
    } else {
      internal_name += object_names_[node_id][0];
    }
    pretty_symbol_table_name_[node_id] = SaveString(internal_name);
  }
}

llvm::ArrayRef<llvm::StringRef> ObjectNames::GetExportedNames(
    int node_id) const {
  auto it = exported_names_.find(node_id);
  if (it != exported_names_.end()) {
    return it->second;
  }
  return {};
}

llvm::StringRef ObjectNames::GetSymbolTableName(int node_id) const {
  auto it = pretty_symbol_table_name_.find(node_id);
  if (it != pretty_symbol_table_name_.end()) {
    return it->second;
  }
  return SaveString(GetDefaultSymbolTableName(node_id));
}

std::string ObjectNames::GetDefaultSymbolTableName(int node_id) const {
  return absl::StrCat("__sm_node", node_id);
}

bool ObjectNames::IsExported(const std::string& name) {
  if (names_to_export_.empty()) {
    return true;
  }
  return names_to_export_.find(name) != names_to_export_.end();
}

void ObjectNames::RecursivelyVisitObjectGraph(int node_id) {
  const SavedObject& object = object_graph_.nodes(node_id);

  switch (object.kind_case()) {
    case SavedObject::kConstant:
    case SavedObject::kFunction:
    case SavedObject::kVariable: {
      object_names_[node_id].push_back(absl::StrJoin(path_segments_, "."));
      break;
    }
    default:
      break;
  }

  for (const auto& child_ref : object.children()) {
    bool on_stack = !on_stack_nodes_.insert(child_ref.node_id()).second;
    if (on_stack) {
      // This is a backedge. Don't traverse it.
      continue;
    }

    path_segments_.push_back(child_ref.local_name());
    RecursivelyVisitObjectGraph(child_ref.node_id());
    path_segments_.pop_back();

    on_stack_nodes_.erase(child_ref.node_id());
  }
}

llvm::StringRef ObjectNames::SaveString(const std::string& s) const {
  return llvm::StringRef(*saved_strings_.insert(s).first);
}

// Extracts a TensorProto for a Const op from a GraphDef, given an op_name.
// Returns nullptr on not found or other mismatch.
// This returns a pointer to the actual node within the graph_def so as to
// avoid expensive copies.
const TensorProto* ExtractConstTensorFromGraph(const GraphDef& graph_def,
                                               const std::string& op_name) {
  const NodeDef* match_node = nullptr;
  for (const auto& node : graph_def.node()) {
    if (node.name() == op_name) {
      match_node = &node;
    }
  }

  if (!match_node) {
    return nullptr;
  }

  auto value_it = match_node->attr().find("value");
  if (value_it == match_node->attr().end()) {
    return nullptr;
  }

  if (!value_it->second.has_tensor()) {
    return nullptr;
  }

  return &value_it->second.tensor();
}

const TrackableObjectGraph::TrackableObject::SerializedTensor*
FindSerializedTensorInTrackable(
    const TrackableObjectGraph::TrackableObject& trackable_object,
    StringPiece name) {
  for (const auto& maybe_serialized_tensor : trackable_object.attributes()) {
    if (maybe_serialized_tensor.name() == name) {
      return &maybe_serialized_tensor;
    }
  }
  return nullptr;
}

Status DiagnoseMultipleConcreteFunctions(const SavedObjectGraph& object_graph,
                                         const ObjectNames& object_names) {
  for (int node_id = 0; node_id < object_graph.nodes_size(); node_id++) {
    const SavedObject& object = object_graph.nodes(node_id);
    if (object_names.GetExportedNames(node_id).empty()) {
      continue;
    }
    if (object.kind_case() == SavedObject::kFunction) {
      // We only allow a single input signature to each SavedFunction.
      // This assumption means we have a 1:1 correspondence between
      // tf.function <=> SavedFunction <=> SavedConcreteFunction <=> FunctionDef
      // This makes defining the ABI easier (or even well-defined at all).
      // TODO(silvasean): How to detect a function that doesn't have an
      // explicitly user-provided input signature, but happens to have been
      // traced exactly once?
      if (object.function().concrete_functions_size() != 1) {
        llvm::SmallVector<std::string, 4> names;
        for (llvm::StringRef s : object_names.GetExportedNames(node_id)) {
          names.push_back("'" + s.str() + "'");
        }
        return errors::InvalidArgument(
            "Exported function with exported name(s) ",
            absl::StrJoin(names, ", "),
            " with multiple concrete functions. Add "
            "@tf.function(input_signature=[...]) on this function, or use a "
            "narrower list of exported names that excludes this function.");
      }
    }
  }
  return Status::OK();
}

// Recursively traverses a StructuredValue, linearizing all the leaves.
//
// This currently only handles the subset of StructuredValue that is needed for
// signatures.
//
// Given a StructuredValue with structure [{"x": leaf0}], the "index path"
// needed to reach leaf0 is `[0, "x"]`, as it would be if you were operating on
// a Python object (`obj[0]["x"] is leaf0`). Each leaf corresponds to a
// linearized function argument or return on a FunctionDef, and hence to an
// mlir::FuncOp argument / return.
//
// This must match the linearization that happens in `tf.nest.flatten`.
// In particular, dict values should be linearized in sorted key order.
//
// The linearized index paths can be returned back to a structured
// representation (e.g. to emit C structs matching a signature) with a simple
// algorithm that recurses on each run of index paths with identical first
// elements.
class StructuredValueLinearizer {
 public:
  StructuredValueLinearizer(const StructuredValue& value,
                            mlir::MLIRContext* context);

  // Returns the list of index paths to each leaf of the StructuredValue,
  // in a linearized order matching `tf.nest.flatten`.
  //
  // If an error occurred during the linearization process, an error message
  // with `error_context` prepended will be included in the returned status.
  StatusOr<llvm::ArrayRef<mlir::ArrayAttr>> GetLeafIndexPaths(
      llvm::StringRef error_context) const;

 private:
  // Main function that recursively traverses the StructuredValue.
  void RecursivelyFindLeaves(const StructuredValue& value);

  mlir::Builder builder_;
  // The current index path. We push/pop this during recursive traversal of the
  // StructuredValue.
  llvm::SmallVector<mlir::Attribute, 4> current_index_path_;
  // The list of leaf index paths we have discovered so far.
  llvm::SmallVector<mlir::ArrayAttr, 4> leaf_index_paths_;
  // If non-empty, an error message to report.
  std::string error_message_;
};

StructuredValueLinearizer::StructuredValueLinearizer(
    const StructuredValue& value, mlir::MLIRContext* context)
    : builder_(context) {
  RecursivelyFindLeaves(value);
}

StatusOr<llvm::ArrayRef<mlir::ArrayAttr>>
StructuredValueLinearizer::GetLeafIndexPaths(
    llvm::StringRef error_context) const {
  if (error_message_.empty()) {
    return llvm::makeArrayRef(leaf_index_paths_);
  }
  return errors::InvalidArgument(
      error_context.str(), error_message_,
      "This likely means that you have @tf.function "
      "on an exported function instead of "
      "@tf.function(input_signature=[...]). Consider annotating an "
      "input_signature or narrowing your set of "
      "exported names to not include this function.");
}

void StructuredValueLinearizer::RecursivelyFindLeaves(
    const StructuredValue& value) {
  switch (value.kind_case()) {
    case StructuredValue::kDictValue: {
      // Dict values must be linearized in sorted order of keys.
      const DictValue& dict = value.dict_value();
      using FieldTy = protobuf::MapPair<std::string, StructuredValue>;
      llvm::SmallVector<const FieldTy*, 4> fields;
      for (auto& field : dict.fields()) {
        fields.push_back(&field);
      }
      llvm::sort(fields, [](const FieldTy* a, const FieldTy* b) {
        return a->first < b->first;
      });
      for (auto& field : fields) {
        current_index_path_.push_back(builder_.getStringAttr(field->first));
        RecursivelyFindLeaves(field->second);
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kTupleValue: {
      const TupleValue& tuple = value.tuple_value();
      for (int i = 0, e = tuple.values_size(); i < e; i++) {
        current_index_path_.push_back(builder_.getI64IntegerAttr(i));
        RecursivelyFindLeaves(tuple.values(i));
        current_index_path_.pop_back();
      }
      return;
    }
    // We don't differentiate between tuples and lists.
    case StructuredValue::kListValue: {
      const ListValue& list = value.list_value();
      for (int i = 0, e = list.values_size(); i < e; i++) {
        current_index_path_.push_back(builder_.getI64IntegerAttr(i));
        RecursivelyFindLeaves(list.values(i));
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kTensorSpecValue: {
      // Base case: record the current path stack as the index path needed to
      // get to this leaf.
      leaf_index_paths_.push_back(builder_.getArrayAttr(current_index_path_));
      return;
    }
    case StructuredValue::kNoneValue: {
      // Base case: do nothing.
      // This arises, for example, as the top-level object of an output
      // signature when there are no return values.
      return;
    }
    default: {
      llvm::raw_string_ostream os(error_message_);
      // TODO(silvasean): Use an enumerant name string instead of a number.
      os << "Unhandled structured value kind " << value.kind_case()
         << " at index path: <value>";
      for (auto path_element : current_index_path_) {
        os << ".";
        if (auto integer = path_element.dyn_cast<mlir::IntegerAttr>()) {
          os << integer.getValue();
        } else {
          auto str = path_element.cast<mlir::StringAttr>();
          os << str.getValue();
        }
      }
      os << "\n";
    }
  }
}

// For exported functions with bound inputs, rewrite the function
// signature to match the requirements of tf_saved_model bound input args.
//
// The raw imported functions have `tensor<*x!tf.resource>` as the type for
// mutable bound inputs and `tensor<...>` as the type for immutable
// bound inputs. Here we canonicalize both of them into
// `tensor<!tf.resource<tensor<...>>>`.
void AdjustBoundInputArgTypes(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  for (auto func : module.getOps<mlir::FuncOp>()) {
    if (!mlir::tf_saved_model::IsExported(func)) continue;
    mlir::OpBuilder builder(func.getBody());
    llvm::SmallVector<mlir::Type, 4> new_input_types;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto arg = func.front().getArgument(i);
      auto global_tensor =
          mlir::tf_saved_model::LookupBoundInput(func, i, symbol_table);
      if (global_tensor) {
        auto old_type = arg.getType();
        auto new_type =
            mlir::tf_saved_model::GetBoundInputArgTypeFor(global_tensor);
        arg.setType(new_type);
        if (global_tensor.is_mutable()) {
          auto arg_with_original_type = builder.create<mlir::TF::CastOp>(
              global_tensor.getLoc(), old_type, arg,
              /*Truncate=*/builder.getBoolAttr(false));
          arg.replaceAllUsesWith(arg_with_original_type);
          // The RAUW replaces the arg with itself, so we need to set it back.
          arg_with_original_type.setOperand(arg);
        } else {
          auto arg_with_original_type =
              builder.create<mlir::TF::ReadVariableOp>(global_tensor.getLoc(),
                                                       old_type, arg);
          arg.replaceAllUsesWith(arg_with_original_type);
          // The RAUW replaces the arg with itself, so we need to set it back.
          arg_with_original_type.setOperand(arg);
        }
      }
      new_input_types.push_back(arg.getType());
    }
    func.setType(mlir::FunctionType::get(
        new_input_types, func.getType().getResults(), module.getContext()));
  }
}

// Reorder the ops in the module to make testing easier and less dependent
// on implementation details such as the order of functions in the
// FunctionDefLibrary.
//
// The order this ensures is:
// 1. GlobalTensorOp's
// 2. FuncOps's.
//
// Within each of 1. and 2., ops are sorted by exported name (if
// available, and only the first exported name is considered), followed by
// non-exported ops.
void SortSavedModelModule(mlir::ModuleOp module) {
  struct NamedGlobalTensor {
    llvm::StringRef name;
    GlobalTensorOp global_tensor;
  };
  llvm::SmallVector<NamedGlobalTensor, 8> named_global_tensors;
  for (auto global_tensor : module.getOps<GlobalTensorOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(global_tensor);
    // We use stable_sort, so duplicate empty names are fine here.
    named_global_tensors.push_back(
        {exported_names.empty() ? "" : exported_names.front(), global_tensor});
  }
  llvm::stable_sort(named_global_tensors,
                    [](const NamedGlobalTensor& a, const NamedGlobalTensor& b) {
                      return std::make_tuple(a.name.empty(), a.name) <
                             std::make_tuple(b.name.empty(), b.name);
                    });

  struct NamedFunc {
    llvm::StringRef name;
    mlir::FuncOp func;
  };
  llvm::SmallVector<NamedFunc, 8> named_funcs;
  for (auto func : module.getOps<mlir::FuncOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
    named_funcs.push_back(
        {exported_names.empty() ? "" : exported_names.front(), func});
  }
  llvm::stable_sort(named_funcs, [](const NamedFunc& a, const NamedFunc& b) {
    return std::make_tuple(a.name.empty(), a.name) <
           std::make_tuple(b.name.empty(), b.name);
  });

  // Move onto the front of the module in reverse of the final desired order.
  for (auto named_func : llvm::reverse(named_funcs)) {
    named_func.func.getOperation()->moveBefore(&module.getBody()->front());
  }
  for (auto named_global_tensor : llvm::reverse(named_global_tensors)) {
    named_global_tensor.global_tensor.getOperation()->moveBefore(
        &module.getBody()->front());
  }
}

Status CreateSavedModelIR(
    const ObjectNames& object_names, mlir::ModuleOp module,
    const SavedObjectGraph& object_graph,
    const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name,
    SavedModelV2Bundle* saved_model) {
  mlir::OpBuilder builder(module.getBodyRegion());
  mlir::SymbolTable symbol_table(module);

  // Create a side data-structure, indexed by the object_graph node_id to
  // a TrackableObject that is restorable.
  absl::flat_hash_map<int, const TrackableObjectGraph::TrackableObject*>
      restored_objects;
  TF_RETURN_IF_ERROR(saved_model->VisitObjectsToRestore(
      [&](int saved_node_id,
          const TrackableObjectGraph::TrackableObject& trackable_object) {
        restored_objects.insert(
            std::make_pair(saved_node_id, &trackable_object));
        return Status::OK();
      }));

  for (int node_id = 0; node_id < object_graph.nodes_size(); node_id++) {
    const SavedObject& object = object_graph.nodes(node_id);
    // For correctness, we cannot import functions that don't have exported
    // names, since they don't necessarily have a well-defined ABI (diagnosed
    // earlier).
    //
    // For variables/constants, pruning them is purely an optimization,
    // and more complicated since it requires use-def analysis of which
    // functions use which variables/constants, so we don't do anything
    // special for them here as part of our initial IR construction.
    if (object.kind_case() == SavedObject::kFunction) {
      if (object_names.GetExportedNames(node_id).empty()) {
        continue;
      }
      std::string error_context =
          "While importing SavedModel function '" +
          object_names.GetExportedNames(node_id)[0].str() + "': ";
      const SavedFunction& function = object.function();
      auto orig_func = symbol_table.lookup<mlir::FuncOp>(
          tf_name_to_mlir_name.find(function.concrete_functions(0))->second);
      mlir::FuncOp func = orig_func;
      // If there are potentially references to this func from within the
      // module, create a wrapper around it and decorate the wrapper with the
      // tf_saved_model attributes instead.
      if (!mlir::SymbolTable::symbolKnownUseEmpty(orig_func.getName(),
                                                  &module.getBodyRegion())) {
        func = orig_func.cloneWithoutRegions();
        module.insert(module.getBody()->begin(), func);
        func.addEntryBlock();
        func.setName("__sm_exported_" + orig_func.getName().str());
        llvm::SmallVector<mlir::Value, 4> args_as_values;
        for (auto block_argument : func.getArguments()) {
          args_as_values.push_back(block_argument);
        }
        mlir::OpBuilder body_builder(&func.getBody());
        auto call = body_builder.create<mlir::TF::StatefulPartitionedCallOp>(
            func.getLoc(), orig_func.getType().getResults(), args_as_values,
            builder.getSymbolRefAttr(orig_func.getName()),
            /*config=*/builder.getStringAttr(""),
            /*config_proto=*/builder.getStringAttr(""),
            /*executor_type=*/builder.getStringAttr(""));
        body_builder.create<mlir::ReturnOp>(func.getLoc(), call.getResults());
      }
      func.setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      const SavedConcreteFunction& concrete_function =
          object_graph.concrete_functions().at(function.concrete_functions(0));

      // We do not handle the other element of this tuple, which corresponds to
      // Python kwonlyargs, since currently TensorFlow prohibits this in
      // combination with input_signature:
      // https://github.com/tensorflow/tensorflow/blob/8cb8627abb5ef83a6fba34f8fd0e4ee430562eb1/tensorflow/python/eager/function.py#L2027-L2030
      // Our SavedModel import requires input_signature on the tf.function, so
      // we never need to handle the kwonlyargs.
      auto positional_arg_structure =
          concrete_function.canonicalized_input_signature()
              .tuple_value()
              .values(0);
      StructuredValueLinearizer input_linearizer(positional_arg_structure,
                                                 builder.getContext());

      int bound_input_base =
          func.getNumArguments() - concrete_function.bound_inputs_size();
      TF_ASSIGN_OR_RETURN(auto input_index_paths,
                          input_linearizer.GetLeafIndexPaths(
                              error_context + "in input signature: "));
      if (bound_input_base != input_index_paths.size()) {
        return errors::InvalidArgument(
            error_context,
            "Argument mismatch between concrete function input signature "
            "vs underlying FunctionDef for concrete function '",
            function.concrete_functions(0), "' (", input_index_paths.size(),
            " vs ", bound_input_base, ")");
      }
      for (auto index_path : llvm::enumerate(input_index_paths)) {
        func.setArgAttr(index_path.index(), "tf_saved_model.index_path",
                        index_path.value());
      }

      for (auto& bound_input :
           llvm::enumerate(concrete_function.bound_inputs())) {
        int arg_index = bound_input_base + bound_input.index();
        auto symbol_ref = builder.getSymbolRefAttr(
            object_names.GetSymbolTableName(bound_input.value()));
        func.setArgAttr(arg_index, "tf_saved_model.bound_input", symbol_ref);
      }

      StructuredValueLinearizer output_linearizer(
          concrete_function.output_signature(), builder.getContext());
      TF_ASSIGN_OR_RETURN(auto output_index_paths,
                          output_linearizer.GetLeafIndexPaths(
                              error_context + "in output signature: "));
      if (func.getNumResults() != output_index_paths.size()) {
        return errors::InvalidArgument(
            error_context,
            "Result mismatch between concrete function output signature "
            "vs underlying FunctionDef for concrete function '",
            function.concrete_functions(0), "' (", output_index_paths.size(),
            " vs ", func.getNumResults(), ")");
      }
      for (auto index_path : llvm::enumerate(output_index_paths)) {
        func.setResultAttr(index_path.index(), "tf_saved_model.index_path",
                           index_path.value());
      }
    } else if (object.kind_case() == SavedObject::kVariable) {
      const SavedVariable& variable = object.variable();
      // Find the trackable in the side data structure.
      auto variable_trackable_it = restored_objects.find(node_id);
      if (variable_trackable_it == restored_objects.end()) {
        return errors::FailedPrecondition("Could not restore saved variable: ",
                                          variable.name());
      }
      const auto* serialized_tensor_attr = FindSerializedTensorInTrackable(
          *variable_trackable_it->second, "VARIABLE_VALUE");
      if (!serialized_tensor_attr) {
        return errors::FailedPrecondition(
            "Could not find serialized tensor for saved variable: ",
            variable.name());
      }
      const auto& checkpoint_key = serialized_tensor_attr->checkpoint_key();

      // Load it from the reader.
      Tensor value;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          saved_model->variable_reader()->Lookup(checkpoint_key, &value),
          "Could not read checkpoint key from variables bundle: ",
          checkpoint_key);
      TF_ASSIGN_OR_RETURN(auto value_attr, ConvertTensor(value, &builder));
      // A variable can have a partially known type, such as tensor<?x27x?xf32>,
      // even if the initializer is a specific static shape.
      TF_ASSIGN_OR_RETURN(
          auto type, ConvertToMlirTensorType(variable.shape(), variable.dtype(),
                                             &builder));
      auto op = builder.create<GlobalTensorOp>(
          builder.getUnknownLoc(),
          builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
          value_attr,
          /*type=*/mlir::TypeAttr::get(type),
          /*is_mutable=*/builder.getUnitAttr());
      op.setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    } else if (object.kind_case() == SavedObject::kConstant) {
      const SavedConstant& constant = object.constant();
      const TensorProto* value = ExtractConstTensorFromGraph(
          saved_model->meta_graph_def().graph_def(), constant.operation());
      if (!value) {
        return errors::FailedPrecondition(
            "Unable to find const node referenced in object graph: ",
            constant.operation());
      }
      TF_ASSIGN_OR_RETURN(auto value_attr,
                          ConvertTensorProto(*value, &builder));
      auto op = builder.create<GlobalTensorOp>(
          builder.getUnknownLoc(),
          builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
          value_attr,
          /*type=*/mlir::TypeAttr::get(value_attr.Attribute::getType()),
          /*is_mutable=*/nullptr);
      op.setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    }
  }
  AdjustBoundInputArgTypes(module);
  module.setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(module);
  return Status::OK();
}

StatusOr<mlir::OwningModuleRef> SavedModelObjectGraphImporter::Convert(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, bool add_default_attributes) {
  GraphDebugInfo dummy_debug_info;
  const GraphDebugInfo& debug_info =
      saved_model->debug_info() ? *saved_model->debug_info() : dummy_debug_info;

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;

  const auto& graphdef = saved_model->meta_graph_def().graph_def();
  PopulateTfVersions(module.get(), graphdef.versions());

  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = add_default_attributes;
  Graph graph(OpRegistry::Global());

  GraphDef preprocessed_graphdef(graphdef);
  if (add_default_attributes) {
    TF_RETURN_IF_ERROR(PreprocessGraphDef(nullptr, &preprocessed_graphdef));
  }

  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(options, preprocessed_graphdef, &graph));

  NameUniquifier function_name_uniquifier(graph.flib_def());
  SavedModelObjectGraphImporter importer(graph.flib_def(), debug_info, specs,
                                         module.get(), &tf_name_to_mlir_name,
                                         &function_name_uniquifier);

  TF_RETURN_IF_ERROR(importer.PrepareConvert(graph));

  auto fn_names = graph.flib_def().ListFunctionNames();
  for (const auto& fn_name : fn_names) {
    TF_RETURN_IF_ERROR(importer.ConvertLibFunction(fn_name));
  }

  if (!saved_model->meta_graph_def().has_object_graph_def()) {
    return errors::InvalidArgument(
        "SavedModel does not have an object graph. Please use TF2.");
  }
  auto& object_graph = saved_model->meta_graph_def().object_graph_def();
  ObjectNames object_names(object_graph, exported_names);

  // Clean up a couple func's that always seem to be present when importing a
  // SavedModel. This is not strictly needed, as there is a separate pass that
  // will clean them up, but this makes staring at the raw IR of minimal
  // examples quite a bit nicer.
  for (auto func : llvm::make_early_inc_range(module->getOps<mlir::FuncOp>())) {
    if (func.getName().startswith("__inference__traced_save_") ||
        func.getName().startswith("__inference__traced_restore_") ||
        func.getName().startswith("__inference_signature_wrapper_")) {
      func.erase();
    }
  }

  // Diagnose SavedFunction's with multiple input signatures.
  TF_RETURN_IF_ERROR(
      DiagnoseMultipleConcreteFunctions(object_graph, object_names));

  // Construct the SavedModel IR.
  TF_RETURN_IF_ERROR(CreateSavedModelIR(object_names, module.get(),
                                        object_graph, tf_name_to_mlir_name,
                                        saved_model));
  assert(mlir::succeeded(mlir::verify(module.get())));

  return module;
}

// A helper class to import a TensorFlow model expressed in SavedModel V1 into
// an MLIR Module in SavedModel dialect.
class SavedModelSignatureDefImporter {
 public:
  // Main entry point: converts all functions (specified by SignatureDefs) in
  // the given meta graph to an MLIR Module.
  static StatusOr<mlir::OwningModuleRef> Convert(const SavedModelBundle& bundle,
                                                 mlir::MLIRContext* context) {
    SavedModelSignatureDefImporter importer(bundle, context);

    return importer.ConvertSignatures();
  }

 private:
  SavedModelSignatureDefImporter(const SavedModelBundle& bundle,
                                 mlir::MLIRContext* context)
      : bundle_(bundle),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(context))) {}

  // Converts the SavedModel to the SavedModel dialect. Creates an MLIR function
  // for each signature.
  StatusOr<mlir::OwningModuleRef> ConvertSignatures();
  Status ConvertSignature(const GraphDef& graphdef,
                          const std::string& sig_def_key,
                          const SignatureDef& signature_def,
                          const GraphDebugInfo& debug_info,
                          const FunctionLibraryDefinition& flib_def);

  // Creates GlobalTensorOp for each variable and moves each VarHandle op to
  // the enclosing function's arguments.
  Status LiftVariables();

  // Moves the result of the VarHandleOp with corresponding global tensor to the
  // enclosing function's argument list and erases this VarHandleOp. The global
  // tensor's shape is used to provide the most accurate nested shape.
  void LiftVariable(VarHandleOp op, GlobalTensorOp global_tensor);

  using VarGlobalMap = llvm::MapVector<
      llvm::StringRef,
      std::pair<GlobalTensorOp, llvm::SmallVector<VarHandleOp, 2>>>;

  // Reads all variables from the SavedModel through session and creates
  // GlobalTensorOp for these variables.
  Status ReadVariablesFromSession(VarGlobalMap* var_globals);

  GraphImportConfig::InputArrays ParseInputArrays(
      const std::vector<std::pair<std::string, TensorInfo>>& inputs);

  const SavedModelBundle& bundle_;
  mlir::OwningModuleRef module_;
};

StatusOr<mlir::OwningModuleRef>
SavedModelSignatureDefImporter::ConvertSignatures() {
  const auto& signatures = bundle_.GetSignatures();
  const auto& graphdef = bundle_.meta_graph_def.graph_def();
  PopulateTfVersions(module_.get(), graphdef.versions());

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graphdef.library());

  // debug_info might not be loaded with loader_lite.
  GraphDebugInfo debug_info;
  if (bundle_.debug_info != nullptr) debug_info = *bundle_.debug_info;

  for (const auto& key_and_signature_def : signatures) {
    const std::string& sig_def_key = key_and_signature_def.first;
    const SignatureDef& signature_def = key_and_signature_def.second;

    // It is safe to skip "__saved_model_init_op" since it is an internal
    // signature that is not user-accessible.
    if (sig_def_key == "__saved_model_init_op") {
      continue;
    }

    TF_RETURN_IF_ERROR(ConvertSignature(graphdef, sig_def_key, signature_def,
                                        debug_info, flib_def));
  }
  TF_RETURN_IF_ERROR(LiftVariables());

  mlir::OpBuilder builder(module_->getBodyRegion());
  module_->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(*module_);

  return std::move(module_);
}

Status SavedModelSignatureDefImporter::ConvertSignature(
    const GraphDef& graphdef, const std::string& sig_def_key,
    const SignatureDef& signature_def, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def) {
  // Create local vectors for the input and output and sort them to be
  // deterministic. We don't want anyone to really depend on the order, client
  // should lookup argument/result mapping by attribute name.
  // To avoid accidentally depending on the order we use an unintuitive sorting.
  std::vector<std::pair<std::string, TensorInfo>> inputs(
      signature_def.inputs().begin(), signature_def.inputs().end());
  llvm::sort(inputs, [](const auto& lhs, const auto& rhs) {
    return lhs.first.size() < rhs.first.size() || lhs.first > rhs.first;
  });
  std::vector<std::pair<std::string, TensorInfo>> outputs(
      signature_def.outputs().begin(), signature_def.outputs().end());
  llvm::sort(outputs, [](const auto& lhs, const auto& rhs) {
    return lhs.first.size() < rhs.first.size() || lhs.first > rhs.first;
  });

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.inputs = ParseInputArrays(inputs);
  for (auto& output : outputs) specs.outputs.push_back(output.second.name());

  // Remove unused nodes and create sub-graphdef.
  GraphDef sub_graph_def;
  TF_RETURN_IF_ERROR(tensorflow::grappler::SetTransitiveFaninGraph(
      graphdef, &sub_graph_def,
      /*terminal_nodes=*/{specs.outputs.begin(), specs.outputs.end()}));

  // Convert sub-graphdef to sub-graph.
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  Graph sub_graph(OpRegistry::Global());

  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(options, sub_graph_def, &sub_graph));

  // Convert sub-graph to MLIR module.
  TF_ASSIGN_OR_RETURN(
      auto sub_module,
      GraphDefImporter::Convert(module_->getContext(), sub_graph, debug_info,
                                flib_def, specs, sig_def_key));
  mlir::OpBuilder builder(sub_module->getBodyRegion());

  // Find the FuncOp which corresponds to current SignatureDef.
  mlir::SymbolTable symbol_table(*sub_module);
  auto func_op = symbol_table.lookup<mlir::FuncOp>(sig_def_key);
  TF_RET_CHECK(func_op)
      << "Graphdef importer should have created a function named "
      << sig_def_key << ".";

  // Use unique SignatureDef key as exported name.
  func_op.setAttr("tf_saved_model.exported_names",
                  builder.getStrArrayAttr({sig_def_key}));

  // Transfer input and output parameter names to index_path attributes.
  for (auto input_and_idx : llvm::enumerate(inputs)) {
    func_op.setArgAttr(input_and_idx.index(), "tf_saved_model.index_path",
                       builder.getStrArrayAttr({input_and_idx.value().first}));
  }
  for (auto output_and_idx : llvm::enumerate(outputs)) {
    func_op.setResultAttr(
        output_and_idx.index(), "tf_saved_model.index_path",
        builder.getStrArrayAttr({output_and_idx.value().first}));
  }

  // Move the converted functions to top level MLIR module.
  auto* block = module_->getBody();
  auto* sub_block = sub_module->getBody();
  block->getOperations().splice(
      mlir::Block::iterator(block->getTerminator()), sub_block->getOperations(),
      sub_block->begin(), mlir::Block::iterator(sub_block->getTerminator()));

  return Status::OK();
}

Status SavedModelSignatureDefImporter::LiftVariables() {
  VarGlobalMap var_globals;

  auto walker = [&var_globals](mlir::Operation* op) {
    if (auto var_handle_op = llvm::dyn_cast<VarHandleOp>(op))
      var_globals[var_handle_op.shared_name()].second.push_back(var_handle_op);
    else if (op->getName().getStringRef() == "tf.VariableV2")
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  };
  bool contains_ref_variable = module_->walk(walker).wasInterrupted();

  if (contains_ref_variable)
    return errors::InvalidArgument(
        "Ref variable created by VariableV2 is not supported.");

  if (var_globals.empty()) return Status::OK();

  TF_RETURN_IF_ERROR(ReadVariablesFromSession(&var_globals));

  for (const auto& it : var_globals)
    for (VarHandleOp var_handle : it.second.second)
      LiftVariable(var_handle, it.second.first);

  return Status::OK();
}

void SavedModelSignatureDefImporter::LiftVariable(
    VarHandleOp op, GlobalTensorOp global_tensor) {
  mlir::OpBuilder builder(&module_->getBodyRegion());

  auto func_op = op.getParentOfType<mlir::FuncOp>();
  builder.setInsertionPoint(func_op);

  auto func_type = func_op.getType();

  // Create the new function type by adding variable type to the arguments.
  llvm::SmallVector<mlir::Type, 4> new_input_types(
      func_type.getInputs().begin(), func_type.getInputs().end());
  mlir::Type resource_type = op.resource().getType();
  // Use the corresponding global tensor's type.
  auto type = global_tensor.type().cast<TensorType>();
  resource_type = mlir::RankedTensorType::get(
      {}, mlir::TF::ResourceType::get({type}, type.getContext()));

  new_input_types.push_back(resource_type);
  auto new_func_type =
      builder.getFunctionType(new_input_types, func_type.getResults());

  func_op.setType(new_func_type);

  // Bind the argument to the corresponding global tensor op.
  func_op.setArgAttr(func_op.getNumArguments() - 1,
                     "tf_saved_model.bound_input",
                     builder.getSymbolRefAttr(op.shared_name()));

  // Add the newly added function param to entry block's arguments.
  auto new_value = func_op.front().addArgument(resource_type);

  // Remove the VarHandleOp also updating the containing island's return type.
  DCHECK(llvm::isa<mlir::tf_executor::IslandOp>(op.getParentOp()));
  DCHECK(llvm::cast<mlir::tf_executor::IslandOp>(op.getParentOp())
             .WrapsSingleOp());
  op.getOperation()->replaceAllUsesWith(llvm::ArrayRef<mlir::Value>(new_value));
  op.getParentOp()->getResult(0).setType(resource_type);
  op.getOperation()->erase();
}

Status SavedModelSignatureDefImporter::ReadVariablesFromSession(
    VarGlobalMap* var_globals) {
  mlir::OpBuilder builder(&module_->getBodyRegion());

  // Read all resource variables from the session.
  std::vector<std::string> variable_names;
  variable_names.reserve(var_globals->size());
  for (const auto& name_and_location : *var_globals)
    variable_names.push_back(name_and_location.first.str());

  std::vector<Tensor> resource_tensors;
  TF_RETURN_IF_ERROR(bundle_.GetSession()->Run(
      /*inputs=*/{}, variable_names,
      /*target_node_names=*/{}, &resource_tensors));

  const DeviceMgr* device_manager;
  TF_RETURN_IF_ERROR(bundle_.GetSession()->LocalDeviceManager(&device_manager));

  // Read all underlying tensors of the variables from the session.
  std::vector<Tensor> tensors;
  tensors.reserve(resource_tensors.size());
  for (const auto& resource_tensor : resource_tensors) {
    const auto& resource_handle = resource_tensor.scalar<ResourceHandle>()();

    Device* device;
    TF_RETURN_IF_ERROR(
        device_manager->LookupDevice(resource_handle.device(), &device));

    Var* var_ptr;
    TF_RETURN_IF_ERROR(device->resource_manager()->Lookup(
        resource_handle.container(), resource_handle.name(), &var_ptr));
    core::RefCountPtr<Var> var(var_ptr);

    // The variable tensor is already loaded into corresponding device's
    // resource manager when we load the saved model using LoadSavedModel().
    // Here we just read its value.
    mutex_lock ml(*var->mu());
    tensors.push_back(*var->tensor());
  }

  for (const auto iter : llvm::zip(*var_globals, tensors)) {
    // Create global tensor op corresponding to the variable. Use the location
    // of the first use encountered.
    VarHandleOp op = std::get<0>(iter).second.second.front();
    const auto& name = std::get<0>(iter).first;
    const auto& tensor = std::get<1>(iter);

    // Create tensor attribute for this variable.
    TF_ASSIGN_OR_RETURN(auto tensor_attr, ConvertTensor(tensor, &builder));

    // Create the global tensor op with the tensor attribute.
    auto type = tensor_attr.getType().cast<TensorType>();
    auto global_tensor = builder.create<GlobalTensorOp>(
        op.getLoc(), builder.getStringAttr(name), tensor_attr,
        mlir::TypeAttr::get(type), builder.getUnitAttr());
    std::get<0>(iter).second.first = global_tensor;
  }

  return Status::OK();
}

GraphImportConfig::InputArrays SavedModelSignatureDefImporter::ParseInputArrays(
    const std::vector<std::pair<std::string, TensorInfo>>& inputs) {
  GraphImportConfig::InputArrays results;
  for (const auto& iter : inputs) {
    const auto& tensor_info = iter.second;

    // Only dense tensor is supported.
    DCHECK_EQ(tensor_info.encoding_case(), tensorflow::TensorInfo::kName);

    ArrayInfo array_info;
    array_info.imported_dtype = tensor_info.dtype();
    array_info.shape = tensor_info.tensor_shape();

    results.insert(std::pair<std::string, ArrayInfo>(tensor_info.name(),
                                                     std::move(array_info)));
  }
  return results;
}

}  // namespace

StatusOr<mlir::OwningModuleRef> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, bool add_default_attributes) {
  return SavedModelObjectGraphImporter::Convert(
      saved_model, context, exported_names, add_default_attributes);
}

StatusOr<mlir::OwningModuleRef> ConvertSavedModelV1ToMlir(
    const SavedModelBundle& saved_model, mlir::MLIRContext* context) {
  return SavedModelSignatureDefImporter::Convert(saved_model, context);
}

std::string MlirModuleToString(mlir::ModuleOp module, bool show_debug_info) {
  std::string txt_module;
  {
    mlir::OpPrintingFlags flags;
    if (show_debug_info) flags.enableDebugInfo();
    llvm::raw_string_ostream os{txt_module};
    module.print(os, flags);
  }
  return txt_module;
}

}  // namespace tensorflow
