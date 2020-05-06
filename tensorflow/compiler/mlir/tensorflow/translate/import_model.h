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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_

#include "tensorflow/compiler/mlir/tensorflow/translate/import_graphdef.h"
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/loader.h"

namespace tensorflow {

// Given a SavedModel, returns a MLIR module containing the functions, expressed
// with tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningModuleRef> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, bool add_default_attributes = true);

// Given a V1 SavedModel, returns a MLIR module containing the functions,
// expressed with tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningModuleRef>
ConvertSavedModelV1ToMlir(const SavedModelBundle& saved_model,
                          mlir::MLIRContext* context);

// Serialize a MLIR module to a string.
std::string MlirModuleToString(mlir::ModuleOp m, bool show_debug_info = false);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
