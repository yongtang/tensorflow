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

#include "tensorflow/core/grappler/optimizers/data/unbatch_and_batch_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kFusedOpName[] = "ExperimentalUnbatchAndBatchDataset";

NodeDef MakeUnbatchAndBatchNode(const NodeDef& unbatch_node, const NodeDef& batch_node,
                            MutableGraphView* graph) {
  NodeDef new_node;
  new_node.set_op(kFusedOpName);
  graph_utils::SetUniqueGraphNodeName(kFusedOpName, graph->graph(), &new_node);

  // Set the `input` input argument.
  new_node.add_input(unbatch_node.input(0));

  // Set the `batch_size` input argument.
  new_node.add_input(batch_node.input(1));

  // Set the `drop_remainder` input argument.
  if (batch_node.op() == "BatchDatasetV2") {
    new_node.add_input(batch_node.input(2));
  } else {
    NodeDef* tmp = graph_utils::AddScalarConstNode<bool>(false, graph);
    new_node.add_input(tmp->name());
  }

  // Required attributes.
  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, batch_node, &new_node);
  }

  return new_node;
}

}  // namespace

Status UnbatchAndBatchFusion::OptimizeAndCollectStats(Cluster* cluster,
                                                  const GrapplerItem& item,
                                                  GraphDef* output,
                                                  OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "BatchDataset" && node.op() != "BatchDatasetV2") {
      continue;
    }

    // Use a more descriptive variable name now that we know the node type.
    const NodeDef& batch_node = node;
    NodeDef* node2 = graph_utils::GetInputNode(batch_node, graph);

    if (node2->op() != "UnbatchDataset") {
      continue;
    }
    // Use a more descriptive variable name now that we know the node type.
    NodeDef* unbatch_node = node2;

    auto* new_node =
        graph.AddNode(MakeUnbatchAndBatchNode(*unbatch_node, batch_node, &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(batch_node.name(), new_node->name()));

    // Mark the `Unbatch` and `Batch` nodes for removal.
    nodes_to_delete.insert(unbatch_node->name());
    nodes_to_delete.insert(batch_node.name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

void UnbatchAndBatchFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimize_output,
                                 double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(UnbatchAndBatchFusion, "unbatch_and_batch_fusion");

}  // namespace grappler
}  // namespace tensorflow
