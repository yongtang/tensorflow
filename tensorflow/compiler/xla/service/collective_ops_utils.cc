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

#include "tensorflow/compiler/xla/service/collective_ops_utils.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"

namespace xla {

absl::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation) {
  namespace m = match;
  const HloInstruction* root = computation->root_instruction();

  auto match_opcode = [&](HloOpcode opcode) {
    return Match(
        root, m::Op()
                  .WithOpcode(opcode)
                  .WithBinaryOperandsAnyOrder(m::Parameter(0), m::Parameter(1))
                  .WithShape(m::Shape().IsEffectiveScalar()));
  };

  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type = computation->root_instruction()->shape().element_type();
  if (match_opcode(HloOpcode::kAdd)) {
    return ReductionKind::SUM;
  } else if (match_opcode(HloOpcode::kMultiply)) {
    return ReductionKind::PRODUCT;
  } else if (match_opcode(HloOpcode::kMinimum) ||
             (type == PRED && match_opcode(HloOpcode::kAnd))) {
    return ReductionKind::MIN;
  } else if (match_opcode(HloOpcode::kMaximum) ||
             (type == PRED && match_opcode(HloOpcode::kOr))) {
    return ReductionKind::MAX;
  } else {
    return absl::nullopt;
  }
}

StatusOr<std::vector<int>> GetParticipatingIDs(
    int current_id, absl::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups) {
  // Empty replica_groups() means that all replicas participate.
  if (groups.empty()) {
    TF_RET_CHECK(total_participant_count.has_value());
    std::vector<int> all_participants(*total_participant_count);
    absl::c_iota(all_participants, 0);
    return all_participants;
  }

  // Figure out the other replicas that go together with this one.
  absl::optional<ReplicaGroup> group;
  for (const ReplicaGroup& g : groups) {
    if (absl::c_linear_search(g.replica_ids(), current_id)) {
      TF_RET_CHECK(!group.has_value())
          << "ID " << current_id << " appears twice in replica groups";
      group = g;
    }
  }
  TF_RET_CHECK(group.has_value())
      << "ID " << current_id << " doesn't appear in replica groups";
  return std::vector<int>(group->replica_ids().begin(),
                          group->replica_ids().end());
}

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, absl::optional<bool> use_global_device_ids) {
  if (!has_channel_id) {
    if (!use_global_device_ids.has_value() || !*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplica;
    } else {
      return InvalidArgument(
          "Invalid combination of has_channel_id and use_global_device_ids");
    }
  } else {
    if (!use_global_device_ids.has_value()) {
      return CollectiveOpGroupMode::kCrossPartition;
    } else if (!*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplicaAndPartition;
    } else {
      return CollectiveOpGroupMode::kFlattenedID;
    }
  }
}

StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  int replica_count = device_assignment.replica_count();
  int partition_count = device_assignment.computation_count();

  std::pair<int, int> logical_ids;
  TF_ASSIGN_OR_RETURN(logical_ids,
                      device_assignment.LogicalIdsForDevice(device_id));
  int current_replica_id = logical_ids.first;
  int current_partition_id = logical_ids.second;

  std::vector<GlobalDeviceId> participants;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      // This is a cross replica operation. replica group contains replica id.
      // use current replica id to find the set of participating replicas. If
      // replica groups are empty, assume a group with all replicas.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(current_replica_id, replica_count,
                                              replica_groups));

      // The set of participating devices is the replicas from the current
      // partition.
      participants.reserve(participating_replicas.size());
      for (int replica_id : participating_replicas) {
        participants.emplace_back(
            device_assignment(replica_id, current_partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossPartition: {
      // replica_groups contain partition_id, group contains all partitions for
      // the current replica.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_partitions,
                          GetParticipatingIDs(current_partition_id,
                                              partition_count, replica_groups));
      participants.reserve(participating_partitions.size());
      for (int partition_id : participating_partitions) {
        participants.emplace_back(
            device_assignment(current_replica_id, partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      // replica_groups contain replica_ids. Group contains replicas for all
      // partitions.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(current_replica_id, replica_count,
                                              replica_groups));
      participants.reserve(participating_replicas.size() * partition_count);
      for (int replica_id : participating_replicas) {
        for (int partition_id = 0; partition_id < partition_count;
             ++partition_id) {
          participants.emplace_back(
              device_assignment(replica_id, partition_id));
        }
      }
      return participants;
    }

    case CollectiveOpGroupMode::kFlattenedID: {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";

      int current_flattened_id =
          current_replica_id * partition_count + current_partition_id;

      // Find participants based on flattened id. replica_groups cannot be empty
      // so no need to pass in total_participant_count.
      TF_ASSIGN_OR_RETURN(
          std::vector<int> participating_flattened_ids,
          GetParticipatingIDs(current_flattened_id,
                              /*total_participant_count=*/absl::nullopt,
                              replica_groups));

      participants.reserve(participating_flattened_ids.size());
      for (int flattened_id : participating_flattened_ids) {
        // Map from flattened id back to replica_id, partition_id.
        int replica_id = flattened_id / partition_count;
        int partition_id = flattened_id % partition_count;
        participants.emplace_back(device_assignment(replica_id, partition_id));
      }
      return participants;
    }
  }
}

}  // end namespace xla
