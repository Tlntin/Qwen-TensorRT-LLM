/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace inflight_batcher
{

namespace batch_manager
{

/// BatchSlotManager
///
/// Helper class to manage batch slots for a custom backend
/// This class is not thread-safe

class BatchSlotManager
{
public:
    BatchSlotManager(uint32_t max_batch_size, uint64_t max_sequence_idle_microseconds);

    /// Function that returns a vector of batch slots for the provided corr_ids
    /// For a new correlation id, a new batch slot will be allocated
    /// In case no batch slot could be allocated or matched, batch slot is set to -1
    std::vector<int32_t> getBatchSlots(
        const std::vector<int32_t>& batch_start_flags, const std::vector<uint64_t>& batch_corr_ids);

    /// Function that frees the batch slot associated with the given correlation id
    void freeBatchSlot(uint64_t corr_id, bool erase_corr_id = true);

    /// Function that frees the batch slots associated with the provided corr_ids
    void freeBatchSlots(const std::vector<int32_t>& batch_end_flags, const std::vector<uint64_t>& batch_corr_ids);

    /// Function that frees batch slots that have been idle for more than
    /// max_sequence_idle_microseconds
    std::tuple<std::vector<int>, std::vector<uint64_t>> freeIdleBatchSlots();

private:
    uint32_t max_batch_size_;
    uint64_t max_sequence_idle_microseconds_;

    std::multimap<uint64_t, int32_t> corr_id_to_batch_slot_;
    std::deque<int32_t> available_batch_slots_;
    std::vector<std::chrono::steady_clock::time_point> last_timepoint_;
};

} // namespace batch_manager
} // namespace inflight_batcher
