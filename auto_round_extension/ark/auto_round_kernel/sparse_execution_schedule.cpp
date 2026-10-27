#include "sparse_execution_schedule.hpp"

#if defined(ARK_XPU)

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>

namespace ark {
namespace {

constexpr int kScheduleWorkgroupSize = 16;
constexpr int kControlBestRound = 0;
constexpr int kControlRounds = 1;
constexpr int kControlOverflow = 2;
constexpr int kControlActiveCount = 3;

int popcount16(std::uint16_t value) {
  int count = 0;
  while (value != 0) {
    count += value & 1;
    value >>= 1;
  }
  return count;
}

int first_set_bit(const std::uint64_t* words, int word_count) {
  for (int word = 0; word < word_count; ++word) {
    std::uint64_t value = words[word];
    if (value == 0) {
      continue;
    }
    int bit = 0;
    while ((value & 1) == 0) {
      value >>= 1;
      ++bit;
    }
    return word * 64 + bit;
  }
  return -1;
}

std::uint64_t candidate_key(int overlap, int free_count, int round, int rows_per_group) {
  return (std::uint64_t{static_cast<unsigned>(overlap)} << 32) |
         (std::uint64_t{static_cast<unsigned>(rows_per_group + 1 - free_count)} << 16) |
         std::uint64_t{static_cast<unsigned>(0xffff - round)};
}

}  // namespace

void build_sparse_execution_schedule(std::uintptr_t stream, std::uintptr_t lut, std::uintptr_t valid_block_num,
                                     std::uintptr_t execution_schedule, std::uintptr_t execution_schedule_rounds,
                                     int batch, int num_heads, int num_q_blocks, int num_k_blocks, int rows_per_group,
                                     int max_rounds) {
  if (stream == 0 || lut == 0 || valid_block_num == 0 || execution_schedule == 0 ||
      execution_schedule_rounds == 0) {
    throw std::invalid_argument("build_sparse_execution_schedule: device pointers must be non-null");
  }
  if (batch <= 0 || num_heads <= 0 || num_q_blocks <= 0 || num_k_blocks <= 0 || rows_per_group < 1 ||
      rows_per_group > 8 || max_rounds < num_k_blocks) {
    throw std::invalid_argument("build_sparse_execution_schedule: invalid problem shape");
  }

  auto* queue = reinterpret_cast<sycl::queue*>(stream);
  auto const* lut_ptr = reinterpret_cast<const int*>(lut);
  auto const* valid_ptr = reinterpret_cast<const int*>(valid_block_num);
  auto* schedule_ptr = reinterpret_cast<int*>(execution_schedule);
  auto* rounds_ptr = reinterpret_cast<int*>(execution_schedule_rounds);

  int q_groups = (num_q_blocks + rows_per_group - 1) / rows_per_group;
  int num_groups = batch * num_heads * q_groups;
  int mask_count = 1 << rows_per_group;
  int bucket_words = (max_rounds + 63) / 64;

  queue->submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint16_t, 1> block_masks(num_k_blocks, handler);
    sycl::local_accessor<std::uint16_t, 1> free_masks(max_rounds, handler);
    sycl::local_accessor<std::uint64_t, 1> buckets(mask_count * bucket_words, handler);
    sycl::local_accessor<int, 1> active_masks(mask_count, handler);
    sycl::local_accessor<int, 1> active_positions(mask_count, handler);
    sycl::local_accessor<int, 1> round_metrics(max_rounds, handler);
    sycl::local_accessor<int, 1> round_order(max_rounds, handler);
    sycl::local_accessor<int, 1> row_copy(max_rounds, handler);
    sycl::local_accessor<int, 1> metric_counts(num_k_blocks + 1, handler);
    sycl::local_accessor<int, 1> control(4, handler);
    sycl::local_accessor<std::uint64_t, 1> lane_keys(kScheduleWorkgroupSize, handler);

    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_groups * kScheduleWorkgroupSize),
                          sycl::range<1>(kScheduleWorkgroupSize)),
        [=](sycl::nd_item<1> item) {
          int lane = item.get_local_id(0);
          int group = item.get_global_id(0) / kScheduleWorkgroupSize;
          int batch_head = group / q_groups;
          int q_group = group - batch_head * q_groups;
          int batch_index = batch_head / num_heads;
          int head = batch_head - batch_index * num_heads;
          int q_block_start = q_group * rows_per_group;
          int rows_left = num_q_blocks - q_block_start;
          int active_rows = rows_left < rows_per_group ? rows_left : rows_per_group;
          std::uint16_t full_mask = static_cast<std::uint16_t>((1 << rows_per_group) - 1);
          int schedule_base = group * rows_per_group * max_rounds;

          for (int index = lane; index < rows_per_group * max_rounds; index += kScheduleWorkgroupSize) {
            schedule_ptr[schedule_base + index] = -1;
          }
          for (int block = lane; block < num_k_blocks; block += kScheduleWorkgroupSize) {
            block_masks[block] = 0;
          }
          sycl::group_barrier(item.get_group());

          if (lane == 0) {
            for (int row = 0; row < active_rows; ++row) {
              int row_index = (batch_head * num_q_blocks) + q_block_start + row;
              int count = valid_ptr[row_index];
              int block = 0;
              const int* row_lut = lut_ptr + static_cast<std::size_t>(row_index) * num_k_blocks;
              for (int index = 0; index < count; ++index) {
                block += row_lut[index];
                block_masks[block] |= static_cast<std::uint16_t>(1u << row);
              }
            }
          }
          sycl::group_barrier(item.get_group());

          int rounds = 0;
          for (int row = 0; row < active_rows; ++row) {
            int row_index = (batch_head * num_q_blocks) + q_block_start + row;
            int count = valid_ptr[row_index];
            rounds = count > rounds ? count : rounds;
          }
          if (rounds == 0) {
            if (lane == 0) {
              rounds_ptr[group] = 0;
            }
            return;
          }

          if (lane == 0) {
            control[kControlBestRound] = -1;
            control[kControlRounds] = rounds;
            control[kControlOverflow] = 0;
            control[kControlActiveCount] = 0;
            for (int round = 0; round < rounds; ++round) {
              free_masks[round] = full_mask;
            }
            for (int bucket = 0; bucket < mask_count; ++bucket) {
              active_positions[bucket] = -1;
              for (int word = 0; word < bucket_words; ++word) {
                buckets[bucket * bucket_words + word] = 0;
              }
            }
            for (int round = 0; round < rounds; ++round) {
              buckets[full_mask * bucket_words + round / 64] |= std::uint64_t{1} << (round % 64);
            }
            active_positions[full_mask] = 0;
            active_masks[0] = full_mask;
            control[kControlActiveCount] = 1;
          }
          sycl::group_barrier(item.get_group());

          for (int window = 0; window < num_k_blocks; window += 256) {
            int window_end = window + 256;
            if (window_end > num_k_blocks) {
              window_end = num_k_blocks;
            }
            for (int selected_count = rows_per_group; selected_count >= 1; --selected_count) {
              for (int block = window; block < window_end; ++block) {
                int remaining = block_masks[block];
                if (popcount16(static_cast<std::uint16_t>(remaining)) != selected_count) {
                  continue;
                }
                while (remaining != 0) {
                  std::uint64_t local_key = 0;
                  int active_count = control[kControlActiveCount];
                  for (int active = lane; active < active_count; active += kScheduleWorkgroupSize) {
                    int free_mask = active_masks[active];
                    int round = first_set_bit(&buckets[free_mask * bucket_words], bucket_words);
                    if (round < 0) {
                      continue;
                    }
                    int overlap = popcount16(static_cast<std::uint16_t>(remaining & free_mask));
                    if (overlap == 0) {
                      continue;
                    }
                    int free_count = popcount16(static_cast<std::uint16_t>(free_mask));
                    std::uint64_t key = candidate_key(overlap, free_count, round, rows_per_group);
                    local_key = key > local_key ? key : local_key;
                  }
                  lane_keys[lane] = local_key;
                  sycl::group_barrier(item.get_group());
                  std::uint64_t best_key = 0;
                  if (lane == 0) {
                    for (int index = 0; index < kScheduleWorkgroupSize; ++index) {
                      best_key = lane_keys[index] > best_key ? lane_keys[index] : best_key;
                    }
                    int best_round = best_key == 0 ? -1 : 0xffff - static_cast<int>(best_key & 0xffff);
                    if (best_round < 0) {
                      best_round = control[kControlRounds];
                      if (best_round >= max_rounds) {
                        control[kControlOverflow] = 1;
                        best_round = -1;
                      } else {
                        control[kControlRounds] = best_round + 1;
                        free_masks[best_round] = full_mask;
                        bool was_empty = true;
                        for (int word = 0; word < bucket_words; ++word) {
                          was_empty &= buckets[full_mask * bucket_words + word] == 0;
                        }
                        buckets[full_mask * bucket_words + best_round / 64] |=
                            std::uint64_t{1} << (best_round % 64);
                        if (was_empty) {
                          int active_count = control[kControlActiveCount];
                          active_positions[full_mask] = active_count;
                          active_masks[active_count] = full_mask;
                          control[kControlActiveCount] = active_count + 1;
                        }
                      }
                    }
                    control[kControlBestRound] = best_round;
                  }
                  sycl::group_barrier(item.get_group());
                  if (control[kControlOverflow] != 0) {
                    if (lane == 0) {
                      rounds_ptr[group] = -1;
                    }
                    return;
                  }

                  int best_round = control[kControlBestRound];
                  int old_free = free_masks[best_round];
                  int piece = remaining & old_free;
                  if (lane == 0) {
                    int word = best_round / 64;
                    std::uint64_t bit = std::uint64_t{1} << (best_round % 64);
                    buckets[old_free * bucket_words + word] &= ~bit;
                    bool old_bucket_empty = true;
                    for (int bucket_word = 0; bucket_word < bucket_words; ++bucket_word) {
                      old_bucket_empty &= buckets[old_free * bucket_words + bucket_word] == 0;
                    }
                    if (old_bucket_empty) {
                      int position = active_positions[old_free];
                      int last = control[kControlActiveCount] - 1;
                      int moved = active_masks[last];
                      active_masks[position] = moved;
                      active_positions[moved] = position;
                      active_positions[old_free] = -1;
                      control[kControlActiveCount] = last;
                    }
                    int new_free = old_free & ~piece;
                    free_masks[best_round] = static_cast<std::uint16_t>(new_free);
                    if (new_free != 0) {
                      bool new_bucket_empty = true;
                      for (int bucket_word = 0; bucket_word < bucket_words; ++bucket_word) {
                        new_bucket_empty &= buckets[new_free * bucket_words + bucket_word] == 0;
                      }
                      buckets[new_free * bucket_words + word] |= bit;
                      if (new_bucket_empty) {
                        int active_count = control[kControlActiveCount];
                        active_positions[new_free] = active_count;
                        active_masks[active_count] = new_free;
                        control[kControlActiveCount] = active_count + 1;
                      }
                    }
                  }
                  if (lane < rows_per_group && (piece & (1 << lane)) != 0) {
                    schedule_ptr[schedule_base + lane * max_rounds + best_round] = block;
                  }
                  sycl::group_barrier(item.get_group());
                  remaining &= ~piece;
                }
              }
            }
          }

          sycl::group_barrier(item.get_group());
          if (lane == 0) {
            int round_count = control[kControlRounds];
            for (int round = 0; round < round_count; ++round) {
              int metric = -1;
              for (int row = 0; row < rows_per_group; ++row) {
                int block = schedule_ptr[schedule_base + row * max_rounds + round];
                metric = block > metric ? block : metric;
              }
              round_metrics[round] = metric;
            }
            for (int index = 0; index <= num_k_blocks; ++index) {
              metric_counts[index] = 0;
            }
            for (int round = 0; round < round_count; ++round) {
              ++metric_counts[round_metrics[round] + 1];
            }
            int metric_offset = 0;
            for (int index = 0; index <= num_k_blocks; ++index) {
              int count = metric_counts[index];
              metric_counts[index] = metric_offset;
              metric_offset += count;
            }
            for (int round = 0; round < round_count; ++round) {
              int metric_index = round_metrics[round] + 1;
              round_order[metric_counts[metric_index]++] = round;
            }
            for (int row = 0; row < rows_per_group; ++row) {
              int row_base = schedule_base + row * max_rounds;
              for (int round = 0; round < round_count; ++round) {
                row_copy[round] = schedule_ptr[row_base + round];
              }
              for (int round = 0; round < round_count; ++round) {
                schedule_ptr[row_base + round] = row_copy[round_order[round]];
              }
            }
            rounds_ptr[group] = round_count;
          }
        });
  });
}

}  // namespace ark

#endif  // ARK_XPU
