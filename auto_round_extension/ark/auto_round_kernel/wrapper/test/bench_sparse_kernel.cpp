#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "../include/xpu_wrapper.hpp"
#include "../include/sycl_tla_common.hpp"

namespace {

#if defined(ARK_SPARSE_DEV_SPARSE_ONLY)
constexpr bool kSparseDevSparseOnly = true;
#else
constexpr bool kSparseDevSparseOnly = false;
#endif

constexpr int kSparseProfileModeFull = 0;
constexpr int kSparseProfileModePvMmaOnly = 9;

enum class ValueDType {
  F16,
  BF16,
};

struct PresetConfig {
  std::string name;
  int batch;
  int num_heads_q;
  int num_heads_kv;
  int seq_len_q;
  int seq_len_kv;
  int head_dim;
  int scale_block_size;
  ValueDType value_dtype;
};

struct CliConfig {
  std::string preset = "flux_joint";
  std::string pattern = "prefix";
  std::string sparse_mode = "sparse";
  std::string profile_mode = "full";
  std::string tensor_layout = "HND";
  std::vector<double> topks{1.0, 0.75, 0.5, 0.25, 0.125};
  int warmup = 2;
  int iters = 5;
  bool half_output_tile = false;
  std::optional<std::string> csv_path;
  std::optional<ValueDType> value_dtype;
  std::optional<int> batch;
  std::optional<int> num_heads_q;
  std::optional<int> num_heads_kv;
  std::optional<int> seq_len_q;
  std::optional<int> seq_len_kv;
  std::optional<int> head_dim;
  std::optional<int> scale_block_size;
  std::optional<int> q_tile_override;
};

struct ResultRow {
  std::string preset;
  std::string mode;
  std::string pattern;
  std::string dtype;
  double requested_topk = 0.0;
  double selected_ratio = 0.0;
  double blocks_per_row = 0.0;
  double latency_ms = 0.0;
  double dense_tflops = 0.0;
  double effective_tflops = 0.0;
};

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) parts.push_back(item);
  }
  return parts;
}

std::vector<double> parse_topks(const std::string& text) {
  std::vector<double> values;
  for (const auto& item : split_csv(text)) {
    values.push_back(std::stod(item));
  }
  if (values.empty()) {
    throw std::invalid_argument("topk list must not be empty");
  }
  return values;
}

PresetConfig get_preset(const std::string& name) {
  if (name == "wan_self") {
    // Representative Wan2.1 self-attention shape from the current default run regime:
    // 480x832, 81 frames, latent downsampled and packed with patch_size [1,2,2].
    return {"wan_self", 1, 12, 12, 32760, 32760, 128, 64, ValueDType::F16};
  }
  if (name == "flux_joint") {
    // Representative FLUX joint attention shape for 1024x1024 with max_sequence_length=512:
    // image tokens 16384 + text tokens 512.
    return {"flux_joint", 1, 24, 24, 16896, 16896, 128, 64, ValueDType::F16};
  }
  if (name == "flux_single") {
    // Representative FLUX single-stream image-only attention shape for 1024x1024.
    return {"flux_single", 1, 24, 24, 16384, 16384, 128, 64, ValueDType::F16};
  }
  throw std::invalid_argument("unknown preset: " + name);
}

template <typename T>
std::vector<T> to_dtype_vector(const std::vector<float>& src) {
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = T(src[i]);
  }
  return dst;
}

std::vector<float> make_random_vector(size_t count, float low, float high, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> values(count);
  for (auto& value : values) value = dist(rng);
  return values;
}

template <typename T>
std::vector<T> hnd_to_nhd(const std::vector<T>& src, int batch, int heads, int seq, int head_dim) {
  std::vector<T> dst(src.size());
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int s = 0; s < seq; ++s) {
        for (int d = 0; d < head_dim; ++d) {
          size_t src_idx = (((size_t(b) * heads + size_t(h)) * seq + size_t(s)) * head_dim) + size_t(d);
          size_t dst_idx = (((size_t(b) * seq + size_t(s)) * heads + size_t(h)) * head_dim) + size_t(d);
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
  return dst;
}

double compute_dense_flops(int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim) {
  int group = num_heads_q / num_heads_kv;
  double qk = double(group) * num_heads_kv * seq_len_q * head_dim * seq_len_kv * 2.0;
  double softmax = double(group) * num_heads_kv * seq_len_q * seq_len_kv * 2.0;
  double pv = double(group) * num_heads_kv * seq_len_q * seq_len_kv * head_dim * 2.0;
  return double(batch) * (qk + softmax + pv);
}

double flops_to_tflops(double flops, double latency_ms) {
  return flops / (latency_ms * 1.0e-3) / 1.0e12;
}

double run_bench(const std::function<void()>& fn, sycl::queue* q, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  q->wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  q->wait();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / double(iters);
}

std::vector<int> build_pattern_indices(const std::string& pattern, int kv_blocks, int num_select) {
  std::vector<int> selected;
  num_select = std::max(1, std::min(num_select, kv_blocks));

  if (pattern == "all_selected") {
    selected.resize(kv_blocks);
    for (int i = 0; i < kv_blocks; ++i) selected[i] = i;
    return selected;
  }

  if (pattern == "prefix") {
    selected.resize(num_select);
    for (int i = 0; i < num_select; ++i) selected[i] = i;
    return selected;
  }

  if (pattern == "stride2") {
    for (int i = 0; i < kv_blocks && int(selected.size()) < num_select; i += 2) selected.push_back(i);
    for (int i = 1; i < kv_blocks && int(selected.size()) < num_select; i += 2) selected.push_back(i);
    return selected;
  }

  if (pattern == "custom_02") return std::vector<int>{0, 2};
  if (pattern == "custom_035") return std::vector<int>{0, 3, 5};
  if (pattern == "custom_135") return std::vector<int>{1, 3, 5};

  throw std::invalid_argument("unknown sparse pattern: " + pattern);
}

void build_sparse_lut(int batch, int num_heads_q, int q_blocks, int kv_blocks, const std::vector<int>& selection,
                      std::vector<int>& lut, std::vector<int>& valid_block_num) {
  lut.assign(size_t(batch) * num_heads_q * q_blocks * kv_blocks, 0);
  valid_block_num.assign(size_t(batch) * num_heads_q * q_blocks, int(selection.size()));

  for (int value : selection) {
    if (value < 0 || value >= kv_blocks) {
      throw std::invalid_argument("sparse pattern contains out-of-range KV block");
    }
  }

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_q; ++h) {
      for (int qblk = 0; qblk < q_blocks; ++qblk) {
        size_t row_offset = (size_t(b) * num_heads_q * q_blocks + size_t(h) * q_blocks + size_t(qblk)) * kv_blocks;
        int prev = 0;
        bool first = true;
        for (size_t i = 0; i < selection.size(); ++i) {
          int selected = selection[i];
          lut[row_offset + i] = first ? selected : (selected - prev);
          prev = selected;
          first = false;
        }
      }
    }
  }
}

std::string dtype_to_string(ValueDType dtype) {
  return dtype == ValueDType::BF16 ? "bf16" : "f16";
}

ValueDType parse_value_dtype(const std::string& text) {
  if (text == "f16") return ValueDType::F16;
  if (text == "bf16") return ValueDType::BF16;
  throw std::invalid_argument("unsupported dtype: " + text + " (expected f16 or bf16)");
}

void print_usage() {
  std::cout
      << "Usage: bench_ARK_XPU [--preset wan_self|flux_joint|flux_single] [--pattern prefix|all_selected|stride2|custom_02|custom_035|custom_135]\n"
      << "                    [--sparse-mode sparse|sparse_row_linear|sparse_row_linear_halfwidth|sparse_row_linear_profile]\n"
      << "                    [--tensor-layout HND|NHD]\n"
      << "                    [--profile-mode full|pv_mma_only] [--half-output-tile] [--q-tile 0|64|128|256]\n"
      << "                    [--topk 1.0,0.75,0.5] [--dtype f16|bf16] [--warmup N] [--iters N] [--csv path]\n"
      << "                    [--batch N] [--heads-q N] [--heads-kv N] [--seq-q N] [--seq-kv N] [--head-dim N] [--block-size N]\n";
}

CliConfig parse_args(int argc, char** argv) {
  CliConfig config;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[++i];
    };

    if (arg == "--preset") config.preset = need_value("--preset");
    else if (arg == "--pattern") config.pattern = need_value("--pattern");
    else if (arg == "--sparse-mode") config.sparse_mode = need_value("--sparse-mode");
    else if (arg == "--tensor-layout") config.tensor_layout = need_value("--tensor-layout");
    else if (arg == "--profile-mode") config.profile_mode = need_value("--profile-mode");
    else if (arg == "--topk") config.topks = parse_topks(need_value("--topk"));
    else if (arg == "--dtype") config.value_dtype = parse_value_dtype(need_value("--dtype"));
    else if (arg == "--warmup") config.warmup = std::stoi(need_value("--warmup"));
    else if (arg == "--iters") config.iters = std::stoi(need_value("--iters"));
    else if (arg == "--half-output-tile") config.half_output_tile = true;
    else if (arg == "--q-tile") config.q_tile_override = std::stoi(need_value("--q-tile"));
    else if (arg == "--csv") config.csv_path = need_value("--csv");
    else if (arg == "--batch") config.batch = std::stoi(need_value("--batch"));
    else if (arg == "--heads-q") config.num_heads_q = std::stoi(need_value("--heads-q"));
    else if (arg == "--heads-kv") config.num_heads_kv = std::stoi(need_value("--heads-kv"));
    else if (arg == "--seq-q") config.seq_len_q = std::stoi(need_value("--seq-q"));
    else if (arg == "--seq-kv") config.seq_len_kv = std::stoi(need_value("--seq-kv"));
    else if (arg == "--head-dim") config.head_dim = std::stoi(need_value("--head-dim"));
    else if (arg == "--block-size") config.scale_block_size = std::stoi(need_value("--block-size"));
    else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown arg: " + arg);
    }
  }
  return config;
}

int parse_sparse_profile_mode(const std::string& text) {
  if (text == "full") return kSparseProfileModeFull;
  if (text == "pv_mma_only") return kSparseProfileModePvMmaOnly;
  throw std::invalid_argument("unsupported profile mode: " + text + " (expected full or pv_mma_only)");
}

int effective_q_tile_override(const CliConfig& cli) {
  if (cli.q_tile_override) return *cli.q_tile_override;
  return cli.sparse_mode == "sparse" ? 0 : 64;
}

std::string sparse_mode_label(const CliConfig& cli) {
  if (cli.sparse_mode == "sparse") return "sparse_kernel_only";
  if (cli.sparse_mode == "sparse_row_linear") return "sparse_row_linear_kernel_only";
  if (cli.sparse_mode == "sparse_row_linear_halfwidth") return "sparse_row_linear_halfwidth_kernel_only";
  if (cli.sparse_mode == "sparse_row_linear_profile") {
    return cli.half_output_tile ? "sparse_row_linear_profile_" + cli.profile_mode + "_halfwidth"
                                : "sparse_row_linear_profile_" + cli.profile_mode;
  }
  throw std::invalid_argument("unsupported sparse mode: " + cli.sparse_mode);
}

PresetConfig apply_overrides(PresetConfig preset, const CliConfig& cli) {
  if (cli.value_dtype) preset.value_dtype = *cli.value_dtype;
  if (cli.batch) preset.batch = *cli.batch;
  if (cli.num_heads_q) preset.num_heads_q = *cli.num_heads_q;
  if (cli.num_heads_kv) preset.num_heads_kv = *cli.num_heads_kv;
  if (cli.seq_len_q) preset.seq_len_q = *cli.seq_len_q;
  if (cli.seq_len_kv) preset.seq_len_kv = *cli.seq_len_kv;
  if (cli.head_dim) preset.head_dim = *cli.head_dim;
  if (cli.scale_block_size) preset.scale_block_size = *cli.scale_block_size;
  return preset;
}

void write_csv(const std::string& path, const std::vector<ResultRow>& rows) {
  std::ofstream out(path);
  out << "preset,mode,pattern,dtype,requested_topk,selected_ratio,blocks_per_row,latency_ms,dense_tflops,effective_tflops\n";
  for (const auto& row : rows) {
    out << row.preset << ',' << row.mode << ',' << row.pattern << ',' << row.dtype << ','
        << std::fixed << std::setprecision(6) << row.requested_topk << ','
        << row.selected_ratio << ',' << row.blocks_per_row << ','
        << row.latency_ms << ',' << row.dense_tflops << ',' << row.effective_tflops << '\n';
  }
}

void print_rows(const std::vector<ResultRow>& rows) {
  std::cout << "| preset | mode | pattern | dtype | topk | selected_ratio | blocks/row | latency (ms) | dense_tflops | effective_tflops |\n";
  std::cout << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
  for (const auto& row : rows) {
    std::cout << "| " << row.preset << " | " << row.mode << " | " << row.pattern << " | " << row.dtype << " | "
              << std::fixed << std::setprecision(3) << row.requested_topk << " | "
              << std::setprecision(6) << row.selected_ratio << " | "
              << std::setprecision(3) << row.blocks_per_row << " | "
              << std::setprecision(3) << row.latency_ms << " | "
              << std::setprecision(3) << row.dense_tflops << " | "
              << std::setprecision(3) << row.effective_tflops << " |\n";
  }
}

template <typename T>
std::vector<ResultRow> run_benchmark_typed(const PresetConfig& preset, const CliConfig& cli) {
  GETQ();
  if (!q) throw std::runtime_error("XPU queue is not available");
  if (preset.num_heads_q % preset.num_heads_kv != 0) {
    throw std::invalid_argument("num_heads_q must be divisible by num_heads_kv");
  }
  if (preset.head_dim != 128 && preset.head_dim != 64) {
    throw std::invalid_argument("benchmark currently supports head_dim 64 or 128 only");
  }

  const int batch = preset.batch;
  const int num_heads_q = preset.num_heads_q;
  const int num_heads_kv = preset.num_heads_kv;
  const int seq_len_q = preset.seq_len_q;
  const int seq_len_kv = preset.seq_len_kv;
  const int head_dim = preset.head_dim;
  const int block_size = preset.scale_block_size;
  const int q_tile_override = effective_q_tile_override(cli);
  const int sparse_profile_mode = parse_sparse_profile_mode(cli.profile_mode);
  const int q_blocks = (seq_len_q + block_size - 1) / block_size;
  const int kv_blocks = (seq_len_kv + block_size - 1) / block_size;
  const float softmax_scale = 1.0f / std::sqrt(float(head_dim));
  const BTLA_DTYPE pv_dtype = preset.value_dtype == ValueDType::BF16 ? BTLA_DTYPE::BF16 : BTLA_DTYPE::F16;

  if (cli.sparse_mode == "sparse_row_linear_profile" && !cli.half_output_tile &&
      sparse_profile_mode != kSparseProfileModePvMmaOnly) {
    throw std::invalid_argument(
        "sparse_row_linear_profile without --half-output-tile currently supports only --profile-mode pv_mma_only");
  }

  size_t q_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
  size_t kv_count = size_t(batch) * num_heads_kv * seq_len_kv * head_dim;
  size_t o_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
  size_t q_scale_count = size_t(batch) * num_heads_q * q_blocks;
  size_t k_scale_count = size_t(batch) * num_heads_kv * kv_blocks;

  auto host_q = to_dtype_vector<T>(make_random_vector(q_count, -1.0f, 1.0f, 101));
  auto host_k = to_dtype_vector<T>(make_random_vector(kv_count, -1.0f, 1.0f, 202));
  auto host_v = to_dtype_vector<T>(make_random_vector(kv_count, -1.0f, 1.0f, 303));
  if (cli.tensor_layout == "NHD") {
    host_q = hnd_to_nhd(host_q, batch, num_heads_q, seq_len_q, head_dim);
    host_k = hnd_to_nhd(host_k, batch, num_heads_kv, seq_len_kv, head_dim);
    host_v = hnd_to_nhd(host_v, batch, num_heads_kv, seq_len_kv, head_dim);
  } else if (cli.tensor_layout != "HND") {
    throw std::invalid_argument("unsupported tensor layout: " + cli.tensor_layout);
  }

  auto* dev_q = reinterpret_cast<T*>(ctx->allocate(q_count * sizeof(T)));
  auto* dev_k = reinterpret_cast<T*>(ctx->allocate(kv_count * sizeof(T)));
  auto* dev_v = reinterpret_cast<T*>(ctx->allocate(kv_count * sizeof(T)));
  auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
  auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(kv_count * sizeof(int8_t)));
  auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
  auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
  T* dev_dense_out = nullptr;
  if (!kSparseDevSparseOnly) {
    dev_dense_out = reinterpret_cast<T*>(ctx->allocate(o_count * sizeof(T)));
  }
  auto* dev_sparse_out = reinterpret_cast<T*>(ctx->allocate(o_count * sizeof(T)));

  int q_in_stride_s;
  int q_in_stride_d = 1;
  int q_in_stride_h;
  int q_in_stride_b;
  int k_in_stride_s;
  int k_in_stride_d = 1;
  int k_in_stride_h;
  int k_in_stride_b;
  int v_stride_d = 1;
  int v_stride_s;
  int v_stride_h;
  int v_stride_b;
  int q_stride_s = head_dim;
  int q_stride_d = 1;
  int q_stride_h = seq_len_q * head_dim;
  int q_stride_b = num_heads_q * seq_len_q * head_dim;
  int k_stride_s = head_dim;
  int k_stride_d = 1;
  int k_stride_h = seq_len_kv * head_dim;
  int k_stride_b = num_heads_kv * seq_len_kv * head_dim;
  int o_stride_s;
  int o_stride_d = 1;
  int o_stride_h;
  int o_stride_b;
  if (cli.tensor_layout == "NHD") {
    q_in_stride_s = num_heads_q * head_dim;
    q_in_stride_h = head_dim;
    q_in_stride_b = seq_len_q * num_heads_q * head_dim;
    k_in_stride_s = num_heads_kv * head_dim;
    k_in_stride_h = head_dim;
    k_in_stride_b = seq_len_kv * num_heads_kv * head_dim;
    v_stride_s = num_heads_kv * head_dim;
    v_stride_h = head_dim;
    v_stride_b = seq_len_kv * num_heads_kv * head_dim;
    o_stride_s = num_heads_q * head_dim;
    o_stride_h = head_dim;
    o_stride_b = seq_len_q * num_heads_q * head_dim;
  } else {
    q_in_stride_s = head_dim;
    q_in_stride_h = seq_len_q * head_dim;
    q_in_stride_b = num_heads_q * seq_len_q * head_dim;
    k_in_stride_s = head_dim;
    k_in_stride_h = seq_len_kv * head_dim;
    k_in_stride_b = num_heads_kv * seq_len_kv * head_dim;
    v_stride_s = head_dim;
    v_stride_h = seq_len_kv * head_dim;
    v_stride_b = num_heads_kv * seq_len_kv * head_dim;
    o_stride_s = head_dim;
    o_stride_h = seq_len_q * head_dim;
    o_stride_b = num_heads_q * seq_len_q * head_dim;
  }

  std::vector<ResultRow> rows;
  double dense_flops = compute_dense_flops(batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim);

  try {
    q->memcpy(dev_q, host_q.data(), q_count * sizeof(T)).wait();
    q->memcpy(dev_k, host_k.data(), kv_count * sizeof(T)).wait();
    q->memcpy(dev_v, host_v.data(), kv_count * sizeof(T)).wait();

    if (cli.tensor_layout == "NHD") {
      ark::XpuWrapper::sage_dynamic_quant_strided<T>(q, dev_q, dev_qi8, dev_qscale, batch, num_heads_q, seq_len_q,
                                                     q_blocks, head_dim, block_size, q_in_stride_s, q_in_stride_d,
                                                     q_in_stride_h, q_in_stride_b);
      ark::XpuWrapper::sage_dynamic_quant_strided<T>(q, dev_k, dev_ki8, dev_kscale, batch, num_heads_kv, seq_len_kv,
                                                     kv_blocks, head_dim, block_size, k_in_stride_s, k_in_stride_d,
                                                     k_in_stride_h, k_in_stride_b);
    } else {
      ark::XpuWrapper::sage_dynamic_quant<T>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len_q, q_blocks,
                                             head_dim, block_size);
      ark::XpuWrapper::sage_dynamic_quant<T>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len_kv, kv_blocks,
                                             head_dim, block_size);
    }

    if (!kSparseDevSparseOnly && cli.sparse_mode == "sparse") {
      double dense_ms = run_bench(
          [&] {
            ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_dense_out, nullptr, block_size, dev_qscale,
                                       dev_kscale, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s,
                                       k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                                       v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                                       num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, false, pv_dtype);
          },
          q, cli.warmup, cli.iters);

      rows.push_back(ResultRow{preset.name, "dense_sagev1", "dense", dtype_to_string(preset.value_dtype), 1.0, 1.0,
                               double(kv_blocks), dense_ms, flops_to_tflops(dense_flops, dense_ms),
                               flops_to_tflops(dense_flops, dense_ms)});
    }

    bool custom_pattern = cli.pattern.rfind("custom_", 0) == 0 || cli.pattern == "all_selected";
    std::vector<double> topks = custom_pattern ? std::vector<double>{1.0} : cli.topks;
    for (double topk : topks) {
      int num_select = std::max(1, std::min(kv_blocks, int(std::ceil(topk * kv_blocks))));
      std::vector<int> selection = build_pattern_indices(cli.pattern, kv_blocks, num_select);
      std::vector<int> host_lut;
      std::vector<int> host_valid;
      build_sparse_lut(batch, num_heads_q, q_blocks, kv_blocks, selection, host_lut, host_valid);

      auto* dev_lut = reinterpret_cast<int*>(ctx->allocate(host_lut.size() * sizeof(int)));
      auto* dev_valid = reinterpret_cast<int*>(ctx->allocate(host_valid.size() * sizeof(int)));
      q->memcpy(dev_lut, host_lut.data(), host_lut.size() * sizeof(int)).wait();
      q->memcpy(dev_valid, host_valid.data(), host_valid.size() * sizeof(int)).wait();

      double sparse_ms = run_bench(
          [&] {
            if (cli.sparse_mode == "sparse") {
              ark::sdpa_impl_qks8_sparse_pvhalf(
                  q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, block_size, dev_qscale, dev_kscale, dev_lut,
                  dev_valid, q_blocks, kv_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                  k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                  o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                  seq_len_kv, head_dim, softmax_scale, false, pv_dtype);
            } else if (cli.sparse_mode == "sparse_row_linear") {
              ark::sdpa_impl_qks8_sparse_row_linear_pvhalf(
                  q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, block_size, dev_qscale, dev_kscale, dev_lut,
                  dev_valid, q_blocks, kv_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                  k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                  o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                  seq_len_kv, head_dim, softmax_scale, false, pv_dtype);
            } else if (cli.sparse_mode == "sparse_row_linear_halfwidth") {
              ark::sdpa_impl_qks8_sparse_row_linear_halfwidth_pvhalf(
                  q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, block_size, dev_qscale, dev_kscale, dev_lut,
                  dev_valid, q_blocks, kv_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                  k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                  o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                  seq_len_kv, head_dim, softmax_scale, false, pv_dtype);
            } else if (cli.sparse_mode == "sparse_row_linear_profile") {
              if (cli.half_output_tile) {
                ark::sdpa_impl_qks8_sparse_row_linear_profile_pvhalf(
                    q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, block_size, dev_qscale, dev_kscale, dev_lut,
                    dev_valid, q_blocks, kv_blocks, q_tile_override, sparse_profile_mode, q_stride_s, q_stride_d,
                    q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
                    v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                    num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, false, pv_dtype);
              } else {
                ark::sdpa_impl_qks8_sparse_row_linear_profile(
                    q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, block_size, dev_qscale, dev_kscale, dev_lut,
                    dev_valid, q_blocks, kv_blocks, q_tile_override, sparse_profile_mode, q_stride_s, q_stride_d,
                    q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
                    v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                    num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, false, pv_dtype);
              }
            } else {
              throw std::invalid_argument("unsupported sparse mode: " + cli.sparse_mode);
            }
          },
          q, cli.warmup, cli.iters);

      double selected_ratio = double(selection.size()) / double(kv_blocks);
      rows.push_back(ResultRow{preset.name, sparse_mode_label(cli), cli.pattern, dtype_to_string(preset.value_dtype),
                               topk, selected_ratio, double(selection.size()), sparse_ms,
                               flops_to_tflops(dense_flops, sparse_ms),
                               flops_to_tflops(dense_flops * selected_ratio, sparse_ms)});

      ctx->deallocate(dev_lut);
      ctx->deallocate(dev_valid);
    }
  } catch (...) {
    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    if (dev_dense_out != nullptr) ctx->deallocate(dev_dense_out);
    ctx->deallocate(dev_sparse_out);
    throw;
  }

  ctx->deallocate(dev_q);
  ctx->deallocate(dev_k);
  ctx->deallocate(dev_v);
  ctx->deallocate(dev_qi8);
  ctx->deallocate(dev_ki8);
  ctx->deallocate(dev_qscale);
  ctx->deallocate(dev_kscale);
  if (dev_dense_out != nullptr) ctx->deallocate(dev_dense_out);
  ctx->deallocate(dev_sparse_out);
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliConfig cli = parse_args(argc, argv);
    PresetConfig preset = apply_overrides(get_preset(cli.preset), cli);
    std::vector<ResultRow> rows =
        preset.value_dtype == ValueDType::BF16 ? run_benchmark_typed<sycl::ext::oneapi::bfloat16>(preset, cli)
                                               : run_benchmark_typed<sycl::half>(preset, cli);
    print_rows(rows);
    if (cli.csv_path) write_csv(*cli.csv_path, rows);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[bench_sparse_kernel] error: " << ex.what() << '\n';
    return 1;
  }
}
