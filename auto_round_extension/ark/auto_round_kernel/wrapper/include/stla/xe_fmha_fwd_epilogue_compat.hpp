#pragma once

namespace cutlass::fmha::kernel::detail {

template <class Epilogue, class TensorO2D, class FragA, class FragARow, class QVCoord>
auto run_fmha_fwd_epilogue(Epilogue& epilogue,
                           TensorO2D const& O,
                           FragA& tArA,
                           FragARow& tA_max,
                           FragARow& tA_sum,
                           QVCoord blk_qv,
                           int thr_id,
                           int head_q,
                           int idx_b,
                           int)
    -> decltype(epilogue(O, tArA, tA_max, tA_sum, blk_qv, thr_id, head_q, idx_b), void()) {
  epilogue(O, tArA, tA_max, tA_sum, blk_qv, thr_id, head_q, idx_b);
}

template <class Epilogue, class TensorO2D, class FragA, class FragARow, class QVCoord>
auto run_fmha_fwd_epilogue(Epilogue& epilogue,
                           TensorO2D const& O,
                           FragA& tArA,
                           FragARow& tA_max,
                           FragARow& tA_sum,
                           QVCoord blk_qv,
                           int thr_id,
                           int,
                           int,
                           long)
    -> decltype(epilogue(O, tArA, tA_max, tA_sum, blk_qv, thr_id), void()) {
  epilogue(O, tArA, tA_max, tA_sum, blk_qv, thr_id);
}

}  // namespace cutlass::fmha::kernel::detail
