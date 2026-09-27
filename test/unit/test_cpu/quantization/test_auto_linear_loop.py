"""Tests for the one-shot auto linear_loop pick (AR_MOE_EXPERTS_IMPL=auto).

Grouped tuning stacks per-expert fp32 tensors on every expert-homing device;
when any such device cannot hold state + stacks + routed transients beside the
reserve, the first MoE tune block must switch the run to linear_loop BEFORE
the loop starts (instead of OOMing inside grouped_mm and re-staging the sliced
fallback). Explicit AR_MOE_EXPERTS_IMPL choices are never overridden.
"""

import unittest
from unittest import mock

GB = 2**30


def _envs_auto():
    from auto_round import envs

    return mock.patch.object(envs, "AR_MOE_EXPERTS_IMPL", "auto")


class _Cfg:
    def __init__(self):
        self._experts_implementation = "linear_grouped"


class TestAutoLinearLoop(unittest.TestCase):
    def setUp(self):
        import auto_round.algorithms.quantization.sign_round.quantizer as q

        self.q = q
        q._MOE_IMPL_AUTO_LINEAR_LOOP_DONE = False
        self.cfg = _Cfg()
        self.state = {"cuda:1": 14 * GB, "cuda:2": 14 * GB}  # 14 B/param layout
        self.act = {"cuda:1": 12 * GB, "cuda:2": 2 * GB}  # stacks dominate cuda:1

    def _run(self, iters=20, stacks=True):
        q = self.q
        with _envs_auto():
            with mock.patch.object(q, "_logical_state_by_device", return_value=self.state):
                with mock.patch.object(q, "_activation_bytes_by_device", return_value=self.act):
                    with mock.patch.object(
                        q, "_grouped_stack_bytes", side_effect=lambda b, d, config=None: 1 if stacks else 0
                    ):
                        with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=16 * GB):
                            q._maybe_auto_linear_loop_for_tuning(object(), [object()], 8, iters, self.cfg, None)

    def test_over_lane_switches_before_the_loop(self):
        # demand on cuda:1 = 14*6/14 (grads+bf16; snapshot parks) + 12 + 0.25
        # = 20.25 GiB > free 16 (guaranteed in-loop terms only)
        self._run()
        self.assertEqual(self.cfg._experts_implementation, "linear_loop")
        self.assertTrue(self.q._MOE_IMPL_AUTO_LINEAR_LOOP_DONE)

    def test_fits_lane_keeps_grouped(self):
        self.act = {"cuda:1": 1 * GB, "cuda:2": 1 * GB}
        self._run()
        self.assertEqual(self.cfg._experts_implementation, "linear_grouped")

    def test_explicit_impl_never_overridden(self):
        from auto_round import envs

        with mock.patch.object(envs, "AR_MOE_EXPERTS_IMPL", "linear_grouped"):
            self.q._maybe_auto_linear_loop_for_tuning(object(), [], 8, 20, self.cfg, None)
        self.assertEqual(self.cfg._experts_implementation, "linear_grouped")

    def test_iters0_untouched(self):
        self._run(iters=0)
        self.assertEqual(self.cfg._experts_implementation, "linear_grouped")

    def test_model_context_config_without_impl_attr_still_decides(self):
        # server regression: model_context.config never received
        # _experts_implementation (prepare writes it on model.config); the
        # old single-candidate read silently returned and the 4x3090 lane
        # never switched. The check must fall through to model.config, and
        # a switch must update BOTH config objects.
        ctx_cfg = type("CtxCfg", (), {})()  # no _experts_implementation attr
        model = type("M", (), {})()
        model.config = self.cfg
        q = self.q
        with _envs_auto():
            with mock.patch.object(q, "_logical_state_by_device", return_value=self.state):
                with mock.patch.object(q, "_activation_bytes_by_device", return_value=self.act):
                    with mock.patch.object(q, "_grouped_stack_bytes", side_effect=lambda b, d, config=None: 1):
                        with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=16 * GB):
                            q._maybe_auto_linear_loop_for_tuning(object(), [object()], 8, 20, ctx_cfg, model)
        self.assertEqual(self.cfg._experts_implementation, "linear_loop")
        self.assertEqual(ctx_cfg._experts_implementation, "linear_loop")

    def test_snapshot_parks_to_host_so_not_charged_as_guaranteed(self):
        # boundary between the old 10/14 and new 6/14 multipliers: this lane
        # must KEEP grouped (a working knife-edge lane must not be switched)
        q = self.q
        state = {"cuda:1": 20 * GB}  # remaining: 10/14 -> 14.3 (old switch), 6/14 -> 8.6
        act = {"cuda:1": 5 * GB}
        with _envs_auto():
            with mock.patch.object(q, "_logical_state_by_device", return_value=state):
                with mock.patch.object(q, "_activation_bytes_by_device", return_value=act):
                    with mock.patch.object(q, "_grouped_stack_bytes", side_effect=lambda b, d, config=None: 1):
                        with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=16 * GB):
                            q._maybe_auto_linear_loop_for_tuning(object(), [object()], 8, 20, self.cfg, None)
        self.assertEqual(self.cfg._experts_implementation, "linear_grouped")

    def test_second_run_same_process_re_decides(self):
        # the done-flag is keyed by the run's config: a second model
        # quantized in the same process must get its own auto decision
        q = self.q
        q._MOE_IMPL_AUTO_LINEAR_LOOP_DONE = True  # a previous run finished
        cfg2 = _Cfg()
        with _envs_auto():
            with mock.patch.object(q, "_logical_state_by_device", return_value=self.state):
                with mock.patch.object(q, "_activation_bytes_by_device", return_value=self.act):
                    with mock.patch.object(q, "_grouped_stack_bytes", side_effect=lambda b, d, config=None: 1):
                        with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=16 * GB):
                            q._maybe_auto_linear_loop_for_tuning(object(), [object()], 8, 20, cfg2, None)
        self.assertEqual(cfg2._experts_implementation, "linear_loop")  # decided, not skipped

    def test_dense_block_no_stacks_no_decision(self):
        # no grouped stacks homed: not the decision point, flag stays unset
        self._run(stacks=False)
        self.assertEqual(self.cfg._experts_implementation, "linear_grouped")
        self.assertFalse(self.q._MOE_IMPL_AUTO_LINEAR_LOOP_DONE)
