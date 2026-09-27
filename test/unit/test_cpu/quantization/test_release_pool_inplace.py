# coding=utf-8
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-place calibration-pool release (composer)."""

import unittest

import torch

from auto_round.algorithms.composer import _release_pool_inplace


class TestReleasePoolInplace(unittest.TestCase):
    def test_list_pool_entries_nulled_and_shared_ref_sees_it(self):
        pool = [torch.zeros(2), torch.zeros(2)]
        shared = pool  # the orchestrator's input_ids view
        _release_pool_inplace(pool)
        self.assertEqual(pool, [None, None])
        self.assertIs(shared, pool)
        self.assertEqual(shared, [None, None])

    def test_dict_of_lists_recurces(self):
        pool = {"hidden_states": [torch.zeros(2)], "other": {"nested": [torch.zeros(2)]}}
        _release_pool_inplace(pool)
        self.assertEqual(pool["hidden_states"], [None])
        self.assertEqual(pool["other"]["nested"], [None])
        self.assertEqual(list(pool.keys()), ["hidden_states", "other"])  # structure preserved

    def test_tuple_skipped_immutable(self):
        pool = (torch.zeros(2), torch.zeros(2))
        _release_pool_inplace(pool)
        # tuples cannot be mutated in place; tensors must remain untouched
        self.assertIsInstance(pool[0], torch.Tensor)
        self.assertIsInstance(pool[1], torch.Tensor)

    def test_mixed_structure(self):
        pool = {"a": [torch.zeros(1), (torch.zeros(1),)], "b": "scalar"}
        _release_pool_inplace(pool)
        self.assertEqual(pool["a"][0], None)
        self.assertIsNone(pool["a"][1])  # slot dropped wholesale; tuple internals moot
        self.assertEqual(pool["b"], "scalar")
