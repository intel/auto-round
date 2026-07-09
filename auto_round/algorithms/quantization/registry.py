# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from auto_round.algorithms.registry import _register_algorithm_entry, list_registered_algorithms, resolve_alg_config


def register_alg(alias, factory):
    _register_algorithm_entry(alias, aliases=(alias,), config_factory=factory)


__all__ = ["register_alg", "resolve_alg_config", "list_registered_algorithms"]
