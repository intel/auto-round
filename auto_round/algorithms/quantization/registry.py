# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from auto_round.algorithms.registry import list_registered_algorithms, register_algorithm, resolve_alg_config


def register_alg(alias, factory):
    register_algorithm(alias, aliases=(alias,), config_factory=factory)


__all__ = ["register_alg", "resolve_alg_config", "list_registered_algorithms"]
