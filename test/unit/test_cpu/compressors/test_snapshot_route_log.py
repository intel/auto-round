# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Snapshot routing: only the OOM fallbacks log; the normal route never does."""

from types import SimpleNamespace

from auto_round.compressors.utils import _snapshot_route_log_


class TestSnapshotRouteLog:
    def test_info_once_then_debug(self, caplog):
        import logging

        from auto_round.logger import logger as ar_logger

        block = SimpleNamespace()
        with caplog.at_level(logging.DEBUG, logger="autoround"):
            ar_logger.addHandler(caplog.handler)
            try:
                _snapshot_route_log_(block, "local", "[snapshot] %.2fGiB stays beside the weights", 1.41, silent=True)
                assert block._snapshot_route == "local"  # route still recorded for stickiness
                _snapshot_route_log_(block, "peer:cuda:1", "[snapshot] cloning %.2fGiB to idle peer %s", 1.61, "cuda:1")
                _snapshot_route_log_(block, "peer:cuda:1", "[snapshot] cloning %.2fGiB to idle peer %s", 1.61, "cuda:1")
                _snapshot_route_log_(block, "host", "[snapshot] parking on host", warn_first=True)
                _snapshot_route_log_(block, "host", "[snapshot] parking on host", warn_first=True)
            finally:
                ar_logger.removeHandler(caplog.handler)
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not [r for r in caplog.records if r.levelno == logging.DEBUG]  # the normal route never logs
        assert len(infos) == 1  # first peer announcement only
        assert len(warns) == 1  # first host fallback only
        assert len(caplog.records) == 2  # repeats log nothing at any level
