# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import Any, Optional

from auto_round.logger import logger


class VLLMModelBackend:
    """vLLM runtime backend for loading and accessing the worker model.

    This class is intentionally small and isolated so any vLLM internal layout
    changes can be handled in one place.
    """

    def __init__(self, model_id_or_path: str, vllm_args: Optional[dict[str, Any]] = None) -> None:
        self.model_id_or_path = model_id_or_path
        self.vllm_args = dict(vllm_args or {})
        self._llm = None

    @staticmethod
    def _ensure_vllm_runtime_env() -> None:
        """Set required vLLM runtime env defaults for stable in-process model access."""
        # Keep single-process engine behavior unless caller explicitly overrides.
        if "VLLM_ENABLE_V1_MULTIPROCESSING" not in os.environ:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 for VLLMCompressor runtime.")

    @property
    def llm(self):
        return self._llm

    @classmethod
    def load(cls, model_id_or_path: str, vllm_args: Optional[dict[str, Any]] = None) -> VLLMModelBackend:
        """Load vLLM engine and return an initialized backend instance."""
        backend = cls(model_id_or_path, vllm_args)
        backend._do_load()
        return backend

    def _do_load(self) -> None:
        """Internal method to initialize the vLLM engine."""
        self._ensure_vllm_runtime_env()
        self._validate_constraints()
        import vllm

        if self._llm is None:
            # Add enforce_eager=True to enable eager execution (no CUDA graphs)
            # This allows direct forward calls on the model without forward_context issues
            vllm_args = dict(self.vllm_args or {})
            if "enforce_eager" not in vllm_args:
                vllm_args["enforce_eager"] = True
                logger.info("Setting enforce_eager=True for calibration (eager execution mode)")
            self._llm = vllm.LLM(model=self.model_id_or_path, **vllm_args)

    def _validate_constraints(self) -> None:
        self._ensure_vllm_runtime_env()
        from packaging.version import Version
        import vllm

        version = Version(vllm.__version__)
        # Support 0.22.x through 0.23.x (adjust as needed for your deployment)
        min_version = Version("0.22.0")
        max_version = Version("0.25.0")
        if not (min_version <= version < max_version):
            raise ValueError(
                f"Unsupported vLLM version {vllm.__version__}. "
                f"Expected >={min_version} and <{max_version} for VLLMCompressor."
            )

        tp = int(self.vllm_args.get("tensor_parallel_size", 1))
        if tp != 1:
            raise ValueError(
                "VLLMCompressor currently supports tensor_parallel_size=1 only, "
                f"but got {tp}."
            )

    def _first_existing_attr(self, obj: Any, path: list[str]) -> Any:
        """Traverse nested attributes safely, returning None if any step fails."""
        current = obj
        for attr_name in path:
            try:
                if current is None:
                    return None
                if not hasattr(current, attr_name):
                    return None
                current = getattr(current, attr_name)
            except Exception:
                return None
        return current

    def get_model(self):
        if self._llm is None:
            raise RuntimeError("Backend is not loaded. Call load() before get_model().")

        # Try vLLM V1 method first: driver_worker.get_model()
        try:
            if hasattr(self._llm, "llm_engine"):
                engine = self._llm.llm_engine
                if hasattr(engine, "model_executor"):
                    executor = engine.model_executor
                    if hasattr(executor, "driver_worker"):
                        driver_worker = executor.driver_worker
                        if hasattr(driver_worker, "get_model"):
                            model = driver_worker.get_model()
                            if model is not None:
                                logger.info("Located model via driver_worker.get_model()")
                                return model
        except Exception as e:
            logger.debug(f"get_model() method failed: {e}")

        # Fallback: try standard attribute paths
        candidate_paths = [
            ["llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model"],
            ["llm_engine", "model_executor", "driver_worker", "model_runner", "model"],
            ["llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"],
            ["llm_engine", "model_executor", "executor", "model"],
        ]

        for path in candidate_paths:
            model = self._first_existing_attr(self._llm, path)
            if model is not None:
                logger.info(f"Located model via path: {'.'.join(path)}")
                return model

        raise RuntimeError(
            "Could not locate worker torch model from vLLM internals. "
            "Tried driver_worker.get_model() and standard attribute paths. "
            "Please verify vLLM installation or version compatibility."
        )

    def get_tokenizer(self):
        if self._llm is None:
            return None
        for attr in ("get_tokenizer", "get_tokenizer_group"):
            if hasattr(self._llm, attr):
                try:
                    val = getattr(self._llm, attr)()
                    if val is not None:
                        return val
                except Exception:
                    continue
        tok = self._first_existing_attr(self._llm, ["llm_engine", "tokenizer"])
        return tok

    def run_prefill(self, inputs: list[str] | str) -> Any:
        """Run lightweight prefill-like execution for calibration collection."""
        if self._llm is None:
            raise RuntimeError("Backend is not loaded. Call load() before run_prefill().")

        prompts = inputs
        if isinstance(prompts, str):
            prompts = [prompts]

        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=1)
        return self._llm.generate(prompts, sampling)

    def close(self) -> None:
        if self._llm is None:
            return
        try:
            llm_engine = getattr(self._llm, "llm_engine", None)
            if llm_engine is not None and hasattr(llm_engine, "shutdown"):
                llm_engine.shutdown()
        except Exception as err:
            logger.debug("Ignoring vLLM engine shutdown error: %s", err)
        finally:
            self._llm = None
