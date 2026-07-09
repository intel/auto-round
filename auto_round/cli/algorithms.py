# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Algorithm discovery, registration, listing, and config building for the CLI.

To add a new algorithm:
1. Write a class that extends AlgorithmHandler and set name/aliases/summary.
2. Implement register(group) to declare argparse flags.
3. Implement build(args, common_kwargs) to return a config object.

No manual registry update needed — subclasses are auto-registered on definition.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from auto_round.algorithms.registry import (
    get_algorithm_entry,
    iter_algorithm_entries,
    register_algorithm,
    resolve_algorithm_alias,
)

# ============================================================================
# Base class + registry
# ============================================================================


class AlgorithmHandler(ABC):
    """Bundles everything the CLI needs to know about one algorithm.

    Concrete subclasses are auto-registered in the class-level registry
    the moment their class body is processed.
    """

    name: str  # canonical name used in --algorithm
    aliases: tuple[str, ...] = ()  # all accepted names, including canonical
    summary: str = ""  # one-liner shown by `auto_round list alg`
    config_factory: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # A class that overrides both register and build is considered a concrete
        # algorithm implementation and must also declare a `name` class attribute.
        if "register" in cls.__dict__ and "build" in cls.__dict__:
            if "name" not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define a 'name' class attribute " f"(e.g. name = 'my_algo').")
            register_algorithm(
                cls.name,
                aliases=cls.aliases,
                config_factory=cls.config_factory,
                cli_handler=cls,
                summary=cls.summary,
            )

    # ------------------------------------------------------------------
    # Abstract interface — implement in each subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def register(self, group) -> None:
        """Add argparse arguments to *group*."""

    @abstractmethod
    def build(self, args, common_kwargs: dict[str, Any]) -> Any:
        """Build and return an algorithm config from parsed *args*."""

    # ------------------------------------------------------------------
    # Registry operations — called on the class, not on instances
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> AlgorithmHandler:
        """Return the handler for a canonical algorithm name. Raises KeyError if unknown."""
        entry = get_algorithm_entry(name)
        if entry.cli_handler is None:
            raise KeyError(f"No handler registered for algorithm '{name}'.")
        return entry.cli_handler()

    @classmethod
    def resolve_alias(cls, user_name: str) -> str | None:
        """Resolve a user-supplied algorithm name or alias to the canonical name.

        Returns None instead of raising so callers can silently skip unknowns.
        """
        return resolve_algorithm_alias(user_name)

    @classmethod
    def add_groups(cls, parser) -> None:
        """Add an argparse argument group for every registered algorithm."""
        for entry in iter_algorithm_entries():
            if entry.cli_handler is None:
                continue
            handler = entry.cli_handler()
            group = parser.add_argument_group(f"Algorithm: {handler.name}")
            handler.register(group)

    @classmethod
    def build_configs(cls, args, common_kwargs: dict[str, Any]) -> list:
        """Build the ordered algorithm config list from parsed CLI args."""
        raw = getattr(args, "algorithm", None) or ""
        names = [n.strip().lower() for n in raw.split(",") if n.strip()]

        # Infer hadamard when --rotation_type is given
        if getattr(args, "rotation_hadamard_type", None) and "hadamard" not in names:
            names.append("hadamard")

        # Resolve aliases, drop unknowns, deduplicate preserving order
        seen: set[str] = set()
        canonical: list[str] = []
        for n in names:
            c = cls.resolve_alias(n)
            if c and c not in seen:
                canonical.append(c)
                seen.add(c)

        # Default quantization algorithm if none was specified
        if not ({"awq", "rtn", "auto_round"} & seen):
            canonical.append("rtn" if getattr(args, "iters", 0) == 0 else "auto_round")

        return [cls.get(name).build(args, common_kwargs) for name in canonical]

    @classmethod
    def format_listing(cls) -> str:
        """Render the short `list alg` output."""
        lines = []
        for entry in iter_algorithm_entries():
            if entry.cli_handler is None:
                continue
            handler = entry.cli_handler()
            other = [a for a in handler.aliases if a != handler.name]
            alias_str = f" (aliases: {', '.join(other)})" if other else ""
            lines.append(f"- {handler.name}{alias_str}: {handler.summary}")
        return "\n".join(lines)

    @classmethod
    def format_detail(cls, name: str) -> str:
        """Render detailed help text for one algorithm."""
        canon = cls.resolve_alias(name)
        if canon is None:
            supported = [entry.name for entry in iter_algorithm_entries() if entry.cli_handler is not None]
            raise ValueError(f"Unknown algorithm '{name}'. " f"Supported: {', '.join(supported)}.")
        handler = cls.get(canon)
        lines = [f"{handler.name}: {handler.summary}"]
        other = [a for a in handler.aliases if a != handler.name]
        if other:
            lines.append(f"Aliases: {', '.join(other)}")
        temp = argparse.ArgumentParser(add_help=False)
        group = temp.add_argument_group(f"Flags for {handler.name}")
        handler.register(group)
        for action in group._group_actions:
            flags = ", ".join(action.option_strings)
            default = f" (default: {action.default})" if action.default is not None else ""
            lines.append(f"  {flags}: {action.help or ''}{default}")
        return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================


def _parse_bool_or_mode(value: str) -> bool | str:
    """Parse AWQ duo_scaling's tri-state: true / false / both."""
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "both":
        return "both"
    raise argparse.ArgumentTypeError("Expected one of: true, false, both")


# ============================================================================
# Algorithm implementations  (auto-registered via __init_subclass__)
# ============================================================================


class AWQ(AlgorithmHandler):
    name = "awq"
    aliases = ("awq",)
    summary = "Activation-Aware Weight Quantization (pre-processing)."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--awq-duo-scaling",
            dest="duo_scaling",
            default=True,
            type=_parse_bool_or_mode,
            metavar="{true,false,both}",
            help="Use activation+weight duo scaling (true/false/both).",
        )
        group.add_argument(
            "--awq-n-grid",
            dest="n_grid",
            default=20,
            type=int,
            help="Number of grid-search points for AWQ scaling ratio.",
        )
        group.add_argument(
            "--awq-apply-clip",
            dest="awq_apply_clip",
            action="store_true",
            help="Search and hard-clamp per-group AWQ weight clipping after smoothing.",
        )
        group.add_argument(
            "--awq-clip-as-init",
            dest="awq_clip_as_init",
            action="store_true",
            help=(
                "Use the searched AWQ clip to initialize the block quantizer's "
                "weight range instead of hard-clamping (requires --awq-apply-clip)."
            ),
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        return AWQConfig(
            duo_scaling=getattr(args, "duo_scaling", True),
            n_grid=getattr(args, "n_grid", 20),
            apply_clip=getattr(args, "awq_apply_clip", False),
            clip_as_init=getattr(args, "awq_clip_as_init", False),
            **common_kwargs,
        )


class RTN(AlgorithmHandler):
    name = "rtn"
    aliases = ("rtn",)
    summary = "Round-To-Nearest quantization."
    config_factory = None

    def register(self, group) -> None:
        mutex = group.add_mutually_exclusive_group()
        mutex.add_argument(
            "--disable_opt_rtn",
            dest="disable_opt_rtn",
            default=None,
            action="store_const",
            const=True,
            help="Force plain RTN (disable optimized path).",
        )
        mutex.add_argument(
            "--enable_opt_rtn",
            dest="disable_opt_rtn",
            action="store_const",
            const=False,
            help="Force optimized RTN path.",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        return RTNConfig(
            disable_opt_rtn=getattr(args, "disable_opt_rtn", None),
            **common_kwargs,
        )


class AutoRound(AlgorithmHandler):
    name = "auto_round"
    aliases = ("auto_round", "autoround", "sign_round", "signround")
    summary = "SignRound-style iterative block quantization."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--iters", "--iter", default=None, type=int, help="Number of optimization iterations per block."
        )
        group.add_argument("--lr", default=None, type=float, help="Learning rate for rounding optimization.")
        group.add_argument("--minmax_lr", default=None, type=float, help="Learning rate for min-max tuning.")
        group.add_argument("--momentum", default=0.0, type=float, help="Momentum factor for the optimizer.")
        group.add_argument("--nblocks", default=1, type=int, help="Number of blocks to optimize together.")
        minmax_mutex = group.add_mutually_exclusive_group()
        minmax_mutex.add_argument(
            "--enable_minmax_tuning",
            default=True,
            dest="enable_minmax_tuning",
            action="store_true",
            help="Tune weight min/max ranges.",
        )
        minmax_mutex.add_argument(
            "--no-enable_minmax_tuning",
            "--disable_minmax_tuning",
            dest="enable_minmax_tuning",
            action="store_false",
            help="Disable weight min/max tuning.",
        )
        group.add_argument(
            "--enable_norm_bias_tuning",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Tune normalization and bias terms.",
        )
        group.add_argument(
            "--gradient_accumulate_steps", default=1, type=int, help="Gradient accumulation steps per update."
        )
        group.add_argument(
            "--enable_alg_ext",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Enable experimental SignRound extension.",
        )
        group.add_argument(
            "--not_use_best_mse",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Skip restoring best-MSE checkpoint.",
        )
        quanted_input_mutex = group.add_mutually_exclusive_group()
        quanted_input_mutex.add_argument(
            "--enable_quanted_input",
            default=True,
            dest="enable_quanted_input",
            action="store_true",
            help="Consume quantized output of previous blocks.",
        )
        quanted_input_mutex.add_argument(
            "--no-enable_quanted_input",
            "--disable_quanted_input",
            dest="enable_quanted_input",
            action="store_false",
            help="Disable quantized-input propagation across blocks.",
        )
        group.add_argument(
            "--enable_adam",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Use the Adam-based SignRound variant.",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        return SignRoundConfig(
            iters=getattr(args, "iters", 200),
            lr=getattr(args, "lr", None),
            minmax_lr=getattr(args, "minmax_lr", None),
            momentum=getattr(args, "momentum", 0.0),
            nblocks=getattr(args, "nblocks", 1),
            enable_minmax_tuning=getattr(args, "enable_minmax_tuning", True),
            enable_norm_bias_tuning=getattr(args, "enable_norm_bias_tuning", False),
            gradient_accumulate_steps=getattr(args, "gradient_accumulate_steps", 1),
            enable_alg_ext=getattr(args, "enable_alg_ext", False),
            not_use_best_mse=getattr(args, "not_use_best_mse", False),
            enable_quanted_input=getattr(args, "enable_quanted_input", True),
            enable_adam=getattr(args, "enable_adam", False),
            **common_kwargs,
        )


class Hadamard(AlgorithmHandler):
    name = "hadamard"
    aliases = ("hadamard", "random_hadamard", "quarot_hadamard")
    summary = "Hadamard rotation/transform applied before quantization."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--rotation_type",
            "--rotation-hadamard-type",
            dest="rotation_hadamard_type",
            default=None,
            choices=["hadamard", "random_hadamard", "quarot_hadamard"],
            help="Hadamard transform variant.",
        )
        group.add_argument(
            "--rotation_backend",
            dest="rotation_backend",
            default="auto",
            choices=["auto", "inplace", "transform"],
            help="Rotation backend to use.",
        )
        group.add_argument(
            "--rotation_block_size",
            dest="rotation_block_size",
            default=None,
            type=int,
            help="Grouped Hadamard block size.",
        )
        group.add_argument(
            "--fuse_online_to_weight",
            default=None,
            action=argparse.BooleanOptionalAction,
            help="Fuse online Hadamard rotation into weights.",
        )
        group.add_argument(
            "--allow_online_rotation",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Allow online activation rotation.",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.transforms.hadamard.config import RotationConfig

        hadamard_type = getattr(args, "rotation_hadamard_type", None) or "hadamard"
        return RotationConfig(
            hadamard_type=hadamard_type,
            backend=getattr(args, "rotation_backend", "auto"),
            block_size=getattr(args, "rotation_block_size", None),
            fuse_online_to_weight=getattr(args, "fuse_online_to_weight", None),
            allow_online_rotation=getattr(args, "allow_online_rotation", True),
        )


def _register_builtin_algorithm_factories() -> None:
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.awq.config import AWQConfig
    from auto_round.algorithms.transforms.hadamard.config import RotationConfig

    register_algorithm("rtn", aliases=("rtn",), config_factory=RTNConfig, cli_handler=RTN, summary=RTN.summary)
    register_algorithm(
        "auto_round",
        aliases=("auto_round", "autoround", "sign_round", "signround"),
        config_factory=SignRoundConfig,
        cli_handler=AutoRound,
        summary=AutoRound.summary,
    )
    register_algorithm("awq", aliases=("awq",), config_factory=AWQConfig, cli_handler=AWQ, summary=AWQ.summary)
    register_algorithm(
        "hadamard",
        aliases=("hadamard", "random_hadamard", "quarot_hadamard"),
        config_factory=RotationConfig,
        cli_handler=Hadamard,
        summary=Hadamard.summary,
        alias_factories={
            "random_hadamard": lambda: RotationConfig(hadamard_type="random_hadamard"),
            "quarot_hadamard": lambda: RotationConfig(hadamard_type="quarot_hadamard"),
        },
    )


_register_builtin_algorithm_factories()
