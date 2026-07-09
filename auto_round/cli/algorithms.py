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
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, ClassVar

from auto_round.algorithms.registry import (
    _register_algorithm_entry,
    get_algorithm_entry,
    iter_algorithm_entries,
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
            _register_algorithm_entry(
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

        # opt_rtn is an alias that should force optimized RTN unless the user
        # explicitly overrode --disable_opt_rtn/--enable_opt_rtn.
        if "opt_rtn" in names and getattr(args, "disable_opt_rtn", None) is None:
            setattr(args, "disable_opt_rtn", False)

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

        # Default quantization algorithm if none was specified.
        # iters=0 means RTN-family; disable_opt_rtn=False means opt_rtn.
        if not ({"awq", "rtn", "auto_round"} & seen):
            default_alg = cls.infer_default_algorithm(args)
            if default_alg == "opt_rtn":
                setattr(args, "disable_opt_rtn", False)
                canonical.append("rtn")
            else:
                canonical.append(default_alg)

        return [cls.get(name).build(args, common_kwargs) for name in canonical]

    @classmethod
    def infer_default_algorithm(cls, args) -> str:
        """Infer default algorithm when --algorithm is not provided."""
        if getattr(args, "iters", 0) == 0:
            if getattr(args, "disable_opt_rtn", None) is False:
                return "opt_rtn"
            return "rtn"
        return "auto_round"

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

    @staticmethod
    def _config_class_from_factory(config_factory: Callable[[], object] | type | None) -> type | None:
        """Resolve a config class from a registered factory/class."""
        if config_factory is None:
            return None
        if inspect.isclass(config_factory):
            return config_factory
        return None

    @classmethod
    def _introspect_config_keys(cls, config_factory: Callable[[], object] | type | None) -> tuple[list[str], list[str]]:
        """Return (direct_keys, forwarded_common_keys) for a config class."""
        config_cls = cls._config_class_from_factory(config_factory)
        if config_cls is None:
            return [], []

        sig = inspect.signature(config_cls.__init__)
        direct_keys: list[str] = []
        accepts_kwargs = False
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue
            direct_keys.append(param.name)

        forwarded_common: list[str] = []
        if accepts_kwargs and hasattr(config_cls, "_scheme_fields"):
            scheme_fields = sorted(getattr(config_cls, "_scheme_fields"))
            forwarded_common = ["scheme", *scheme_fields]
        return direct_keys, forwarded_common

    @classmethod
    def format_config_keys(cls, alg_name: str | None = None) -> str:
        """Render supported config keys for one algorithm or all algorithms."""
        entries = [entry for entry in iter_algorithm_entries() if entry.cli_handler is not None]
        if alg_name:
            canon = cls.resolve_alias(alg_name)
            if canon is None:
                supported = [entry.name for entry in entries]
                raise ValueError(f"Unknown algorithm '{alg_name}'. Supported: {', '.join(supported)}.")
            entries = [entry for entry in entries if entry.name == canon]

        lines: list[str] = []
        for entry in entries:
            config_cls = cls._config_class_from_factory(entry.config_factory)
            direct_keys, forwarded_common = cls._introspect_config_keys(entry.config_factory)
            lines.append(f"[{entry.name}] config class: {config_cls.__name__ if config_cls else 'N/A'}")
            lines.append(f"  direct_keys: {', '.join(direct_keys) if direct_keys else '-'}")
            if forwarded_common:
                lines.append(f"  forwarded_common_keys(**kwargs): {', '.join(forwarded_common)}")
        return "\n".join(lines)

    @staticmethod
    def _resolve_scalar_type(param: inspect.Parameter):
        """Infer argparse scalar type from default/annotation."""
        if param.default is not inspect._empty and param.default is not None:
            if isinstance(param.default, bool):
                return bool
            if isinstance(param.default, int):
                return int
            if isinstance(param.default, float):
                return float
            if isinstance(param.default, str):
                return str

        ann = param.annotation
        if ann in (int, float, str):
            return ann
        origin = getattr(ann, "__origin__", None)
        if origin is None:
            origin = getattr(ann, "origin", None)
        if origin in (UnionType, getattr(__import__("typing"), "Union", None)):
            args = [a for a in getattr(ann, "__args__", ()) if a is not type(None)]
            if len(args) == 1 and args[0] in (int, float, str):
                return args[0]
        return None

    @staticmethod
    def _is_optional_bool_param(param: inspect.Parameter) -> bool:
        """Return True for Optional[bool] / bool|None with default None."""
        if param.default is not None:
            return False
        ann = param.annotation
        if ann is bool:
            return True
        origin = getattr(ann, "__origin__", None)
        if origin is None:
            origin = getattr(ann, "origin", None)
        args = getattr(ann, "__args__", ())
        if origin in (UnionType, getattr(__import__("typing"), "Union", None)) and bool in args and type(None) in args:
            return True
        return False

    @staticmethod
    def _bool_flag_name(param_name: str, default_value: bool, *, option_prefix: str = "") -> str:
        """Build bool flag name using enable/disable naming conventions."""
        if default_value:
            if param_name.startswith("enable_"):
                flag = f"disable_{param_name[len('enable_') :]}"
            else:
                flag = f"disable_{param_name}"
        else:
            if param_name.startswith("disable_"):
                flag = f"enable_{param_name[len('disable_') :]}"
            elif param_name.startswith("enable_"):
                flag = param_name
            else:
                flag = f"enable_{param_name}"
        return f"--{option_prefix}{flag}"

    @classmethod
    def register_config_arguments(
        cls,
        group,
        config_factory: Callable[[], object] | type | None,
        *,
        exclude: set[str] | None = None,
        option_prefix: str = "",
        custom_types: dict[str, Any] | None = None,
    ) -> None:
        """Register argparse flags from a config __init__ signature."""
        config_cls = cls._config_class_from_factory(config_factory)
        if config_cls is None:
            return
        exclude = exclude or set()
        custom_types = custom_types or {}

        actions = getattr(group, "_actions", [])
        existing_dests = {a.dest for a in actions if getattr(a, "dest", None)}
        existing_opts = {opt for a in actions for opt in getattr(a, "option_strings", [])}

        sig = inspect.signature(config_cls.__init__)
        for param in sig.parameters.values():
            if param.name in {"self", "kwargs"} or param.name in exclude:
                continue

            dest = param.name
            if dest in existing_dests:
                # Already declared by common/runtime parser args.
                continue
            help_text = f"Config parameter: {dest}."

            # Bool parameters: default True => disable flag; default False => enable flag.
            if isinstance(param.default, bool):
                flag = cls._bool_flag_name(dest, bool(param.default), option_prefix=option_prefix)
                if flag in existing_opts:
                    continue
                action = "store_false" if param.default is True else "store_true"
                group.add_argument(flag, dest=dest, default=param.default, action=action, help=help_text)
                continue

            # Optional bool (tri-state): add explicit enable/disable pair.
            if cls._is_optional_bool_param(param):
                if dest.startswith("enable_"):
                    base = dest[len("enable_") :]
                    disable_name = f"disable_{base}"
                    enable_name = dest
                elif dest.startswith("disable_"):
                    base = dest[len("disable_") :]
                    disable_name = dest
                    enable_name = f"enable_{base}"
                else:
                    disable_name = f"disable_{dest}"
                    enable_name = f"enable_{dest}"

                disable_flag = f"--{option_prefix}{disable_name}"
                enable_flag = f"--{option_prefix}{enable_name}"
                if disable_flag in existing_opts and enable_flag in existing_opts:
                    continue
                mutex = group.add_mutually_exclusive_group()
                if disable_flag not in existing_opts:
                    mutex.add_argument(
                        disable_flag, dest=dest, default=None, action="store_const", const=True, help=help_text
                    )
                if enable_flag not in existing_opts:
                    mutex.add_argument(enable_flag, dest=dest, action="store_const", const=False, help=help_text)
                continue

            flag = f"--{option_prefix}{dest}"
            if flag in existing_opts:
                continue
            kwargs: dict[str, Any] = {"dest": dest, "help": help_text}
            if param.default is not inspect._empty:
                kwargs["default"] = param.default

            arg_type = custom_types.get(dest)
            if arg_type is None:
                arg_type = cls._resolve_scalar_type(param)
            if arg_type in (int, float, str):
                kwargs["type"] = arg_type
            elif arg_type is not None:
                # custom parser callable (e.g. duo_scaling)
                kwargs["type"] = arg_type

            group.add_argument(flag, **kwargs)

    @staticmethod
    def collect_config_kwargs(
        args, config_factory: Callable[[], object] | type | None, *, exclude: set[str] | None = None
    ) -> dict[str, Any]:
        """Collect config kwargs from parsed args using config signature keys."""
        config_cls = config_factory if inspect.isclass(config_factory) else None
        if config_cls is None:
            return {}
        exclude = exclude or set()
        sig = inspect.signature(config_cls.__init__)
        out: dict[str, Any] = {}
        for param in sig.parameters.values():
            if param.name in {"self", "kwargs"} or param.name in exclude:
                continue
            out[param.name] = getattr(args, param.name, param.default)
        return out


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


@dataclass(frozen=True)
class ConfigDrivenAlgoSpec:
    name: str
    aliases: tuple[str, ...]
    summary: str
    config_cls: type
    exclude: frozenset[str] = frozenset()
    option_prefix: str = ""
    custom_types: dict[str, Any] | None = None


# Central place for non-Hadamard algorithm definitions.
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.awq.config import AWQConfig

CONFIG_DRIVEN_ALGO_SPECS: dict[str, ConfigDrivenAlgoSpec] = {
    "awq": ConfigDrivenAlgoSpec(
        name="awq",
        aliases=("awq",),
        summary="Activation-Aware Weight Quantization (pre-processing).",
        config_cls=AWQConfig,
        exclude=frozenset({"mappings"}),
        option_prefix="awq-",
        custom_types={"duo_scaling": _parse_bool_or_mode},
    ),
    "rtn": ConfigDrivenAlgoSpec(
        name="rtn",
        aliases=("rtn", "opt_rtn"),
        summary="Round-To-Nearest quantization.",
        config_cls=RTNConfig,
    ),
    "auto_round": ConfigDrivenAlgoSpec(
        name="auto_round",
        aliases=("auto_round", "autoround", "sign_round", "signround"),
        summary="SignRound-style iterative block quantization.",
        config_cls=SignRoundConfig,
    ),
}


# ============================================================================
# Algorithm implementations  (auto-registered via __init_subclass__)
# ============================================================================


class AWQ(AlgorithmHandler):
    _SPEC = CONFIG_DRIVEN_ALGO_SPECS["awq"]
    name = _SPEC.name
    aliases = _SPEC.aliases
    summary = _SPEC.summary
    config_factory = _SPEC.config_cls

    def register(self, group) -> None:
        self.register_config_arguments(
            group,
            self._SPEC.config_cls,
            exclude=set(self._SPEC.exclude),
            option_prefix=self._SPEC.option_prefix,
            custom_types=self._SPEC.custom_types,
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        cfg_kwargs = self.collect_config_kwargs(args, self._SPEC.config_cls, exclude=set(self._SPEC.exclude))
        return self._SPEC.config_cls(**cfg_kwargs, **common_kwargs)


class RTN(AlgorithmHandler):
    _SPEC = CONFIG_DRIVEN_ALGO_SPECS["rtn"]
    name = _SPEC.name
    aliases = _SPEC.aliases
    summary = _SPEC.summary
    config_factory = _SPEC.config_cls

    def register(self, group) -> None:
        self.register_config_arguments(group, self._SPEC.config_cls)

    def build(self, args, common_kwargs: dict[str, Any]):
        cfg_kwargs = self.collect_config_kwargs(args, self._SPEC.config_cls)
        return self._SPEC.config_cls(**cfg_kwargs, **common_kwargs)


class AutoRound(AlgorithmHandler):
    _SPEC = CONFIG_DRIVEN_ALGO_SPECS["auto_round"]
    name = _SPEC.name
    aliases = _SPEC.aliases
    summary = _SPEC.summary
    config_factory = _SPEC.config_cls

    def register(self, group) -> None:
        self.register_config_arguments(group, self._SPEC.config_cls)

    def build(self, args, common_kwargs: dict[str, Any]):
        cfg_kwargs = self.collect_config_kwargs(args, self._SPEC.config_cls)
        return self._SPEC.config_cls(**cfg_kwargs, **common_kwargs)


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
    from auto_round.algorithms.transforms.hadamard.config import RotationConfig

    _register_algorithm_entry(
        CONFIG_DRIVEN_ALGO_SPECS["rtn"].name,
        aliases=CONFIG_DRIVEN_ALGO_SPECS["rtn"].aliases,
        config_factory=CONFIG_DRIVEN_ALGO_SPECS["rtn"].config_cls,
        cli_handler=RTN,
        summary=CONFIG_DRIVEN_ALGO_SPECS["rtn"].summary,
    )
    _register_algorithm_entry(
        CONFIG_DRIVEN_ALGO_SPECS["auto_round"].name,
        aliases=CONFIG_DRIVEN_ALGO_SPECS["auto_round"].aliases,
        config_factory=CONFIG_DRIVEN_ALGO_SPECS["auto_round"].config_cls,
        cli_handler=AutoRound,
        summary=CONFIG_DRIVEN_ALGO_SPECS["auto_round"].summary,
    )
    _register_algorithm_entry(
        CONFIG_DRIVEN_ALGO_SPECS["awq"].name,
        aliases=CONFIG_DRIVEN_ALGO_SPECS["awq"].aliases,
        config_factory=CONFIG_DRIVEN_ALGO_SPECS["awq"].config_cls,
        cli_handler=AWQ,
        summary=CONFIG_DRIVEN_ALGO_SPECS["awq"].summary,
    )
    _register_algorithm_entry(
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
