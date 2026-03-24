
from typing import Any

from auto_round.experimental.transform.transform_config import TransformConfig
from auto_round.experimental.transform.transforms import TRANSFORMS

def _normalize_transform_config(transform_config: Any, scheme: str = None) -> dict[str, Any]:
    """
    Normalize and validate `transform_config`.

    Supported input types:
        - None          -> {}
        - dict          -> validated via TransformConfig
        - TransformConfig -> validated & converted to dict
        - str           -> shorthand for `transform_type` in TRANSFORMS keys

    On any validation failure, raises ValueError/TypeError.
    """
    # 1) None -> {}
    if transform_config is None:
        return {}

    # 2) Already a TransformConfig instance
    if isinstance(transform_config, TransformConfig):
        # Ensure it passes its own validation and convert to dict
        cfg = TransformConfig.model_validate(transform_config).model_dump()
        return cfg

    # 3) dict -> validate via TransformConfig
    if isinstance(transform_config, dict):
        try:
            cfg = TransformConfig.model_validate(transform_config).model_dump()
        except Exception as e:
            raise ValueError(f"Invalid transform_config dict: {e}") from e
        return cfg

    # 4) str -> shorthand for transform_type
    if isinstance(transform_config, str):
        key = transform_config.strip()
        if not key:
            return {}

        if key not in TRANSFORMS:
            raise ValueError(
                f"Invalid transform_config string: {key!r}. "
                f"Expected one of {sorted(TRANSFORMS.keys())}."
            )

        cfg_dict = {"transform_type": key, "quant_scheme": scheme}

        try:
            cfg = TransformConfig.model_validate(cfg_dict).model_dump()
        except Exception as e:
            raise ValueError(
                f"transform_config built from string {key!r} is invalid for TransformConfig: {e}"
            ) from e

        return cfg

    raise TypeError(
        "transform_config must be one of: None, dict, TransformConfig, or str "
        f"(got {type(transform_config).__name__})"
    )
