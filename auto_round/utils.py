from typing import Dict, List
from auto_round.utils import to_standard_regex, matches_any_regex


def generate_ignore_regex_list(regex_config: Dict[str, Dict], layer_config: Dict[str, Dict]) -> List[str]:
    """
    Generate ignore regex list for llm_compressor based on regex_config and layer_config.

    Rules:
    1. Any layer in regex_config with bits >= 16 is ignored.
    2. Any layer in layer_config with bits >= 16 is ignored if not already included.
    3. Output regex patterns are normalized for llm_compressor ('re:...' style).
    
    Args:
        regex_config (Dict[str, Dict]): dynamic quantization config
        layer_config (Dict[str, Dict]): layer-wise quantization config

    Returns:
        List[str]: List of regex patterns to ignore during quantization.
    """
    prefix = "re:"
    ignore_regex: List[str] = []

    # Step 1: Add regex_config keys with bits >= 16
    for key, cfg in regex_config.items():
        bits = cfg.get("bits")
        if bits > 8:
            ignore_regex.append(prefix + to_standard_regex(key))

    # Step 2: Add all full named layer from layer_config if bits >= 16
    for key, cfg in layer_config.items():
        bits = cfg.get("bits")
        if bits > 8:
            ignore_regex.append(key)

    return ignore_regex