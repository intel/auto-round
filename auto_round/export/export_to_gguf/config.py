GGUF_CONFIG = {}

GGUF_CONFIG["gguf:q4_0"] = {"bits": 4, "act_bits": 16, "group_size": 32, "asym": False}

GGUF_CONFIG["gguf:q4_1"] = {"bits": 4, "act_bits": 16, "group_size": 32, "asym": True}

GGUF_CONFIG["gguf:q4_k"] = GGUF_CONFIG["gguf:q4_k_S"] = {
    "bits": 4,
    "act_bits": 16,
    "group_size": 32,
    "asym": True,
    "data_type": "int_asym_dq"
}
