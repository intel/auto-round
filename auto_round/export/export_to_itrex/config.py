#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configs for Autoround quantization."""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import torch
from transformers import PretrainedConfig

from auto_round.utils import convert_dtype_str2torch, convert_dtype_torch2str, logger

QUANT_CONFIG = "quantize_config.json"


class QuantConfig(PretrainedConfig):
    """A brief quantization configuration for reference when performing model de-quantization."""

    def __init__(
        self,
        bits=4,
        scale_dtype="fp32",
        group_size=128,
        sym=False,
        quant_method="autoround",
        model_name_or_path=None,
        model_file_base_name="model",
        enable_minmax_tuning=True,
        iters=1000,
        lr=0.001,
        minmax_lr=0.001,
        enable_quanted_input=True,
        compute_dtype=None,
        **kwargs,
    ):
        self.bits = bits
        self.group_size = group_size
        self.scale_dtype = convert_dtype_torch2str(scale_dtype)
        self.sym = sym
        self.quant_method = quant_method
        self.model_name_or_path = model_name_or_path
        self.model_file_base_name = model_file_base_name
        self.enable_minmax_tuning = enable_minmax_tuning
        self.iters = iters
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.enable_quanted_input = enable_quanted_input
        self.compute_dtype = convert_dtype_torch2str(compute_dtype)

        if "export_to_xpu" not in kwargs or not kwargs["export_to_xpu"]:
            ### Redundant parameters, will be removed later. ###
            self.damp_percent = 0.01
            self.desc_act = False
            self.true_sequential = False
            self.quant_method = "gptq"
        else:
            ### XPU special parameters. ###
            self.weight_dtype = "int4_fullrange"  # Due to ipex format limitations. Actually, it's int4_clip.

    def post_init(self):
        r"""Safety checker for CPU that arguments are correct
        also replaces some NoneType arguments with their default values."""

        if self.scale_dtype not in ["fp32", "fp16", "bf16"]:
            raise ValueError("scale_dtype must be 'fp32', 'fp16' or 'bf16'.")

        if self.group_size not in [-1, 32, 128]:
            raise ValueError("group_size must be an integer in [-1, 32, 128]")

    def post_init_xpu(self):
        r"""
        Safety checker for XPU that arguments are correct
        - also replaces some NoneType arguments with their default values.
        """

        if self.compute_dtype is not None and self.compute_dtype not in ["fp16"]:
            raise ValueError("compute_dtype must be 'fp16'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp16"

        if self.bits is None:
            self.bits = 4
        elif self.bits not in [4]:
            raise ValueError(f"Only support quantization to [4] bits but found {self.bits}")

        if self.weight_dtype is None:
            self.weight_dtype = "int4_fullrange"

        elif self.weight_dtype not in [
            "int4_fullrange",
        ]:
            raise ValueError(f"weight_dtype must be a string in 'int4_fullrange', but get {self.weight_dtype}.")

        if self.scale_dtype is not None and self.scale_dtype not in ["fp16"]:
            raise ValueError("scale_dtype must be a string in 'fp16'")
        elif self.scale_dtype is None:
            self.scale_dtype = "fp16"

        # if not isinstance(self.use_double_quant, bool):
        #     raise ValueError("use_double_quant must be a boolean")

        # if self.use_double_quant and not isinstance(self.double_quant_dtype, str):
        #     raise ValueError("double_quant_dtype must be a string")

        # if self.use_double_quant and not isinstance(self.scale_dtype, str):
        #     raise ValueError("scale_dtype must be a string")

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if self.sym is not True:
            raise ValueError("asym is not support, only support 'sym' now!")
        self.use_neural_speed = False

    def quantization_method(self):
        r"""This method returns the quantization method used for the model."""
        return self.quant_method

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """Instantiates a [`QuantConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.
        Returns:
            [`QuantConfig`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file_path, return_unused_kwargs, **kwargs):
        with open(json_file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, return_unused_kwargs, **kwargs)

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary.

        Returns:
        `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """

        output = copy.deepcopy(self.__dict__)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantConfig()`
                is serialized to JSON string.
        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """Removes all attributes from config which correspond to the default config attributes
        for better readability and serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = QuantConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise OSError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, QUANT_CONFIG)

        self.to_json_file(output_config_file, use_diff=False)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token", None),
            )

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return super().get_config_dict(pretrained_model_name_or_path, _configuration_file=QUANT_CONFIG, **kwargs)

    def remove_redundant_parameters(self):
        remove_parameters = [
            "calib_dataloader",
            "dataset",
            "scheme",
            "tokenizer",
            "use_neural_speed",
            "enable_quanted_input",
            "layer_wise",
            "nsamples",
            "lr",
            "minmax_lr",
            "iters",
            "enable_quanted_input",
            "model_file_base_name",
            "enable_minmax_tuning",
            "model_name_or_path",
        ]
        for parameter in remove_parameters:
            if hasattr(self, parameter):
                delattr(self, parameter)
