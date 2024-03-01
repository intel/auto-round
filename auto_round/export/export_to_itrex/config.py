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

from auto_round.utils import logger

QUANT_CONFIG = "quantize_config.json"


class QuantConfig(PretrainedConfig):
    """A brief quantization configuration for reference when performing model dequantization."""

    def __init__(
        self,
        bits=4,
        scale_dtype="torch.float32",
        group_size=128,
        sym=False,
        quant_method="autoround",
        model_name_or_path=None,
        model_file_base_name="model",
        enable_minmax_tuning=True,
        iters=1000,
        lr=0.001,
        minmax_lr=0.001,
        use_quant_input=True,
        **kwargs,
    ):
        self.bits = bits
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.sym = sym
        self.quant_method = quant_method
        self.model_name_or_path = model_name_or_path
        self.model_file_base_name = model_file_base_name
        self.enable_minmax_tuning = enable_minmax_tuning
        self.iters = iters
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.use_quant_input = use_quant_input

        ### Redundant parameters, will be removed later. ###
        self.damp_percent = 0.01
        self.desc_act = False
        self.true_sequential = False
        self.quant_method = "gptq"

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.scale_dtype not in ["torch.float32", "torch.float16", "torch.bfloat16"]:
            raise ValueError("scale_dtype must be 'fp32', 'fp16' or 'bf16'.")

        if self.group_size not in [-1, 32, 128]:
            raise ValueError("group_size must be an integer in [-1, 32, 128]")

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
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

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
