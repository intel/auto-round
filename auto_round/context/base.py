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

from auto_round.logger import logger


class AutoSkipInitMeta(type):

    def __new__(mcs, name, bases, namespace):
        if "__init__" in namespace:
            original_init = namespace["__init__"]

            def wrapped_init(self, *args, **kwargs):
                if getattr(self, "_singleton_skip_init", False):
                    return
                original_init(self, *args, **kwargs)
                self._singleton_skip_init = True

            namespace["__init__"] = wrapped_init

        namespace["_instances"] = {}
        return super().__new__(mcs, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = cls.__new__(cls, *args, **kwargs)
            cls._instances[cls] = instance
            instance.__init__(*args, **kwargs)

        return cls._instances[cls]


class BaseContext(metaclass=AutoSkipInitMeta):
    _instances = {}
    _internal_attrs = {"_context_state", "_singleton_skip_init"}

    def __init__(self):
        if "_context_state" not in self.__dict__:
            object.__setattr__(self, "_context_state", {})
        logger.info(f"{self.__class__.__name__} context initialized.")

    def __getattr__(self, name):
        context_state = object.__getattribute__(self, "_context_state")
        if name in context_state:
            return context_state[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_") or name in self._internal_attrs:
            object.__setattr__(self, name, value)
            return
        self._set_context_attr(name, value)

    def _set_context_attr(self, name, value):
        context_state = object.__getattribute__(self, "_context_state")
        context_state[name] = value

    def _update_context_attrs(self, **kwargs):
        for name, value in kwargs.items():
            self._set_context_attr(name, value)

    @classmethod
    def get_context(cls):
        assert cls in cls._instances, f"{cls.__name__} context has not been created yet."
        return cls._instances.get(cls)

    @classmethod
    def create_context(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def reset_context(cls):
        cls._instances.pop(cls, None)
