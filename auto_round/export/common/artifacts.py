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

from auto_round.export.common.serialization import save_pretrained_artifact


def save_artifacts(output_dir, tokenizer=None, processor=None, image_processor=None):
    """Save supported pretrained artifacts when an output directory was requested."""
    if output_dir is None:
        return
    for name, artifact in (("tokenizer", tokenizer), ("processor", processor), ("image_processor", image_processor)):
        save_pretrained_artifact(artifact, output_dir, artifact_name=name)
