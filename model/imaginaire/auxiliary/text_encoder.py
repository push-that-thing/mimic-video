# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import ClassVar, TypeAlias

import attrs
import torch
import transformers
import transformers.utils.logging
from torch import nn
from transformers import T5EncoderModel, T5TokenizerFast

from imaginaire.constants import T5_MODEL_DIR, TextEncoderClass
from imaginaire.utils import log

transformers.utils.logging.set_verbosity_error()

NUM_EMBEDDING_PADDING_TOKENS = 512


@attrs.define(slots=False)
class CosmosT5TextEncoderConfig:
    """
    Config for the T5 text encoder model
    """

    CKPT_PATH: ClassVar[str] = T5_MODEL_DIR
    NUM_TOKENS: ClassVar[int] = 512
    EMBED_DIM: ClassVar[int] = 1024

    ckpt_path: str = CKPT_PATH
    num_tokens: int = NUM_TOKENS
    embed_dim: int = EMBED_DIM


class CosmosT5TextEncoder(nn.Module):
    """Handles T5 text encoding operations."""

    def __init__(
        self,
        config: CosmosT5TextEncoderConfig,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
    ):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: The device to use for computations.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.tokenizer = T5TokenizerFast.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype)
        self.text_encoder = T5EncoderModel.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype).to(device)
        self.text_encoder.eval()

        log.info("T5 Text encoder model instantiated")

    @property
    def model(self):
        return self

    @torch.inference_mode()
    def encode_prompts(self, prompts: str | list[str], max_length: int | None = None, return_mask: bool = False):
        """Encodes text prompts into hidden state representations.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to num_tokens.
            return_mask: If True, returns the attention mask along with encoded text.
                Defaults to False.

        Returns:
            If return_mask is False:
                torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).
            If return_mask is True:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - Encoded text embeddings of shape (batch_size, max_length, hidden_size)
                    - Attention mask of shape (batch_size, max_length) as boolean tensor

        Raises:
            ValueError: If the input prompts list is empty.
        """

        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("The input prompt list is empty.")
        if max_length is None:
            max_length = self.config.num_tokens

        batch_encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attn_mask = batch_encoding.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        if return_mask:
            return encoded_text, attn_mask.bool()
        return encoded_text


@attrs.define(slots=False)
class CosmosTextEncoderConfig:
    NUM_TOKENS: ClassVar[int] = CosmosT5TextEncoderConfig.NUM_TOKENS
    EMBED_DIM: ClassVar[int] = CosmosT5TextEncoderConfig.EMBED_DIM

    cls: TextEncoderClass = TextEncoderClass.T5
    t5: CosmosT5TextEncoderConfig = attrs.field(factory=CosmosT5TextEncoderConfig)


CosmosTextEncoder: TypeAlias = CosmosT5TextEncoder
