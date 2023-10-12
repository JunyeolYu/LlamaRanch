# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import inspect
import argparse
import dataclasses
import json
import os
import pathlib
import typing

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

import time
str_type_map = {"fp32": torch.float32, "fp16": torch.float16}

class LlamaWeights(object):
    def __init__(self, 
                 head_num, size_per_head, inter_size, layer_num, vocab_size, 
                 max_seq_len, tensor_para_size, pipeline_para_size, use_gptj_residual, 
                 inference_data_type: str = "fp16",
                 weights_data_type: np.dtype = np.float32):
        assert(head_num % tensor_para_size == 0)

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        self.use_gptj_residual = use_gptj_residual

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = inter_size // tensor_para_size

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units

        if isinstance(weights_data_type, str):
            try:
                weights_data_type = {
                    "fp16": np.float16,
                    "fp32": np.float32,
                    "float16": np.float16,
                    "float32": np.float32,
                }[weights_data_type]
            except KeyError:
                raise ValueError(f"Don't know how to interpret weights_data_type: {weights_data_type}")

        assert weights_data_type in [np.float32, np.float16]
        self.weights_data_type = weights_data_type
        self.inference_data_type = str_type_map[inference_data_type]

        self.w = []
        # Transformer blocks
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # 0 pre_layernorm_weights.beta
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # 1 pre_layernorm_weights.gamma
        self.w.extend([torch.zeros(global_hidden_units, local_hidden_units * 3, dtype=self.inference_data_type)] * layer_num)   # 2 self_attention_weights.query_weight.kernel
        self.w.extend([torch.zeros(local_hidden_units * 3, dtype=self.inference_data_type)] * layer_num)                        # 3 self_attention_weights.query_weight.bias
        self.w.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype=self.inference_data_type)] * layer_num)       # 4 self_attention_weights.attention_output_weight.kernel
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type) if not use_gptj_residual else torch.empty(0)] * layer_num) #  5  self_attention_weights.attention_output_weight.bias
        self.w.extend([torch.zeros(global_hidden_units, local_inter_size, dtype=self.inference_data_type)] * layer_num)         # 6 ffn_weights.intermediate_weight.kernel
        self.w.extend([torch.zeros(local_inter_size, dtype=self.inference_data_type)] * layer_num)                              # 7 ffn_weights.intermediate_weight2.kernel
        self.w.extend([torch.zeros(global_hidden_units, local_inter_size, dtype=self.inference_data_type)] * layer_num)         # 8 ffn_weights.intermediate_weight.kernel
        self.w.extend([torch.zeros(local_inter_size, dtype=self.inference_data_type)] * layer_num)                              # 9 ffn_weights.intermediate_weight2.kernel

        self.w.extend([torch.zeros(local_inter_size, global_hidden_units, dtype=self.inference_data_type)] * layer_num)         # 10 ffn_weights.output_weight.kernel
        self.w.extend([torch.zeros(local_hidden_units, dtype=self.inference_data_type)] * layer_num)                            # 11 ffn_weights.output_weight.bias 
        self.w.extend([torch.zeros(local_hidden_units, dtype=self.inference_data_type)] * layer_num)                            # 12 post_attention_layernorm_weights.beta
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # 13 post_attention_layernorm_weights.gamma

        # After Transformer blocks
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype=self.inference_data_type))                             # pre_decoder_embedding_table
        self.w.append(torch.zeros(global_hidden_units, dtype=self.inference_data_type))                                         # post_decoder_layernorm.beta
        self.w.append(torch.zeros(global_hidden_units, dtype=self.inference_data_type))                                         # post_decoder_layernorm.gamma
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype=self.inference_data_type))                             # post_decoder_embedding.kernel

        # Initialization
        # self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=0.01))

    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for i in range(len(self.w)):
            if isinstance(self.w[i], list):
                for j in range(len(self.w[i])):
                    self.w[i][j] = func(self.w[i][j])
            else:
                self.w[i] = func(self.w[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False
        w = []
        # Load
        def is_load(i):
            return i >= self.layers_per_device * pipeline_para_rank and i < self.layers_per_device * (pipeline_para_rank + 1)

        print("\n\n========== LOAD ==========")
        print(f"tensor {tensor_para_rank}, pipeline {pipeline_para_rank}")
        print("==========================")

        file_names = [
                      None,   # 0
                      "input_layernorm.weight",  # 1
                      "attention.query_key_value.weight.%d" % tensor_para_rank, # 2
                      None,  # 3
                      "attention.dense.weight.%d" % tensor_para_rank, # 4
                      None,  # 5
                      "mlp.gate_proj.weight.%d" % tensor_para_rank,   # 6
                      None,  # 7
                      "mlp.up_proj.weight.%d" % tensor_para_rank,     # 8
                      None,  # 9
                      "mlp.down_proj.weight.%d" % tensor_para_rank,   # 10
                      None,  # 11
                      None,  # 12
                      "post_attention_layernorm.weight"               # 13
                      ]

        for file_name in file_names:
            for i in range(self.layer_num):
                if file_name is not None and is_load(i):
                    w.append(torch.from_numpy(np.fromfile(
                                "%s/model.layers.%d.%s.bin" % (ckpt_path, i, file_name),
                                dtype=self.weights_data_type)).to(self.inference_data_type))
                else:
                    w.append(torch.empty(0).to(self.inference_data_type))

        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.weight.bin", dtype=self.weights_data_type)).to(self.inference_data_type))
        w.append(torch.zeros(self.global_hidden_units, dtype=self.inference_data_type))                                     
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.bin", dtype=self.weights_data_type)).to(self.inference_data_type))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.lm_head.weight.bin", dtype=self.weights_data_type)).to(self.inference_data_type))

        total = len(w) - 4 # For non-transformer layers
        try:
            # For transformer layers
            for i in range(total):
                layer = i%self.layer_num # single gpu must load a portion of the layers
                if w[i].nelement() > 0:
                    self.w[i] = w[i].reshape(self.w[i].shape)
                elif not is_load(layer):
                    self.w[i] = w[i]
            
            # For non-transformer layers
            for i in range(total,len(w)):
                self.w[i] = w[i].reshape(self.w[i].shape)
                
        except RuntimeError:
            raise RuntimeError(
                f"head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training "
                f"(idx: {i} expected shape: {self.w[i].shape} got shape: {w[i].shape})."
            )
        return True


class Llama(nn.Module):
    def __init__(self,
                 head_num, size_per_head, inter_size,
                 vocab_size, rotary_embedding_dim, layernorm_eps,
                 start_id, end_id, layer_num,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 use_gptj_residual,
                 lib_path,
                 inference_data_type: str = "fp16",
                 weights_data_type: np.dtype = np.float32):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.inter_size = inter_size
        self.vocab_size = vocab_size
        self.rotary_embedding_dim = rotary_embedding_dim
        self.layernorm_eps = layernorm_eps
        self.start_id = start_id
        self.end_id = end_id
        self.max_seq_len = max_seq_len
        self.layer_num = layer_num
        self.use_gptj_residual = use_gptj_residual

        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.build_model = False
        self.weights_data_type = weights_data_type
        self.inference_data_type = inference_data_type

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(lib_path))
        
        # Prepare weights
        self.weights = LlamaWeights(head_num, size_per_head, inter_size, layer_num, vocab_size,
                                      max_seq_len, tensor_para_size, pipeline_para_size, use_gptj_residual,
                                      weights_data_type=weights_data_type, inference_data_type=inference_data_type)
        
        # Prepare for tensor/pipeline parallel
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        print("world size is " + str(world_size))
        print(str(tensor_para_size * pipeline_para_size) + f"<<- TP({tensor_para_size}) * PP({pipeline_para_size})")
        assert world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        # Create and copy model to the device.
        # self.cuda()

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        self.half()
        return is_load

    def half(self):
        self.weights._map(lambda w: w.half())
        self.cuda()

    def cuda(self):
        self.weights._map(lambda w: w.cuda(self.device))

        if self.build_model:
            del self.model
            self.build_model = False
        self.model = torch.classes.FasterTransformer.LlamaOp(self.head_num, self.size_per_head, self.inter_size,
                                                               self.layer_num, self.vocab_size, self.rotary_embedding_dim, self.layernorm_eps,
                                                               self.start_id, self.end_id, self.tensor_para_size, self.pipeline_para_size,
                                                               self.max_seq_len, self.use_gptj_residual, self.weights.w)
        self.build_model = True
        torch.cuda.empty_cache()

    def forward(self,
                start_ids: torch.Tensor,
                start_lengths: torch.Tensor,
                output_len,
                beam_width = 1,
                top_k: torch.Tensor = None,
                top_p: torch.Tensor = None,
                beam_search_diversity_rate: torch.Tensor = None,
                temperature: torch.Tensor = None,
                len_penalty: torch.Tensor = None,
                repetition_penalty: torch.Tensor = None,
                random_seed: torch.Tensor = None,
                return_output_length = False,
                return_cum_log_probs=0):
        if not self.build_model:
            self.cuda()
            torch.cuda.empty_cache()  # clean cache for model weight preprocessing
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."

        # Inputs to device
        input_ids = start_ids.cuda(self.device)
        input_lengths = start_lengths.cuda(self.device)
        
        # Bug fixed. Third argument must be <int>
        outlen = output_len
        if type(output_len) != int: outlen = int(output_len[0])
        
        outputs = self.model.forward(input_ids,
                                     input_lengths,
                                     outlen, #output_len
                                     beam_width, # optional, can be None
                                     top_k, # optional, can be None
                                     top_p, # optional, can be None
                                     beam_search_diversity_rate, # optional, can be None
                                     temperature, # optional, can be None
                                     len_penalty, # optional, can be None
                                     repetition_penalty, # optional, can be None
                                     random_seed, # optional, can be None
                                     return_cum_log_probs,# optional, can be None
                                     1) # optional, output_log_probs

        if return_cum_log_probs == 0:
            output_ids, output_lengths, output_log_probs = outputs
        else:
            output_ids, output_lengths, output_cum_log_probs, output_log_probs = outputs
            
        if return_output_length:
            if return_cum_log_probs > 0:
                return output_ids, output_lengths, output_cum_log_probs, output_log_probs
            else:
                return output_ids, output_lengths, output_log_probs
        else:
            return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
