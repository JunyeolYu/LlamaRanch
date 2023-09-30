# 💡 Samsung Computer Engineering Challenge 💡
- Team name: 서교수네 라마농장
- Affiliation: Computer Systems Lab. (CSL), Sungkyunkwan University
- Members: Junyeol Yu, Gwanjong Park, Khan Osama
- E-mail: junyeol.yu@skku.edu, jesj74@g.skku.edu, khan980@g.skku.edu
- Challenge site: [[link]](https://cechallenge.github.io/)
<br>

# FasterTransformer

This repository is forked from the [repository](https://github.com/vitrun/FasterTransformer) providing a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA.

It aims to efficiently perform LLaMA-30B model inference using Fastertransformer as an inference engine for zero-shot Hellaswag task processing.

>In NLP, encoder and decoder are two important components, with the transformer layer becoming a popular architecture for both components. FasterTransformer implements a highly optimized transformer layer for both the encoder and decoder for inference. On Volta, Turing and Ampere GPUs, the computing power of Tensor Cores are used automatically when the precision of the data and weights are FP16. FasterTransformer is built on top of CUDA, cuBLAS, cuBLASLt and C++. We provide at least one API of the following frameworks: TensorFlow, PyTorch and Triton backend. Users can integrate FasterTransformer into these frameworks directly. For supporting frameworks, we also provide example codes to demonstrate how to use, and show the performance on these frameworks.

## Features
- Supporting for FP-16 LLaMA-30B inference with PyTorch framework based on [this branch](https://github.com/vitrun/FasterTransformer/tree/llama_torch)
- Supporting pipeline parallelism for the target model
- Suited for Hellaswag task (zero-shot, sequence classification)
- Removed auto-regressive decode (generation) module
- Calculating prediction class from prefill module's output
- Modified to prevent unnecessary invocation of some of the existing implementations that are not related to performing the target sequence classification task
- Modified to prevent unnecessary loading of layer weights on each GPU

## Setup

## Inference
