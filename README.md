# 💡 Samsung Computer Engineering Challenge 💡
- Team name: 서교수네 라마농장
- Affiliation: Computer Systems Lab. (CSL), Sungkyunkwan University
- Members: Junyeol Yu, Gwanjong Park, Osama Khan
- E-mail: junyeol.yu@skku.edu, jesj74@g.skku.edu, khan980@g.skku.edu
- Challenge site: [[link]](https://cechallenge.github.io/)
<br>

# Llama Ranch: Batching scheme for improving inference throughput

>Due to the nature of the language model, sequences of various sizes may come into the input of inference.
>When batching multiple sequences to improve inference throughput, the length of the batch sequence for this is the same based on the longest sequence with padding the remaining sequences.

>Depending on the combination of the sequences that make up the batch, padding process would be unnecessarily performed. When the input sequence lengths of the dataset are arranged in descending order, the cost of padding for batch processing increases monotonically when inference is performed sequentially.
>In addition, the runtime gain obtained by increasing the size of the batch decreases as the batch size increases.

Given these facts, the goal is to determine batch sizes that can minimize the computational overhead of zero-padding 

## Features
- Organizing datasets based on sequence length to minimize the need for excessive padding
- Determining batch size to fully utilize available GPU memory and maximize throughput

## Build

## Quick Start

