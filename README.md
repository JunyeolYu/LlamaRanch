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

# Quick Start
## Building the image
With the given `Dockerfile`, build your testbed image. This image is based on `junyeolyu/torch:2.0.1`. Therefore, it will take a while for pull the base image.

``` bash
$ docker build -t cechallenge .
```

## Running the testbed
Assuming the model repository is available in `/path/to/model`.
Use the following command to run the container for evaluation.
``` bash
$ docker run --rm --gpus all --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=134217728 -v /path/to/model:/model -it cechallenge bash
```
The entry point is `/worksapce`.

## FasterTransformer Evaluation
We need to build the library before evaluation. DSM should be set to 70 for the Tesla V100.
``` bash
cd /workspace/src/FasterTransformer
mkdir build && cd build
git submodule init && git submodule update
# These packages are already installed during the image building
pip3 install fire jax jaxlib transformers datasets sentencepiece numpysocket

CUDAFLAGS="-include stdio.h" cmake -DSM=70 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -D PYTHON_PATH=/usr/bin/python3 ..
make -j$(nproc)
```
Then, you can run the evaluation script.
``` bash
cd /workspace/src/FasterTransformer/examples/pytorch/llama
FMHA_ENABLE=ON ./exec_evaluation.sh
#mpirun -n 4 --allow-run-as-root python llama_example.py --output_len 1 --pipeline_para_size 4 --ckpt_path /model/$MODEL_PATH --tokenizer_path /model/$HF_TOKENIZER_PATH --lib_path /workspace/src/FT/build/lib/libth_transformer.so
``` 

For more details, see [FasterTransformer:Setup](https://github.com/JunyeolYu/LlamaRanch/tree/main/src/FT#setup)

## Meta Evaluation (1st Round)
The provided `example.py` can be run on a single or multiple GPUs with torchrun and will output completions for two pre-defined prompts.

In this repository, a 4-GPU inference setting is considered.
``` bash
cd /workspace/src/Meta
# Install this repository, if you need
pip install -e .
torchrun --nproc_per_node 4 example.py --ckpt_dir /model/$TARGET_FOLDER --tokenizer_path /model/$TARGET_FOLDER/tokenizer.model
```

- `example.py` will produce `Meta 4_bins`
- `example_opt.py` will produce `Meta Greedy`

For testing vanilla, change the branch main to `vanilla`.
``` bash
cd /workspace/src/Meta
git switch vanilla
pip install -e .
torchrun --nproc_per_node 4 example.py --ckpt_dir /model/$TARGET_FOLDER --tokenizer_path /model/$TARGET_FOLDER/tokenizer.model
```
- In this branch, `example.py` generates `Meta Vanilla`
