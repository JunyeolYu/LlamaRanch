# Base image is
ARG BASE_IMAGE=junyeolyu/torch:2.0.1

FROM ${BASE_IMAGE}

WORKDIR /workspace/LlamaRanch
COPY . ${WORKDIR}

# Checkout two submoudle repository
#RUN git submodule update --init

RUN python -m pip install --upgrade pip
#RUN git clone https://github.com/JunyeolYu/FasterTransformer_.git
#RUN git clone https://github.com/JunyeolYu/llama_v1.git

# Install torch-2.0
#RUN pip3 uninstall -y torch
#RUN pip3 install torch-2.0.0-cp310-cp310-linux_x86_64.whl

# Install requirements
RUN python -m pip install jax jaxlib datasets sentencepiece transformers fairscale fire numpysocket
RUN apt-get update
RUN apt-get install bc
# Install Meta
#WORKDIR /workspace/src/Meta
#RUN python -m pip install -e .

WORKDIR /workspace
