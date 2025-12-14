FROM nvcr.io/nvidia/pytorch:25.08-py3

RUN apt-get update
RUN apt-get install -y python3-tk

RUN python -m pip install --upgrade pip

RUN pip install torchopt torchrl gpytorch tensordict-nightly pytorch-ignite
RUN pip install jupyter notebook matplotlib typing jaxtyping
RUN pip install numpy scipy gymnasium black flake8 pygame isort
