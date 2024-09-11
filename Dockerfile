FROM nvcr.io/nvidia/pytorch:24.08-py3

COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt
RUN pip install segyio timm==0.4.12

WORKDIR /workspace/