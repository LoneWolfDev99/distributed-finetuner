FROM --platform=amd64 pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

USER root 
RUN mkdir -p /home/jovyan && mkdir -p /data && chmod 777 /data

WORKDIR /home/jovyan

COPY requirements.txt /home/jovyan/

RUN cd /home/jovyan && pip install -r requirements.txt

COPY gemma.py /home/jovyan/gemma.py
COPY helpers.py /home/jovyan/helpers.py

CMD [ "python3", "gemma.py" ]