FROM --platform=amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

USER root 
RUN mkdir -p /home/jovyan && mkdir -p /data && chmod 777 /data

WORKDIR /home/jovyan

COPY . /home/jovyan/

RUN chmod +x main.sh

RUN cd /home/jovyan && pip install -r requirements.txt

ENTRYPOINT ["/home/jovyan/main.sh"]