FROM python:3.9

USER root 
RUN mkdir -p /home/jovyan && mkdir -p /data && chmod 777 /data

WORKDIR /home/jovyan

COPY test_retry.py /home/jovyan/

CMD [ "python3", "test_retry.py" ]
