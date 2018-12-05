ARG TF_TAG=latest-gpu-py3
FROM tensorflow/tensorflow:$TF_TAG
ADD . /home/Doom/DoomPCGML
WORKDIR /home/Doom/DoomPCGML
RUN apt-get update
RUN apt-get install -y vim
# Installing required packages (some are already present in tensorflow distribution)
RUN pip install request dicttoxml scikit-image networkx scikit-learn seaborn beautifulsoup4 
# Exposing port 6006 to host for tensorboard
EXPOSE 6006
CMD bash -C 'extract.sh'; '/bin/bash'
