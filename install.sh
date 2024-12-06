#!/bin/bash

# wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
# wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512
# tar -xzf elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
# sudo chown -R daemon:daemon elasticsearch-7.9.2/
# shasum -a 512 -c elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512

mkdir -p datasets
pip install -r requirements.txt
python setup.py
ls datasets/msmarco/




