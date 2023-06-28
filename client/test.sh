#!/bin/sh
conda activate asr
paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8091 --input data/zh.wav
