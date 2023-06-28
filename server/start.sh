#!/bin/sh
conda activate asr
paddlespeech_server start --config_file ./conf/application.yaml --log_file logs/server.log
