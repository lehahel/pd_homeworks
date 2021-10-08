#!/bin/bash

# -- DO NOT EXECUTE MANUALLY -- #

rm -rf env

python3 -m venv env

env/bin/pip3 install -r requirements.txt
