#!/bin/bash
# Install extra requirements
pip3 install ninja
pip3 install xformers==0.0.23.post1
pip3 install bitsandbytes==0.41.3
pip3 install flash-attn==2.5.6 --no-build-isolation
