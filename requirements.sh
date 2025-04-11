#!/bin/bash

# Install required packages
pip install git+https://github.com/openai/clip
pip install ftfy regex tqdm
pip install ipyplot
pip install open-clip-torch

# Install from requirements.txt if it exists
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi
