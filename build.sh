#!/usr/bin/env bash
# exit on error
set -o errexit

# Install native dependencies for dlib and opencv-python
apt-get update && apt-get install -y build-essential cmake

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
