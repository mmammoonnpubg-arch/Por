#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q fal-client pillow
echo "put your key:  export FAL_KEY=xxxx"
python barca_girl.py --in IMG_7072.png --out ./out --video