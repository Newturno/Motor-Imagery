#!/bin/bash
echo "Hello, world!"
chmod +x run.sh
cd /Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python
conda deactivate
conda activate mi
python Experiment.py