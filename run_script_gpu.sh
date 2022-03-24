#!/bin/env bash
# oarsub -l /core=2/gpunum=1 -S "./$*" 

# Oarsub parameters (150 hours)
#OAR -l /core=2/gpunum=1,walltime=150
#OAR -p gpu='YES' and gpucapability>='5.2'

module load cuda/10.0
module load cudnn/7.4-cuda-10.0

echo $*

# Run script
python3 $*
