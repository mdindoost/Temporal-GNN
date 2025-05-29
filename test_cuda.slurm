#!/bin/bash
#SBATCH --job-name=test-cuda
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=bader
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=output_test_cuda.log
#SBATCH --error=error_test_cuda.log

module purge
module load easybuild
module load foss/2022b
module load Python/3.10.8
source ~/temporal-gnn-env/bin/activate

python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

