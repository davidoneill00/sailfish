#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

#SBATCH --gres=gpu:v100:4
#SBATCH --mem=5GB


#SBATCH -o /home/do2364/sailfish/Outputs/acc10_%j.out # Standard out       
#SBATCH -e /home/do2364/sailfish/Outputs/acc10_%j.err # Standard err 

# Load Singularity module if required
# module load singularity

# Specify the command to execute within the Singularity container
singularity exec --nv \
  --overlay /scratch/do2364@/pytorch-example/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
  /bin/bash -c \
  "source /ext3/env.sh && python3 ~/sailfish/bin/sailfish binary-inspiral -n 3000 -o /scratch/do2364@/sfish-test/Retrograde_Circular_Iso_Test/e0 -c 10 -t 0.1 -e 500 -g --patches 1 --new-timestep-cadence 10 --patches=4 --model which_diagnostics="david" domain_radius=10. retrograde=True sink_model="torque_free" init_eccentricity=0. softening_length=0.05 sink_rate=1. sink_radius=0.05 init_separation_rg=100. inspiral_start_time=500. init_eccentricity=0. integration_timestep=0.001 GM=1. mass_ratio=1."

