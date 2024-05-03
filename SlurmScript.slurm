#!/bin/bash
#SBATCH --job-name=bin_insp
#SBATCH -t 0-01:00 # Runtime in D-HH:MM 

#SBATCH --mem=10G
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1                                                 

#SBATCH --threads-per-core=1
#SBATCH -p astro3_short

#SBATCH -o /groups/astro/davidon/GDF_Python/Hyperbolic/acc10_%j.out # Standard out       
#SBATCH -e /groups/astro/davidon/GDF_Python/Hyperbolic/acc10_%j.err # Standard err        

module load astro

python3 ~/sailfish/bin/sailfish binary-inspiral -n 3000 -o /lustre/astro/davidon/Storage/sfish-test --patches 4 -g -c 1 -t 0.1 -e 5  --model which_diagnostics="david"
