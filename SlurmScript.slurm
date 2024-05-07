#!/bin/bash
#SBATCH --job-name=bin_insp
#SBATCH -t 0-01:00 # Runtime in D-HH:MM 

#SBATCH --mem=10G
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1 

#SBATCH --threads-per-core=1
#SBATCH -p astro2_gpu

#SBATCH -o /groups/astro/davidon/sailfish/Outputs/acc10_%j.out # Standard out       
#SBATCH -e /groups/astro/davidon/sailfish/Outputs/acc10_%j.err # Standard err        

module load astro


## Retrograde Isothermal Testing

python3 ~/sailfish/bin/sailfish binary-inspiral -n 1500 -o /lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e01 -c 10 -t 0.1 -e 500 -g --patches 3  --model which_diagnostics="david" domain_radius=10. retrograde=True sink_model="torque_free" init_eccentricity=0.1 



##  Restart From directory '/lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test'

#python3 ~/sailfish/bin/sailfish /lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e0 -n 1500 -o /lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e0 -c 10 -t 0.1 -e 1000 -g --patches 3 --restart-dir --model which_diagnostics="david" domain_radius=10. retrograde=True sink_model="torque_free" sink_rate=1.


#python3 ~/sailfish/bin/sailfish binary-inspiral -n 3000 -o /lustre/astro/davidon/Storage/sfish-test --patches 4 -g -c 1 -t 0.1 -e 5  --model which_diagnostics="david"
