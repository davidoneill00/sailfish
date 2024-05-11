#!/bin/bash
#SBATCH --job-name=retro_e
#SBATCH -t 0-12:00 # Runtime in D-HH:MM 

#SBATCH --mem=10G
####SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=10G

#SBATCH --threads-per-core=1
#SBATCH -p astro3_short

#SBATCH -o /groups/astro/davidon/sailfish/Outputs/acc10_%j.out # Standard out       
#SBATCH -e /groups/astro/davidon/sailfish/Outputs/acc10_%j.err # Standard err        

module load astro




######### Retrograde Isothermal Testing

python3 ~/sailfish/bin/sailfish binary-inspiral -n 3000 -o /lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e1 -c 10 -t 0.1 -e 500 -p --new-timestep-cadence 10 --model which_diagnostics="david" domain_radius=10. retrograde=True sink_model="torque_free" init_eccentricity=0.1 softening_length=0.05 sink_rate=1. sink_radius=0.05








#########  Restart From directory '/lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test'

#####python3 ~/sailfish/bin/sailfish /lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e0 -n 3000 -o /lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e0 -c 10 -t 0.1 -e 500 -p --restart-dir --model which_diagnostics="david" domain_radius=10. retrograde=True sink_model="torque_free" sink_rate=1. sink_radius=0.05 softening_length=0.05 init_eccentricity=0.


