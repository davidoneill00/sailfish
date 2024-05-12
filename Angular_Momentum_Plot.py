import sys 
import numpy as np 
import pickle as pk
import matplotlib.pyplot as plt 
import sailfish
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", type=str, nargs="+")
parser.add_argument(
        "--Momentum_Change",
        "-M",
        default=False,
        help="whether to plot the total change in momentum timeseries",
    )
parser.add_argument(
        "--Torque_Components",
        "-t",
        default=False,
        help="whether to plot the torque components from the binary and buffer",
    )
parser.add_argument(
        "--Accretion",
        "-a",
        default=False,
        help="whether to plot the binary's accretion timeseries",
    )
parser.add_argument(
        "--Orbital_Elements",
        "-o",
        default=False,
        help="whether to plot the binary's changing orbital elements",
    )
args = parser.parse_args()


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as file:
        chkpt = pk.load(file)
        return chkpt

def E_from_M(M, e=1.0):
    f = lambda E: E - e * np.sin(E) - M
    E = scipy.optimize.root_scalar(f, x0=M, x1=M + 0.1, method='secant').root
    return E

class DavidTimeseries:

    def __init__(self, chkpt):
        ts = load_checkpoint(chkpt)['timeseries']
        self.time           = np.array([s[ 0] for s in ts])
        self.semimajor_axis = np.array([s[ 1] for s in ts])
        self.eccentricity   = np.array([s[ 2] for s in ts])
        self.mdot1          = np.array([s[ 3] for s in ts])
        self.mdot2          = np.array([s[ 4] for s in ts])
        self.torque_g       = np.array([s[ 5] for s in ts])
        self.torque_a       = np.array([s[ 6] for s in ts])
        self.jdisk          = np.array([s[ 7] for s in ts])
        self.torque_b       = np.array([s[ 8] for s in ts])
        self.mdot_b         = np.array([s[ 9] for s in ts])
        self.disk_ecc       = np.array([s[10] for s in ts])

    @property
    def dt(self):
        return np.r_[0.0, np.diff(self.time * 2 * np.pi)]
    
    @property
    def mean_anomaly(self):
        return self.time * 2 * np.pi

    @property
    def eccentric_anomaly(self):
        return np.array([E_from_M(x, e=e) for x, e in zip(self.mean_anomaly, self.eccentricity)])

    @property
    def binary_torque(self):
        return self.torque_g + self.torque_a

    @property
    def binary_delta_j(self):
    	return (self.torque_g + self.torque_a) * self.dt

    @property
    def buffer_delta_j(self):
    	return self.torque_b * self.dt

    @property
    def total_angular_momentum(self):
    	return self.jdisk + self.binary_delta_j + self.buffer_delta_j # self.gw_delta_j
      

if __name__ == '__main__':
    filename         = args.checkpoints[0]
    CurrentTime      = load_checkpoint(filename)["time"]/ 2 / np.pi     ########FIX
    ts               = DavidTimeseries(filename)
    Model_Parameters = load_checkpoint(filename)['model_parameters']
    Number_of_Orbits = 10.
    Final_Orbits     = ts.time[ts.time>CurrentTime-Number_of_Orbits*2*np.pi]

    if args.Momentum_Change:
        plt.figure()
        plt.plot(Final_Orbits, ts.total_angular_momentum[-len(Final_Orbits):], c = 'black')
        plt.xlabel('time')
        plt.title('Total Angular Momentum')
        savename = os.getcwd() + "/Outputs/TotalAngularMomentum.%04d.png"%(CurrentTime)
        plt.savefig(savename, dpi=400)

    if args.Torque_Components:
        plt.figure()
        plt.xlabel('time')
        plt.title('Torque Components')
        plt.plot(Final_Orbits,ts.torque_b[-len(Final_Orbits):],c='green',label = 'Buffer Torque')
        plt.plot(Final_Orbits,ts.binary_torque[-len(Final_Orbits):],c = 'black',label = 'Binary Torque')

        plt.legend()
        savename = os.getcwd() + "/Outputs/TorqueComponents.%04d.png"%(CurrentTime)
        plt.savefig(savename, dpi=400)

    if args.Accretion:
        plt.figure()
        plt.plot(Final_Orbits,ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):],label='mdot',linewidth = 1, c = 'red')
        plt.xlabel('time')
        plt.title('Accretion Rate')
        savename = os.getcwd() +  "/Outputs/AccretionRate.%04d.png"%(CurrentTime)
        plt.savefig(savename, dpi=400)

    '''
    if Steady_State:
        viscosity              = Model_Parameters['nu']
        Initial_Density        = Model_Parameters['initial_sigma']
        CheckpointCadence      = load_checkpoint(filename)['driver'].events['checkpoint'].interval
        Steady_State_Accretion = 3 * np.pi * viscosity * Initial_Density
        mdot                   = ts.mdot1+ts.mdot2
        Number_of_Orbits       = 10

        plt.figure()
        Later_Times   = ts.time[len(ts.time)-Number_of_Orbits*10:len(ts.time)-1]
        Later_Torques = ts.torque_g[len(ts.time)-Number_of_Orbits*10:len(ts.time)-1]
        Later_mdot    = mdot[len(ts.time)-Number_of_Orbits*10:len(ts.time)-1]
        plt.plot(Later_Times,Later_Torques/Later_mdot, c= 'black', label = 'Unitless Torque')
        plt.plot(Later_Times,Later_mdot/Steady_State_Accretion,c = 'red',label = r'$\dot{M}/M_0$ ')
        plt.xlabel('time')
        plt.legend()
        savename = os.getcwd() +  "/Outputs/SteadyState_TorqueAccretion.%04d.png"%(CurrentTime)
        plt.savefig(savename, dpi=400)

        plt.figure()
        Split_Array  = np.array_split([Later_Torques[i]/Later_mdot[i] for i in range(0,Number_of_Orbits*10 -1)],Number_of_Orbits)
        Mean_torques = [np.mean(Split_Array[i]) for i in range(0,Number_of_Orbits)]
        Final_Orbits = np.arange(np.round(CurrentTime) - Number_of_Orbits,np.round(CurrentTime),1)
        plt.plot(Final_Orbits,Mean_torques,c = 'black',label = 'Orbit Averaged Torque')
        plt.axhline(y = np.mean(Mean_torques), c = 'red',label = 'Global Mean Torque')
        plt.xlabel('Time [t/P]')
        plt.legend()
        savename = os.getcwd() +  "/Outputs/MeanTorque_FinalOrbits.%04d.png"%(CurrentTime)
        plt.savefig(savename, dpi=400)
    '''
    if args.Orbital_Elements:
        plt.figure()
        plt.plot(Final_Orbits,ts.semimajor_axis[-len(Final_Orbits):], label = 'SemiMajor Axis')
        plt.plot(Final_Orbits,ts.eccentricity[-len(Final_Orbits):], label = 'Eccentricity')
        plt.xlabel('Time')
        plt.ylabel('Orbital Elements')
        plt.legend()
        savename = os.getcwd() +  "/Outputs/OrbitalElements.%04d.png"%(CurrentTime)
        plt.savefig(savename, dpi=400)






























