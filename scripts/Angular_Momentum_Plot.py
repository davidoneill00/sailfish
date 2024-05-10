import sys 
import numpy as np 
import pickle as pk
import matplotlib.pyplot as plt 

sys.path.insert(1,"/groups/astro/davidon/sailfish/")
import sailfish

Momentum_Timeseries  = False
Accretion_Timeseries = False
Steady_State         = False

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
        #print(load_checkpoint(chkpt).keys())
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
    def binary_delta_j(self):
    	return (self.torque_g + self.torque_a) * self.dt

    @property
    def buffer_delta_j(self):
    	return self.torque_b * self.dt

    @property
    def total_angular_momentum(self):
    	return self.jdisk + self.binary_delta_j + self.buffer_delta_j # self.gw_delta_j
      

if __name__ == '__main__':
    filepath = '/lustre/astro/davidon/Storage/sfish-test/Retrograde_Circular_Iso_Test/e0/'
    filename = filepath + 'chkpt.%04d.pk'%(int(sys.argv[1]))
    
    Savepath = filepath.replace("/lustre/astro/davidon/Storage/sfish-test/","/groups/astro/davidon/sailfish/Outputs/") 

    print(load_checkpoint(filename)['model_parameters'])

    if Momentum_Timeseries:
        ts   = DavidTimeseries(filename)
        j0   = ts.jdisk
        plt.plot(ts.time, 1-ts.jdisk / j0, label=r'$\Delta j_\mathrm{disk}/j_0 $',c = 'blue')
        plt.plot(ts.time, ts.binary_delta_j / j0, label='binary torque',c = 'black')
        plt.plot(ts.time, ts.buffer_delta_j / j0, label='buffer torque',c = 'green')
        plt.plot(ts.time, 1-ts.total_angular_momentum / j0, label='1-total/j0',linestyle = 'dashed')
        plt.xlabel('time')
        #plt.ylabel()
        YBounds = max(np.max(np.abs(1-ts.jdisk / j0)), np.max(np.abs(ts.binary_delta_j / j0)), np.max(np.abs(ts.buffer_delta_j / j0)))
        plt.ylim([-YBounds/10,YBounds/10])
        plt.legend()
        savename = "AngularMomentum.%04d.png"%(int(sys.argv[1]))
        plt.savefig(Savepath + savename, dpi=400)

    if Accretion_Timeseries:
        Model_Parameters       = load_checkpoint(filename)['model_parameters']
        viscosity              = Model_Parameters['nu']
        Initial_Density        = Model_Parameters['initial_sigma']
        Steady_State_Accretion = 3 * np.pi * viscosity * Initial_Density

        ts   = DavidTimeseries(filename)
        mdot = ts.mdot1+ts.mdot2

        plt.figure()
        plt.plot(ts.time,mdot,label='mdot',linewidth = 1)
        YBounds = np.max(np.abs(mdot))
        plt.ylim([-YBounds/5,0])
        plt.xlabel('time')
        plt.legend()
        savename = "AccretionRate.%04d.png"%(int(sys.argv[1]))
        plt.savefig(Savepath + savename, dpi=400)

    if Steady_State:
        Model_Parameters  = load_checkpoint(filename)['model_parameters']
        viscosity         = Model_Parameters['nu']
        Initial_Density   = Model_Parameters['initial_sigma']
        CheckpointCadence = load_checkpoint(filename)['driver'].events['checkpoint'].interval

        ts                     = DavidTimeseries(filename)
        Steady_State_Accretion = 3 * np.pi * viscosity * Initial_Density
        mdot                   = ts.mdot1+ts.mdot2
        Number_of_Orbits       = 100

        
        plt.figure()
        Later_Times   = ts.time[len(ts.time)-Number_of_Orbits*10:len(ts.time)-1]
        Later_Torques = ts.torque_g[len(ts.time)-Number_of_Orbits*10:len(ts.time)-1]
        Later_mdot    = mdot[len(ts.time)-Number_of_Orbits*10:len(ts.time)-1]
        plt.plot(Later_Times,Later_Torques/Later_mdot, c= 'black', label = 'Unitless Torque')
        plt.plot(Later_Times,Later_mdot/Steady_State_Accretion,c = 'red',label = r'$\dot{M}/M_0$ ')
        #plt.plot(ts.time,ts.torque_g/mdot,label='Unitless gravitational torque')
        plt.xlabel('time')
        plt.legend()
        savename = "SteadyState_TorqueAccretion.%04d.png"%(int(sys.argv[1]))
        plt.savefig(Savepath + savename, dpi=400)

    
        plt.figure()
        Split_Array  = np.array_split([Later_Torques[i]/Later_mdot[i] for i in range(0,Number_of_Orbits*10 -1)],Number_of_Orbits)
        Mean_torques = [np.mean(Split_Array[i]) for i in range(0,Number_of_Orbits)]
        #print(Mean_torques)
        Final_Orbits = np.arange(int(sys.argv[1]) * CheckpointCadence - Number_of_Orbits,int(sys.argv[1])* CheckpointCadence,1)
        plt.plot(Final_Orbits,Mean_torques,c = 'black',label = 'Orbit Averaged Torque')
        plt.axhline(y = np.mean(Mean_torques), c = 'red',label = 'Global Mean Torque')
        plt.xlabel('Time [t/P]')
        plt.legend()
        savename = "MeanTorque_FinalOrbits.%04d.png"%(int(sys.argv[1]))
        plt.savefig(Savepath + savename, dpi=400)
