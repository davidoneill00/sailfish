import sys 
import numpy as np 
import pickle as pk
import matplotlib.pyplot as plt 
import sailfish
import os
import argparse
from sailfish.setup_base import SetupBase
from sailfish.physics.kepler import OrbitalState, PointMass

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
        Checkpoint = load_checkpoint(chkpt)
        ts = Checkpoint['timeseries']

        self.pointmasses    = Checkpoint["point_masses"]
        self.currenttime    = Checkpoint["time"] / 2 / np.pi 
        self.modelparams    = Checkpoint['model_parameters'] 
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
        #self.Inspiral_Times = Checkpoint["model_parameters"]["inspiral_time_list"]
        #self.Orbital_Phase  = Checkpoint["model_parameters"]["Fixed_Phases"]


        if Checkpoint["model_parameters"]["which_diagnostics"] == "david_new":
            self.innertorque    = np.array([s[11] for s in ts])
            self.outertorque    = np.array([s[12] for s in ts])
            self.power_g1       = np.array([s[13] for s in ts])
            self.power_a1       = np.array([s[14] for s in ts])
            self.power_g2       = np.array([s[15] for s in ts])
            self.power_a2       = np.array([s[16] for s in ts])
            self.innerpower_1   = np.array([s[17] for s in ts])
            self.outerpower_1   = np.array([s[18] for s in ts])
            self.innerpower_2   = np.array([s[19] for s in ts])
            self.outerpower_2   = np.array([s[20] for s in ts])

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

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--Output",
        "-o",
        default=None,
        type=str,
        help="Where to save the output png files",
    )
    parser.add_argument(
        "--Disk_Momentum",
        "-jd",
        action='store_true',
        help="whether to plot the total change in momentum timeseries",
    )

    parser.add_argument(
        "--Torque_Components",
        "-t",
        action='store_true',
        help="whether to plot the torque components from the binary",
    )
    parser.add_argument(
        "--Accretion",
        "-a",
        action='store_true',
        help="whether to plot the binary's accretion timeseries",
    )
    parser.add_argument(
        "--Orbital_Elements",
        "-OE",
        action='store_true',
        help="whether to plot the binary's changing orbital elements",
    )
    parser.add_argument(
        "--Power_Components",
        "-p",
        action='store_true',
        help="whether to plot the power exerted on the binary",
    )
    args = parser.parse_args()
    

    filename            = args.checkpoints[0]
    ts                  = DavidTimeseries(filename)
    Primary,Secondary   = ts.pointmasses
    Point_MassPrimary   = PointMass(Primary.mass, Primary.position_x, Primary.position_y, Primary.velocity_x, Primary.velocity_y)
    Point_MassSecondary = PointMass(Secondary.mass,Secondary.position_x,Secondary.position_y,Secondary.velocity_x,Secondary.velocity_y)
    OrbitalEccentricity = OrbitalState(Point_MassPrimary,Point_MassSecondary).eccentricity

    CurrentTime         = ts.currenttime
    Model_Parameters    = ts.modelparams

    Number_of_Orbits    = 1500.
    Final_Orbits        = ts.time[ts.time>CurrentTime-Number_of_Orbits]
    TimeBins            = np.arange(Final_Orbits[0],Final_Orbits[-1],1)

    hist, edges         = np.histogram(Final_Orbits, bins=int(Number_of_Orbits))
    CumulativeTimeBin   = np.cumsum(hist)
    viscosity           = Model_Parameters["nu"]
    Sigma_0             = Model_Parameters["initial_sigma"]
    M_dot_0             = 3 * np.pi * viscosity * Sigma_0
    


    if args.Disk_Momentum:
        plt.figure()
        plt.plot(Final_Orbits, ts.total_angular_momentum[-len(Final_Orbits):], c = 'black')
        plt.xlabel('time')
        plt.title('Total Angular Momentum e = %g Retrograde'%(np.round(OrbitalEccentricity,3)))
        try:
            savename = os.getcwd() + "/TotalAngularMomentum.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()

    if args.Torque_Components:
        InnerClipped_Torque = ts.innertorque[-len(Final_Orbits):] / M_dot_0
        OuterClipped_Torque = ts.outertorque[-len(Final_Orbits):] / M_dot_0
        Normalised_Torque   = ts.torque_g[-len(Final_Orbits):] / M_dot_0

        plt.figure()
        plt.xlabel('time')
        if Model_Parameters['retrograde']:
            plt.title(r'Torque Retrograde $\nu = %g$'%(viscosity))
        else:
            plt.title(r'Torque Prograde $\nu = %g$'%(viscosity))
        
        MeanTorque = [np.mean(Normalised_Torque[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]

        plt.plot(Final_Orbits,Normalised_Torque, c = 'blue', linewidth = 0.1)
        plt.plot(TimeBins[1:],MeanTorque,linewidth = 0.5, label = 'Binned Torque Mean', c = 'black')
        plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')
        plt.legend(loc = 'upper right')
        #plt.ylim([-2.5,5])
        plt.ylabel(r'$\tau/\dot{M}_0$')
        try:
            savename = args.Output +  "/MeanTorque.%04d_nu%g.png"%(CurrentTime,viscosity)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()




    if args.Power_Components:
        Normalised_Power    = (ts.power_g1[-len(Final_Orbits):]+ts.power_g2[-len(Final_Orbits):]) / M_dot_0
        InnerClipped_Power  = (ts.innerpower_1[-len(Final_Orbits):]+ts.innerpower_2[-len(Final_Orbits):]) / M_dot_0
        OuterClipped_Power  = (ts.outerpower_1[-len(Final_Orbits):]+ts.outerpower_2[-len(Final_Orbits):]) / M_dot_0
        
        plt.figure()
        plt.xlabel('time')
        if Model_Parameters['retrograde']:
            plt.title(r'Power Retrograde $\nu = %g$'%(viscosity))
        else:
            plt.title(r'Power Prograde $\nu = %g$'%(viscosity))

        MeanPower = [np.mean(Normalised_Power[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        
        plt.plot(Final_Orbits,Normalised_Power, c = 'Purple', label = 'Power', linewidth = 0.1,)
        plt.plot(TimeBins[1:],MeanPower,linewidth = 0.5, label = 'Binned Means', c = 'black')
        plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')
        plt.legend(loc = 'upper right')
        #plt.ylim([-10,10])
        plt.ylabel(r'$\mathcal{P}/\dot{M}_0$')
        try:
            savename = args.Output +  "/MeanPower.%04d_nu%g.png"%(CurrentTime,viscosity)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()




    if args.Accretion:

        plt.figure()
        plt.plot(Final_Orbits,(ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):])/np.mean(ts.mdot1[-len(Final_Orbits)-100:-len(Final_Orbits)]+ts.mdot2[-len(Final_Orbits)-100:-len(Final_Orbits)]),label='mdot',linewidth = 0.1, c = 'red')
        plt.xlabel('Time [P]')
        plt.ylabel(r'$\dot{M}/\langle\dot{M}_0\rangle$')
        plt.title(r'Accretion Rate e = %g, $\nu=%g$'%(np.round(OrbitalEccentricity,3),viscosity))
        plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')

        plt.ylim([0,2])
        AccretionRate = (ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):])
        MeanAccretion = [np.mean(AccretionRate[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        plt.plot(TimeBins[1:],MeanAccretion/np.mean(ts.mdot1[-len(Final_Orbits)-100:-len(Final_Orbits)]+ts.mdot2[-len(Final_Orbits)-100:-len(Final_Orbits)]),linewidth = 0.5, label = 'Binned Means', c = 'black')
        plt.legend(loc = 'upper right')
        try:
            savename = args.Output +  "/AccretionRate.%04d_nu%g.png"%(CurrentTime,viscosity)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()



    if args.Orbital_Elements:
        plt.figure()
        plt.plot(ts.time,ts.semimajor_axis, label = 'SemiMajor Axis')
        plt.plot(ts.time,ts.eccentricity, label = 'Eccentricity')
        plt.title(r'Orbital Elements $e_0 =$%g Retrograde'%(np.round(OrbitalEccentricity,3)))
        plt.xlabel('Time')
        plt.ylabel('Orbital Elements')
        plt.legend()
        try:
            savename = args.Output +  "/OrbitalElements.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()






