import sys 
import numpy as np 
import pickle as pk
import matplotlib.pyplot as plt 
import sailfish
import os
import argparse
from sailfish.setup_base import SetupBase
from sailfish.physics.kepler import OrbitalState, PointMass



class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt
    #with open(filename, "rb") as file:
    #    chkpt = pk.load(file)
    #    return chkpt




class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt

#def E_from_M(M, e=1.0):
#    f = lambda E: E - e * np.sin(E) - M
#    E = scipy.optimize.root_scalar(f, x0=M, x1=M + 0.1, method='secant').root
#    return E

class DavidTimeseries:
    def __init__(self, Checkpoint):
        #Checkpoint = load_checkpoint(chkpt)

        timeseries_data = Checkpoint['timeseries']
        max_len         = max(len(arr) for arr in timeseries_data)
        ts              = np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in timeseries_data])

        ts = Checkpoint['timeseries']

        self.pointmasses     = Checkpoint["point_masses"]
        self.currenttime     = Checkpoint["time"] / 2 / np.pi 
        self.modelparams     = Checkpoint['model_parameters'] 
        self.time            = np.array([s[ 0] for s in ts])
        self.semimajor_axis  = np.array([s[ 1] for s in ts])
        self.eccentricity    = np.array([s[ 2] for s in ts])
        self.density_floor   = np.array([s[ 3] for s in ts])
        self.pressure_floor  = np.array([s[ 4] for s in ts])
        self.Accreted_energy = np.array([s[ 5] for s in ts])

        self.optical         = np.array([s[ 6] for s in ts])
        self.infared         = np.array([s[ 7] for s in ts])
        self.bolometric      = np.array([s[ 8] for s in ts])
        self.uncounted_cells = np.array([s[ 9] for s in ts])

        self.mdot1           = np.array([s[10] for s in ts])
        self.mdot2           = np.array([s[11] for s in ts])
        self.torque_g        = np.array([s[12] for s in ts])
        self.torque_a        = np.array([s[13] for s in ts])
        self.power_g1        = np.array([s[14] for s in ts])
        self.power_g2        = np.array([s[15] for s in ts])
        self.power_a1        = np.array([s[16] for s in ts])
        self.power_a2        = np.array([s[17] for s in ts])
        self.jdisk           = np.array([s[18] for s in ts])
        self.uv              = np.array([s[19] for s in ts])
        self.xray            = np.array([s[20] for s in ts])
        #self.Max_temp        = np.array([s[21] for s in ts])


    @property
    def dt(self):
        return np.r_[0.0, np.diff(self.time * 2 * np.pi)]
    
    @property
    def mean_anomaly(self):
        return self.time * 2 * np.pi

    #@property
    #def eccentric_anomaly(self):
    #    return np.array([E_from_M(x, e=e) for x, e in zip(self.mean_anomaly, self.eccentricity)])

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
    parser.add_argument(
        "--Accreted_Energy",
        "-ae",
        action='store_true',
        help="whether to plot the energy of the gas accreted by the binary",
    )
    parser.add_argument(
        "--Energy",
        "-e",
        action='store_true',
        help="whether to plot the energy emmitted by the disk",
    )
    parser.add_argument(
        "--Lightcurves",
        "-lc",
        action='store_true',
        help="whether to plot the optical and infared luminosities of the disk",
    )
    parser.add_argument(
        "--FloorCount",
        "-fc",
        action='store_true',
        help="whether to plot the number of cells that have reached the floor values",
    )
    parser.add_argument(
        "--Max_temperature",
        "-MT",
        action='store_true',
        help="whether to plot the maximum temperature of the disk",
    )
    args = parser.parse_args()
    

    filename            = args.checkpoints[0]
    chkpt               = load_checkpoint(filename)
    ts                  = DavidTimeseries(chkpt)

    try:
        Primary,Secondary   = ts.pointmasses
        Point_MassPrimary   = PointMass(Primary.mass, Primary.position_x, Primary.position_y, Primary.velocity_x, Primary.velocity_y)
        Point_MassSecondary = PointMass(Secondary.mass,Secondary.position_x,Secondary.position_y,Secondary.velocity_x,Secondary.velocity_y)
        OrbitalEccentricity = OrbitalState(Point_MassPrimary,Point_MassSecondary).eccentricity
    except:
        Primary             = ts.pointmasses
        OrbitalEccentricity = 0.


    Primary,Secondary   = ts.pointmasses
    Point_MassPrimary   = PointMass(Primary.mass, Primary.position_x, Primary.position_y, Primary.velocity_x, Primary.velocity_y)
    Point_MassSecondary = PointMass(Secondary.mass,Secondary.position_x,Secondary.position_y,Secondary.velocity_x,Secondary.velocity_y)
    OrbitalEccentricity = OrbitalState(Point_MassPrimary,Point_MassSecondary).eccentricity


    CurrentTime         = ts.currenttime
    Model_Parameters    = ts.modelparams

    Number_of_Orbits    = 10.
    Final_Orbits        = ts.time[ts.time>CurrentTime-Number_of_Orbits]
    TimeBins            = np.arange(Final_Orbits[0],Final_Orbits[-1],1)

    hist, edges         = np.histogram(Final_Orbits, bins=int(Number_of_Orbits))
    CumulativeTimeBin   = np.cumsum(hist)
    alpha               = Model_Parameters["alpha"]

    if args.Max_temperature:
        plt.figure()
        plt.plot(ts.time[-len(Final_Orbits):], ts.Max_temp[-len(Final_Orbits):], c = 'black')
        plt.xlabel('time')
        plt.title('Maximum Temperature e = %g'%(np.round(OrbitalEccentricity,3)))
        plt.yscale('log')
        try:
            savename = os.getcwd() + "/MaxTemperature.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()


    import cooling
    from cooling import gamma_law_index, EffectiveTemperature, cgs
    print(chkpt['model_parameters'])
    gamma = gamma_law_index(chkpt['model_parameters']['beta'], chkpt['model_parameters']['gamma_law_index_gas'])

    try:
        length_scale_pc = chkpt['model_parameters']['length_scale_pc']
    except KeyError as e:
        r_g             = cgs['G'] * chkpt['model_parameters']['central_mass_msun'] * cgs['msun'] / cgs['c'] / cgs['c']
        length_scale_pc = r_g * chkpt['model_parameters']['init_separation_rg'] / cgs['pc']


    SS73 = cooling.ShakuraSunyaevDisk(
        central_mass_msun = chkpt['model_parameters']['central_mass_msun'], 
        length_scale_pc   = length_scale_pc,
        mach_number_a     = chkpt['model_parameters']['mach_number_a'],
        alpha             = chkpt['model_parameters']['alpha'],
        gamma             = gamma
        )

    if args.Max_temperature:
        plt.figure()
        plt.plot(ts.time[-len(Final_Orbits):], ts.Max_temp[-len(Final_Orbits):], c = 'black')
        plt.xlabel('time')
        plt.title('Maximum Temperature e = %g'%(np.round(OrbitalEccentricity,3)))
        plt.yscale('log')
        try:
            savename = os.getcwd() + "/MaxTemperature.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()

    if args.FloorCount:
        plt.figure()
        plt.plot(ts.time[-len(Final_Orbits):], ts.density_floor[-len(Final_Orbits):], c = 'black', label = r'$N_\mathrm{cells}$ at density floor')     
        plt.plot(ts.time[-len(Final_Orbits):], ts.pressure_floor[-len(Final_Orbits):], c = 'black', linestyle ='dashed', label = r'$N_\mathrm{cells}$ at pressure floor')     
        plt.plot(ts.time[-len(Final_Orbits):], ts.uncounted_cells[-len(Final_Orbits):], c = 'red', label = r'$N_\mathrm{cells}$ ignored by lightcurves') 
        plt.axhline(y = 4000000)
        #plt.ylim([0,10])
        

    if args.FloorCount:
        plt.figure()
        plt.plot(ts.time, ts.floor, c = 'black', label = r'$N_\mathrm{cells}$ at floor')     

        plt.xlabel('time')
        plt.legend()
        try:
            savename = os.getcwd() + "/FloorCount.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()
        #print("Nu",ts.floor)
        print("Nu",ts.floor)

        #print("Nu",ts.floor)
    
    if args.Lightcurves:
        plt.figure(figsize = (10,3))
        plt.plot(Final_Orbits, ts.infared[-len(Final_Orbits):], c = 'red', label = 'infared luminosity')
        plt.plot(Final_Orbits, ts.optical[-len(Final_Orbits):], c = 'blue', label = 'optical luminosity')
        plt.plot(Final_Orbits, ts.uv[-len(Final_Orbits):],   c = 'purple', label = 'uv')
        plt.plot(Final_Orbits, ts.xray[-len(Final_Orbits):], c = 'green', label = 'xray', linewidth = 0.6) 
        plt.plot(Final_Orbits, ts.bolometric[-len(Final_Orbits):], c = 'black', label = 'bolometric luminosity', linewidth = 0.6)
        plt.xlabel('time')
        plt.title('Multiband Lightcurves')
        plt.yscale('log')

        plt.title('Multiband Lightcurves e = %g'%(np.round(OrbitalEccentricity,3)))
        plt.ylim([0,1e40])

        plt.ylim([1e30, 1e46])
        plt.ylim([1e40, 6e48])
        plt.legend()
        try:
            savename = os.getcwd() + "/Lightcurves.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()

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


    if args.Energy:
        import cooling

        Press = chkpt['solution'][...,3]
        sigma = chkpt['solution'][...,0]

        SS73 = cooling.ShakuraSunyaevDisk(
            central_mass_msun = chkpt['model_parameters']['central_mass_msun'], 
            length_scale_pc   = chkpt['model_parameters']['length_scale_pc'],
            mach_number_3a    = chkpt['model_parameters']['mach_number_3a'],
            alpha             = chkpt['model_parameters']['alpha']
        )
        
        kb_code     = cooling.cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
        mp_code     = cooling.cgs['mp'] / (SS73._mass)
        kappa_code  = cooling.cgs['kappa'] / (SS73._length**2 / SS73._mass)
        sigmab_code = cooling.cgs['sigmab'] / (SS73._mass / SS73._time**3)	
        mid_T = (mp_code/kb_code) * (Press/sigma)

        eff_T = ((4/3) * mid_T**4 / (sigma * kappa_code))**0.25

        Q_dot = 2*sigmab_code * eff_T ** 4
        # Keep in code units so multiply this by the area of each cell in code units
        dx      = chkpt['mesh'].dx
        Total_E = np.sum(Q_dot) * dx**2

        plt.figure()
        plt.plot(Final_Orbits, ts.energy[-len(Final_Orbits):], c = 'black', label = 'Total Energy')
        plt.plot(Final_Orbits, ts.Accreted_energy[-len(Final_Orbits):], c = 'black', label = 'Accreted Energy')
        #plt.scatter(Final_Orbits[-1], Total_E, marker = "*")
        plt.xlabel('time')
        plt.title('Total Energy emitted by disk')

        try:
            savename = os.getcwd() + "/TotalEnergyEmitted.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()
        #plt.show()

    if args.Torque_Components:
        #InnerClipped_Torque = ts.innertorque[-len(Final_Orbits):] / M_dot_0
        #OuterClipped_Torque = ts.outertorque[-len(Final_Orbits):] / M_dot_0
        Normalised_Torque_g = ts.torque_g[-len(Final_Orbits):] #/ M_dot_0
        Normalised_Torque_a = ts.torque_a[-len(Final_Orbits):] #/ M_dot_0


        plt.figure()
        plt.xlabel('time')
        if Model_Parameters['retrograde']:
            plt.title(r'Torque Retrograde $\alpha = %g$'%(chkpt['model_parameters']['alpha']))
        else:
            plt.title(r'Torque Prograde $\alpha = %g$'%(chkpt['model_parameters']['alpha']))
        
        MeanTorque_g = [np.mean(Normalised_Torque_g[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        MeanTorque_a = [np.mean(Normalised_Torque_a[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]

        
        #plt.plot(Final_Orbits,Normalised_Torque_g, c = 'blue', linewidth = 0.1)
        plt.plot(TimeBins[1:],MeanTorque_g,linewidth = 0.5, label = 'Binned Torque Mean Gravitational', c = 'black')
        plt.plot(TimeBins[1:],MeanTorque_a,linewidth = 0.5, label = 'Binned Torque Mean Accretion',linestyle = 'dashed', c = 'black')
        
        #plt.plot(ts.time, ts.jdisk)
        #plt.xlim([CurrentTime-Number_of_Orbits,CurrentTime])


        plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')
        plt.legend(loc = 'upper right')
        #plt.ylim([-2.5,5])
        plt.ylabel(r'$\tau/\dot{M}_0$')
        try:
            savename = args.Output +  "/MeanTorque.%04d_alpha%g.png"%(CurrentTime,chkpt['model_parameters']['alpha'])
            plt.savefig(savename, dpi=400)
        except:
            plt.show()


        #print('Torque Mean at t=1000 is',MeanTorque_g[0]+MeanTorque_a[0])




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
        
        plt.xlim([CurrentTime-Number_of_Orbits,CurrentTime])
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

        M_dot_0 = - SS73._eddington_rate /( SS73.central_mass_msun * cgs['msun'] / cgs['year'])
    
        #Mean_Norm_Factor = np.array(max(ts.mdot1[-len(Final_Orbits)-100:-len(Final_Orbits)]+ts.mdot2[-len(Final_Orbits)-100:-len(Final_Orbits)]))
        #print('Mean_Norm_Factor', Mean_Norm_Factor)
        plt.figure()
        plt.plot(Final_Orbits,(ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):]),label='mdot',linewidth = 0.1, c = 'red')

        plt.xlabel('Time [P]')
        plt.ylabel(r'$\dot{M}/\langle\dot{M}_0\rangle$')
        plt.title(r'Accretion Rate e = %g, $\alpha=%g$'%(np.round(OrbitalEccentricity,3),alpha))
        plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')

        plt.ylim([0,300])
        plt.xlim([CurrentTime-Number_of_Orbits,CurrentTime])
        AccretionRate = (ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):])#/M_dot_0
        MeanAccretion = np.array([np.mean(AccretionRate[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))])/M_dot_0

        plt.plot(TimeBins[1:],MeanAccretion,linewidth = 0.5, label = 'Binned Means', c = 'black')
        plt.legend(loc = 'upper right')
        try:
            savename = args.Output +  "/AccretionRate.%04d_nu%g.png"%(CurrentTime,alpha)
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






