import numpy as np 
import pickle as pk
import matplotlib.pyplot as plt 
import sailfish
import argparse
from sailfish.physics.kepler import OrbitalState, PointMass
from sailfish.physics.cooling import cgs
import os
from plot import configure_matplotlib
from scipy.signal import lombscargle


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = pk.load(f)
    return chkpt

text_width   = 3.55
column_width = text_width / 2.
configure_matplotlib()

class DavidTimeseries:
    def __init__(self, Checkpoint):
        timeseries_data  = Checkpoint['timeseries']
        self.pointmasses = Checkpoint["point_masses"]
        self.currenttime = Checkpoint["time"] / 2 / np.pi 
        self.modelparams = Checkpoint['model_parameters'] 
        self.SS73        = Checkpoint['SS73']
        max_length       = max(len(arr) for arr in timeseries_data)
        ts               = np.array([np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in timeseries_data])

        if Checkpoint['driver'].model_parameters['which_diagnostics'] == 'david':
            self.time            = np.array([s[ 0] for s in ts])
            self.semimajor_axis  = np.array([s[ 1] for s in ts])
            self.eccentricity    = np.array([s[ 2] for s in ts])
            self.density_floor   = np.array([s[ 3] for s in ts])
            self.pressure_floor  = np.array([s[ 4] for s in ts])
            self.Accreted_energy = np.array([s[ 5] for s in ts])

            self.infared         = np.array([s[ 6] for s in ts])
            self.optical         = np.array([s[ 7] for s in ts])
            self.bolometric      = np.array([s[ 8] for s in ts])
            self.uv              = np.array([s[ 9] for s in ts])
            self.xray            = np.array([s[10] for s in ts])

            self.uncounted_cells = np.array([s[11] for s in ts])
            self.mdot1           = np.array([s[12] for s in ts])
            self.mdot2           = np.array([s[13] for s in ts])
            self.torque_g        = np.array([s[14] for s in ts])
            self.torque_a        = np.array([s[15] for s in ts])
            self.power_g1        = np.array([s[16] for s in ts])
            self.power_g2        = np.array([s[17] for s in ts])
            self.power_a1        = np.array([s[18] for s in ts])
            self.power_a2        = np.array([s[19] for s in ts])
            self.jdisk           = np.array([s[20] for s in ts])
            self.Max_temp        = np.array([s[21] for s in ts])

    @property
    def power_g(self):
        return self.power_g1 + self.power_g2
    
    @property
    def power_a(self):
        return self.power_a1 + self.power_a2

    @property
    def dt(self):
        return np.r_[0.0, np.diff(self.time * 2 * np.pi)]

    @property
    def mean_anomaly(self):
        return self.time * 2 * np.pi
    
    @property
    def binary_torque(self):
        return self.torque_g + self.torque_a

    @property
    def binary_delta_j(self):
        return self.binary_torque * self.dt

    #@property
    #def buffer_delta_j(self):
    #    return self.torque_b * self.dt

    #@property
    #def total_angular_momentum(self):
    #    return self.jdisk + self.binary_delta_j + self.buffer_delta_j # self.gw_delta_j


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument("--Number_of_Orbits", "-N", default=20., type=float, help="Number of orbits to plot")
    parser.add_argument(
        "--Outputs",
        "-o",
        default=None,
        type=str,
        help="Where to save the output png files",
    )
    parser.add_argument(
        "--Accretion",
        "-a",
        action='store_true',
        help="whether to plot the binary's accretion timeseries",
    )
    parser.add_argument(
        "--Torque_Components",
        "-t",
        action='store_true',
        help="whether to plot the torque components from the binary",
    )
    parser.add_argument(
        "--Power_Components",
        "-p",
        action='store_true',
        help="whether to plot the power exerted on the binary",
    )
    parser.add_argument(
        "--Lightcurves",
        "-lc",
        action='store_true',
        help="whether to plot the optical and infared luminosities of the disk",
    )
    parser.add_argument(
        "--Orbital_Elements",
        "-oe",
        action='store_true',
        help="whether to plot the binary's changing orbital elements",
    )
    parser.add_argument(
        "--Accreted_Energy",
        "-ae",
        action='store_true',
        help="whether to plot the energy of the gas accreted by the binary",
    )
    parser.add_argument(
        "--Disk_Momentum",
        "-jd",
        action='store_true',
        help="whether to plot the total change in momentum timeseries",
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
    parser.add_argument(
        "--Fourier",
        "-f",
        action='store_true',
        help="whether to plot the Fourier transforms",
    )
    args     = parser.parse_args()
    filename = args.checkpoints[0]
    chkpt    = load_checkpoint(filename)
    ts       = DavidTimeseries(chkpt)

    # ======= Useful timeseries data ========
    SS73                = ts.SS73
    gamma               = ts.modelparams['gamma_law_index']
    alpha               = ts.modelparams["alpha"]
    primary, secondary  = ts.pointmasses
    OrbitalEccentricity = ts.eccentricity[-1]
    Final_Orbits        = ts.time[ts.time>(ts.currenttime-args.Number_of_Orbits)]
    TimeBins            = np.arange(Final_Orbits[0],Final_Orbits[-1],1)
    hist, edges         = np.histogram(Final_Orbits, bins=int(args.Number_of_Orbits))
    CumulativeTimeBin   = np.cumsum(hist)
    

    # ======= Plotting Blocks ========
    if args.Accretion:
        cs_a          = (SS73.gamma * (SS73.surface_pressure_profile(r=1) / SS73.surface_density_profile(r=1)))**0.5
        nu_a          = SS73.alpha * cs_a**2 # in code units as omega_bin = 1 at r = 1
        M_dot_0       = 3 * np.pi * (nu_a * SS73.surface_density_profile(r=1))
        Accretion_1   = ts.mdot1[-len(Final_Orbits):] / M_dot_0
        Accretion_2   = ts.mdot2[-len(Final_Orbits):] / M_dot_0
        AccretionRate = Accretion_1 + Accretion_2
        MeanAccretion = np.array([np.mean(AccretionRate[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))])

        if args.Fourier:
            signal = AccretionRate - np.mean(AccretionRate)
            freq   = np.logspace(-2, 2, 100)      # cycles / orbit
            omega  = 2 * np.pi * freq             # rad / orbit
            power  = lombscargle(Final_Orbits, signal, omega)
            power /= np.var(signal)


            fig, ax = plt.subplots(figsize=[text_width, text_width])
            ax.plot(freq, power, c='black')
            ax.set_xscale('log')
            ax.set_xlabel(r'$f\ \mathrm{[cycles/orbit]}$')   # or just f
            ax.set_ylabel('Power')
            ax.set_xlim([4e-2, 10])

            ax.set_ylabel('Power')
            savename      = "Accretion_Fourier"

        else:
            fig, ax = plt.subplots(figsize=[2*text_width, text_width])
            plt.plot(Final_Orbits, -AccretionRate, label='mdot',linewidth = 0.1, c = 'black')
            plt.plot(Final_Orbits, -Accretion_1  , label='mdot1',linewidth = 0.1, c = 'blue' )
            plt.plot(Final_Orbits, -Accretion_2  , label='mdot2',linewidth = 0.1, c = 'red'  )
            #plt.plot(TimeBins[1:], -MeanAccretion,linewidth = 1, label = 'Binned Means', c = 'black')
            plt.yscale('log')
            plt.xlabel('Time [P]')
            plt.ylabel(r'$\log_{10}\left(\dot{M}/\langle\dot{M}_0\rangle\right)$')
            plt.title(r'Accretion Rate')
            plt.legend(loc = 'upper right')
            savename       = "Accretion"

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.Torque_Components:
        Normalised_Torque_g = ts.torque_g[-len(Final_Orbits):] / M_dot_0
        Normalised_Torque_a = ts.torque_a[-len(Final_Orbits):] / M_dot_0
        Mean_Torque_g       = [np.mean(Normalised_Torque_g[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        Mean_Torque_a       = [np.mean(Normalised_Torque_a[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        Total_Torque        = Normalised_Torque_g + Normalised_Torque_a
        savename            = "Torque"

        fig, ax = plt.subplots(figsize=[2*text_width, text_width])
        plt.plot(Final_Orbits, Normalised_Torque_g, c = 'blue'    , label = 'Gravitational Torque', linewidth = 0.1)
        #plt.plot(Final_Orbits, Normalised_Torque_a, c = 'darkblue', label = 'Accretion Torque'    , linewidth = 0.1) 
        plt.plot(TimeBins[1:], Mean_Torque_g, linewidth = 0.5, label = 'Gravitational Mean ', c = 'black')
        #plt.plot(TimeBins[1:], Mean_Torque_a, linewidth = 0.5, label = 'Accretion Mean'     , c = 'black')
        #plt.plot(Final_Orbits, Total_Torque , linewidth = 0.5, label = 'Total Torque'       , c = 'black')
        plt.xlabel('Time [P]')
        plt.ylabel(r'$\tau/\dot{M}_0$')
        plt.legend(loc = 'upper right')

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.Power_Components:
        Normalised_Power_g = ts.power_g[-len(Final_Orbits):] / M_dot_0
        Normalised_Power_a = ts.power_a[-len(Final_Orbits):] / M_dot_0
        Mean_Power_g       = [np.mean(Normalised_Power_g[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        Mean_Power_a       = [np.mean(Normalised_Power_a[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        savename           = "Power"

        fig, ax = plt.subplots(figsize=[2*text_width, text_width])
        plt.plot(Final_Orbits, Normalised_Power_g, c = 'Purple'    , label = 'Gravitational Power', linewidth = 0.1)
        #plt.plot(Final_Orbits, Normalised_Power_a, c = 'darkblue', label = 'Accretion Power'    , linewidth = 0.1) 
        plt.plot(Final_Orbits, Mean_Power_g,linewidth = 0.5, label = 'Gravitational Mean ', c = 'black')
        #plt.plot(Final_Orbits, Mean_Power_a,linewidth = 0.5, label = 'Accretion Mean'     , c = 'black', linestyle = 'dashed')
        #plt.plot(Final_Orbits, Total_Power, linewidth = 0.5, label = 'Total Power'        , c = 'black')
        plt.xlabel('Time [P]')
        plt.ylabel(r'$\mathcal{P}/\dot{M}_0$')
        plt.legend(loc = 'upper right')

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.Lightcurves:
        fig, ax = plt.subplots(figsize=[2*text_width, text_width])
        plt.plot(Final_Orbits, ts.infared[-len(Final_Orbits):]   , c = 'red'   , label = 'infared luminosity')
        plt.plot(Final_Orbits, ts.optical[-len(Final_Orbits):]   , c = 'blue'  , label = 'optical luminosity')
        plt.plot(Final_Orbits, ts.uv[-len(Final_Orbits):]        , c = 'purple', label = 'uv')
        plt.plot(Final_Orbits, ts.xray[-len(Final_Orbits):]      , c = 'green' , label = 'xray') 
        plt.plot(Final_Orbits, ts.bolometric[-len(Final_Orbits):], c = 'black' , label = 'bolometric luminosity')
        
        plt.xlabel('Time [P]')
        plt.title('Electromagnetic Emission from Disk')
        plt.yscale('log')
        plt.ylim([1e40, 1e48])
        plt.legend(loc='lower left')
        savename = "Lightcurves"

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.Orbital_Elements:
        SemiMajorAxis = ts.semimajor_axis[-len(Final_Orbits):]
        Eccentricity  = ts.eccentricity[-len(Final_Orbits):]
        savename      = "OrbitalElements"

        fig, ax = plt.subplots(figsize=[2*text_width, text_width])
        plt.plot(Final_Orbits,SemiMajorAxis, label = 'Semi-major axis')
        plt.plot(Final_Orbits,Eccentricity , label = 'Eccentricity')
        plt.xlabel('Time [P]')
        plt.title('Orbital Elements')
        #plt.yscale('log')
        plt.legend(loc='lower left')
        
        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.Disk_Momentum:
        DiskMomentum = ts.jdisk[-len(Final_Orbits):]

        fig, ax = plt.subplots(figsize=[2*text_width, text_width])
        plt.plot(Final_Orbits, DiskMomentum, c = 'black')
        plt.xlabel('Time [P]')
        plt.title('Disk Angular Momentum')
        plt.legend(loc='lower left')
        savename = "JDisk"

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.FloorCount:
        DensityFloor   = ts.density_floor[-len(Final_Orbits):]
        PressureFloor  = ts.pressure_floor[-len(Final_Orbits):] 
        UncountedCells = ts.uncounted_cells[-len(Final_Orbits):]

        fig, ax = plt.subplots(figsize=[text_width, text_width])
        plt.plot(Final_Orbits, DensityFloor  , c = 'black', label = r'$N_\mathrm{cells}$ at density floor'      , linewidth = 0.2)     
        plt.plot(Final_Orbits, PressureFloor , c = 'black', label = r'$N_\mathrm{cells}$ at pressure floor'     , linewidth = 0.2, linestyle ='dashed')     
        plt.plot(Final_Orbits, UncountedCells, c = 'red'  , label = r'$N_\mathrm{cells}$ ignored by lightcurves', linewidth = 0.2) 
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Time')
        plt.title('Floor Count')
        savename = "FloorCount"

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.Max_temperature:
        MaxTemp = ts.Max_temp[-len(Final_Orbits):]

        fig, ax = plt.subplots(figsize=[text_width, text_width])
        plt.plot(Final_Orbits, MaxTemp, c = 'black')
        plt.xlabel('Time')
        plt.title('Maximum Temperature')
        plt.yscale('log')
        savename = "MaxTemperature"

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
