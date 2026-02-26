#from ideas import solver
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

text_width   = 6.8
column_width = 3.3
configure_matplotlib()

def ComputeBinnedMeans(times, field, Averaging_Window):
    
    # Create bins
    num_bins  = int(np.ceil((times[-1] - times[0]) / Averaging_Window))
    bin_edges = np.linspace(times[0], times[-1], num_bins + 1)
    
    # Compute statistics for each bin
    bin_means   = np.zeros(num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for i in range(num_bins):
        bin_mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
        bin_data = field[bin_mask]
        
        if len(bin_data) > 0:
            bin_means[i] = np.mean(bin_data)
        else:
            bin_means[i] = np.nan
    
    return bin_centers, bin_means

class DavidTimeseries:
    def __init__(self, Checkpoint):
        timeseries_data  = Checkpoint['timeseries']
        self.pointmasses = Checkpoint["point_masses"]
        self.currenttime = Checkpoint["time"] / 2 / np.pi 
        self.modelparams = Checkpoint['model_parameters'] 
        self.SS73        = Checkpoint['SS73']
        #self.dt_cadence  = Checkpoint['driver'].events['timeseries'].interval

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
            
            try:
                self.torque_b        = np.array([s[22] for s in ts])
                self.torque_b_dyn    = np.array([s[23] for s in ts])
                self.inflow_b        = np.array([s[24] for s in ts])
            except IndexError:
                #self.inflow_b        = np.array([s[23] for s in ts])
                pass
            
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

    @property
    def buffer_torque(self):
        return self.torque_b
    
    @property
    def buffer_inflow(self):
        return self.inflow_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument("--Number_of_Orbits", "-N", default=20., type=float, help="Number of orbits to plot")
    parser.add_argument("--Number_of_Averages", "-A", default=5., type=float, help="Number of orbits to average over")

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
        "--Lightcurves",
        "-lc",
        action='store_true',
        help="whether to plot the optical and infared luminosities of the disk",
    )
    parser.add_argument(
        "--Orbital_Elements",
        "-orb",
        action='store_true',
        help="whether to plot the binary's changing orbital elements",
    )
    parser.add_argument(
        "--OrbitalEvolution",
        "-oe",
        action='store_true',
        help="whether to plot the binary's changing orbital elements",
    )
    parser.add_argument(
        "--Disk_Momentum",
        "-jd",
        action='store_true',
        help="whether to plot the total change in momentum timeseries",
    )
    parser.add_argument(
        "--Buffer_Torque",
        "-b",
        action='store_true',
        help="whether to plot the buffer torque",
    )
    parser.add_argument(
        "--FloorCount",
        "-fc",
        action='store_true',
        help="whether to plot the number of cells that have reached the floor values",
    )
    parser.add_argument(
        "--Torque",
        "-t",
        action='store_true',
        help="whether to plot the binary's torque timeseries",
    )
    parser.add_argument(
        "--Max_temperature",
        "-MT",
        action='store_true',
        help="whether to plot the maximum temperature of the disk",
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
    cs_a                = (SS73.gamma * (SS73.surface_pressure_profile(r=1) / SS73.surface_density_profile(r=1)))**0.5
    nu_a                = SS73.alpha * cs_a**2
    M_dot_0             = SS73.Mdot_inf
    Final_Orbits        = ts.time[ts.time>(ts.currenttime-args.Number_of_Orbits)]

    # ======= Plotting Blocks ========
    if args.Accretion:
        Accretion_1   = ts.mdot1[-len(Final_Orbits):] / M_dot_0
        Accretion_2   = ts.mdot2[-len(Final_Orbits):] / M_dot_0
        #Inflow_buff   = ts.inflow_b[-len(Final_Orbits):] / M_dot_0
        AccretionRate = Accretion_1 + Accretion_2
        #Inflow        = ts.buffer_inflow[-len(Final_Orbits):] / M_dot_0
        cmap          = plt.cm.hot
        n_lines       = 3
        signal        = AccretionRate - np.mean(AccretionRate)
        freq          = np.logspace(-1, 1, 1000)      # cycles / orbit
        omega         = 2 * np.pi * freq              # rad / orbit
        power         = lombscargle(Final_Orbits, signal, omega)
        power        /= np.var(signal)
        savename      = "Accretion"
        
        fig = plt.figure(figsize=(1.2 * text_width, 1.2 * 0.25*text_width))
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.33], hspace=0., wspace=0.)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        ax0.plot(Final_Orbits, -AccretionRate, label='$\dot{M}_\mathrm{t}$'      , linewidth = 1.0, c = cmap(0/n_lines)  )
        ax0.plot(Final_Orbits, -Accretion_2  , label=r'$\dot{M}_2$'              , linewidth = 1.0, c = cmap(1.9/n_lines))
        ax0.plot(Final_Orbits, -Accretion_1  , label=r'$\dot{M}_1$'              , linewidth = 1.0, c = cmap(0.7/n_lines), alpha = 0.7)
        #ax0.plot(Final_Orbits,  Inflow_buff  , label=r'$\dot{M}_\mathrm{buffer}$', linewidth = 1.2, c = cmap(2.4/n_lines), linestyle='dashed')
        ax0.plot(*ComputeBinnedMeans(Final_Orbits, -AccretionRate, args.Number_of_Averages), label='Binned Means', linewidth = 1.2, c = cmap(0/n_lines), linestyle='dashed')
        ax0.set_xlabel('Time [P]')
        ax0.set_ylabel(r'$\dot{M}/\langle\dot{M}_0\rangle$')
        ax0.set_title(r'Accretion Timeseries')
        ax0.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        #ax0.set_ylim([0, 100])

        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.spines["right"].set_visible(True)
        ax1.spines["left"].set_visible(False)
        ax1.plot(freq, power/np.max(power), c='black', linewidth = 0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$f\ \mathrm{[\Omega]}$')
        ax1.set_ylabel('Power')
        ax1.set_title('Accretion Periodogram')
        ax1.set_ylabel('Power')
        
        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')



    if args.OrbitalEvolution:
        Total_Torque = -np.array(ts.torque_g[-len(Final_Orbits):]) / M_dot_0 + np.array(ts.torque_a[-len(Final_Orbits):]) / M_dot_0
        Total_Power  = -np.array(ts.power_g[-len(Final_Orbits):] )/ M_dot_0  + np.array(ts.power_a[-len(Final_Orbits):] )/ M_dot_0 
        Accretion_1  = -np.array(ts.mdot1[-len(Final_Orbits):] / M_dot_0)
        Accretion_2  = -np.array(ts.mdot2[-len(Final_Orbits):] / M_dot_0)
        a            = np.array(ts.semimajor_axis[-len(Final_Orbits):])
        e            = np.array(ts.eccentricity[-len(Final_Orbits):])
        Energydot    = Total_Power                    # total orbital energy change rate
        Ldot         = Total_Torque                   # total orbital angular momentum change rate
        Mdot         = Accretion_1 + Accretion_2      # total mass accretion rate
        m1, m2       = primary.mass, secondary.mass
        M, G         = m1 + m2 , 1.0

        Energy       = - G * M**2  / (8 * a)          # total orbital energy
        L            = (m1*m2 / M) * np.sqrt(G * M * a * (1 - e**2))
        adot         = - a * Energydot / Energy + 2 * a * Mdot / M  # Fix for q, eta =/= 1 !!!
        edot         = (1 - e**2) / (2*e) * (5 * Mdot / M - Energydot / Energy - 2 * Ldot / L)  # Fix for q, eta =/= 1 !!!
        savename     = 'OrbitalEvolution'


        fig, ax = plt.subplots(figsize=[text_width, column_width])
        plt.plot(Final_Orbits, adot, c = 'darkblue'    , label = r'$\dot{a}/a$', linewidth = 0.1)
        plt.plot(Final_Orbits, edot, c = 'darkred'     , label = r'$\dot{e}$', linewidth = 0.1)
        plt.plot(*ComputeBinnedMeans(Final_Orbits, adot, args.Number_of_Averages), linewidth = 1, label = 'Mean ', c = 'black')
        plt.plot(*ComputeBinnedMeans(Final_Orbits, edot, args.Number_of_Averages), linewidth = 1,                  c = 'black')
        plt.xlabel('Time [P]')
        plt.ylim([-100,100])
        plt.ylabel(r'$\dot{a}/a, \dot{e} \left[\Omega\right]$')
        plt.legend(loc = 'upper right')
        ax.text(0.5, 0.75, r'$~\frac{1}{a}\frac{da}{d(\Omega t)} = %g,~~\frac{de}{d(\Omega t)} = %g~$'% (np.round(np.mean(ComputeBinnedMeans(Final_Orbits, adot, args.Number_of_Averages)[1]), 2), np.round(np.mean(ComputeBinnedMeans(Final_Orbits, edot, args.Number_of_Averages)[1]), 2)), transform=ax.transAxes, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')



    if args.Buffer_Torque:
        Buffer_Torque        = ts.torque_b[-len(Final_Orbits):] / M_dot_0
        Binary_Torque        = ts.torque_a[-len(Final_Orbits):] / M_dot_0 + ts.torque_g[-len(Final_Orbits):] / M_dot_0
        savename             = "BufferTorque"
        DynamicalTorque      = ts.torque_b_dyn[-len(Final_Orbits):] / M_dot_0

        fig, ax = plt.subplots(figsize=[text_width, 0.5*text_width])
        plt.plot(Final_Orbits, Buffer_Torque  , c = 'blue', label = 'Buffer Torque'     , linewidth = 0.8)
        plt.plot(Final_Orbits, DynamicalTorque, c = 'red' , label = 'Dynamical Torque'  , linewidth = 0.8)
        plt.plot(*ComputeBinnedMeans(Final_Orbits, Buffer_Torque, args.Number_of_Averages), c = 'blue', label = 'Mean Buffer Torque', linewidth = 0.5)
        plt.axhline(y=0, c = 'black', linestyle='dashed')
        plt.axhline(y=np.mean(Binary_Torque), c = 'black', linewidth = 1.0, label = 'Mean Binary Torque', linestyle='dashdot')
        plt.xlabel('Time [P]')
        #plt.ylim([-4.0, 4.0])
        plt.ylabel(r'$\tau_\mathrm{b}/\dot{M}_0$')
        plt.legend(loc = 'best')

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')



    if args.Lightcurves:
        Infared        = ts.infared[-len(Final_Orbits):]
        Optical        = ts.optical[-len(Final_Orbits):]
        UV             = ts.uv[-len(Final_Orbits):]
        XRay           = ts.xray[-len(Final_Orbits):]
        Infared_signal = Infared - np.mean(Infared)
        Optical_signal = Optical - np.mean(Optical)
        UV_signal      = UV      - np.mean(UV)
        XRay_signal    = XRay    - np.mean(XRay)
        freq           = np.logspace(-2, 1, 1000)      # cycles / orbit
        omega          = 2 * np.pi * freq              # rad / orbit
        Infared_power  = lombscargle(Final_Orbits, Infared_signal, omega)
        Optical_power  = lombscargle(Final_Orbits, Optical_signal, omega)
        UV_power       = lombscargle(Final_Orbits, UV_signal     , omega)
        XRay_power     = lombscargle(Final_Orbits, XRay_signal   , omega)
        Infared_power /= np.var(Infared_signal)
        Optical_power /= np.var(Optical_signal)
        UV_power      /= np.var(UV_signal)
        XRay_power    /= np.var(XRay_signal)
        savename       = "Lightcurves"
        
        
        fig = plt.figure(figsize=(1.2 * text_width, 1.2 * 0.25*text_width))
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.5], hspace=0., wspace=0.0)
        ax0 = fig.add_subplot(gs[0])
        periodogram_grid = gs[1].subgridspec(3, 1, hspace=0.00)

        ax0.plot(Final_Orbits, ts.infared[-len(Final_Orbits):]   , c = 'red'       , label = 'Infared', linewidth = 0.8)
        ax0.plot(Final_Orbits, ts.optical[-len(Final_Orbits):]   , c = 'blue'      , label = 'Optical', linewidth = 0.8)
        ax0.plot(Final_Orbits, ts.uv[-len(Final_Orbits):]        , c = 'tab:purple', label = 'UV'     , linewidth = 0.8)
        ax0.set_xlabel('Time [P]')
        ax0.set_title('Emission Timeseries')
        ax0.set_yscale('log')
        ax0.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))

        band_periodograms = [
            ('Infrared', Infared_power, 'red'),
            ('Optical',  Optical_power, 'blue'),
            ('UV',       UV_power,     'tab:purple'),
        ]
        period_axes = []
        for idx, (label, power_arr, color) in enumerate(band_periodograms):
            sharex = period_axes[0] if period_axes else None
            ax_band = fig.add_subplot(periodogram_grid[idx, 0], sharex=sharex)
            norm = np.max(power_arr)
            normalized_power = power_arr / norm if norm > 0 else power_arr
            ax_band.plot(freq, normalized_power, c=color)
            ax_band.set_ylim([0, 1.1])
            ax_band.set_xscale('log')
            ax_band.set_ylabel(f'{label}')
            ax_band.yaxis.tick_right()
            ax_band.yaxis.set_label_position("right")
            ax_band.spines["right"].set_visible(True)
            ax_band.spines["left"].set_visible(False)
            if idx < len(band_periodograms) - 1:
                ax_band.set_xticklabels([])
            period_axes.append(ax_band)
        period_axes[-1].set_xlabel(r'$f\ \mathrm{[\Omega]}$')
        period_axes[0].set_title('Emission Periodograms (Normalised)')
        period_axes[0].axvline(x=0.5, color='gray', linestyle=':', linewidth = 0.75)
        period_axes[1].axvline(x=0.5, color='gray', linestyle=':', linewidth = 0.75)
        period_axes[2].axvline(x=0.5, color='gray', linestyle=':', linewidth = 0.75)
        
        
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

        fig, ax = plt.subplots(figsize=[text_width, column_width])
        plt.plot(Final_Orbits,SemiMajorAxis, label = 'Semi-major axis')
        plt.plot(Final_Orbits,Eccentricity , label = 'Eccentricity')
        plt.xlabel('Time [P]')
        plt.title('Orbital Elements')
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


    if args.Torque:
        Torque_g      = ts.torque_a[-len(Final_Orbits):] #/ M_dot_0
        Torque_a      = ts.torque_g[-len(Final_Orbits):] #/ M_dot_0
        Torque        = Torque_g + Torque_a
        signal        = Torque - np.mean(Torque)
        freq          = np.logspace(-1, 1, 1000)      # cycles / orbit
        omega         = 2 * np.pi * freq              # rad / orbit
        power         = lombscargle(Final_Orbits, signal, omega)
        power        /= np.var(signal)
        savename      = "Torque"
        
        fig = plt.figure(figsize=(1.2 * text_width, 1.2 * 0.25*text_width))
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.33], hspace=0., wspace=0.)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        ax0.plot(Final_Orbits, -Torque  , label='$\mathcal{T}$'   , linewidth = 0.4, c = 'royalblue')
        ax0.plot(Final_Orbits, -Torque_g, label=r'Grav', linewidth = 0.2, c = 'blue')
        ax0.plot(Final_Orbits, -Torque_a, label=r'Acc' , linewidth = 0.2, c = 'silver')
        ax0.plot(*ComputeBinnedMeans(Final_Orbits, Torque, args.Number_of_Averages), c = 'royalblue', label = 'Mean Torque', linewidth = 0.5, linestyle='dashed')
        ax0.set_xlabel('Time [P]')
        ax0.set_ylabel(r'$\mathcal{T}$')
        ax0.set_title(r'Torque Timeseries')
        ax0.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))

        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.spines["right"].set_visible(True)
        ax1.spines["left"].set_visible(False)
        ax1.plot(freq, power/np.max(power), c='black', linewidth = 0.1)
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$f\ \mathrm{[\Omega]}$')   # or just f
        ax1.set_ylabel('Power')
        ax1.set_title('Torque Periodogram')
        ax1.set_ylabel('Power')
        
        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
