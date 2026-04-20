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

def solve_newton_rapheson(f, g, x: float) -> float:
    n = 0
    while abs(f(x)) > 1e-15:
        x -= f(x) / g(x)
        n += 1
        if n > 10:
            raise ValueError("solve_newton_rapheson: no solution")
    return x

def eccentric_anomaly(time_since_periapse, e, a, M):
    """
    Compute the eccentric anomaly from the time since any periapse.
    """
    omega = (1.0 * M / a / a / a)**0.5
    P = 2.0 * np.pi / omega
    t = time_since_periapse - P * np.floor(time_since_periapse / P)
    n = omega * t                        # n := mean anomaly M
    f = lambda k: k - e * np.sin(k) - n  # k := eccentric anomaly E
    g = lambda k: 1.0 - e * np.cos(k)
    return solve_newton_rapheson(f, g, n)

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

def ComputeBinnedStats(times, field, Averaging_Window):
    """Return bin centers, means, and 1-sigma standard deviations."""
    num_bins    = int(np.ceil((times[-1] - times[0]) / Averaging_Window))
    bin_edges   = np.linspace(times[0], times[-1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means   = np.zeros(num_bins)
    bin_stds    = np.zeros(num_bins)

    for i in range(num_bins):
        bin_mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
        bin_data = field[bin_mask]
        if len(bin_data) > 1:
            bin_means[i] = np.mean(bin_data)
            bin_stds[i]  = np.std(bin_data)
        elif len(bin_data) == 1:
            bin_means[i] = bin_data[0]
            bin_stds[i]  = np.nan
        else:
            bin_means[i] = np.nan
            bin_stds[i]  = np.nan

    return bin_centers, bin_means, bin_stds

class DavidTimeseries:
    def __init__(self, Checkpoint):
        timeseries_data  = Checkpoint['timeseries']
        self.pointmasses = Checkpoint["point_masses"]
        self.currenttime = Checkpoint["time"] / 2 / np.pi 
        self.modelparams = Checkpoint['model_parameters'] 
        self.SS73        = Checkpoint['SS73']

        max_length       = max(len(arr) for arr in timeseries_data)
        ts               = np.array([np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in timeseries_data])

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
                self.mdot_flux_outer = np.array([s[25] for s in ts])
            except IndexError:
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
        "--Sweep",
        "-s",
        action='store_true',
        help="whether to plot the binary's torque timeseries",
    )
    parser.add_argument(
        "--Preferential_Accretion",
        "-pe",
        action='store_true',
        help="whether to plot the binary's preferential accretion timeseries",
    )
    parser.add_argument(
        "--Max_temperature",
        "-MT",
        action='store_true',
        help="whether to plot the maximum temperature of the disk",
    )
    parser.add_argument(
        "--StreamEfficiency",
        "-se",
        action='store_true',
        help="whether to plot the stream efficiency of the disk",
    )
    args     = parser.parse_args()
    filename = args.checkpoints[0]
    chkpt    = load_checkpoint(filename)
    ts       = DavidTimeseries(chkpt)

    # ======= Useful timeseries data ========
    SS73                = ts.SS73
    gamma               = ts.modelparams['gamma_law_index']
    alpha               = ts.modelparams["alpha"]
    cs_a                = (SS73.gamma * (SS73.surface_pressure_profile(r=1) / SS73.surface_density_profile(r=1)))**0.5
    nu_a                = SS73.alpha * cs_a**2
    M_dot_0             = SS73.Mdot_inf
    Final_Orbits        = ts.time[ts.time>(ts.currenttime-args.Number_of_Orbits)]

    if len(ts.pointmasses) == 1:
        Single              = True
        primary             = ts.pointmasses[0]
        m                   = primary.mass
        Accretion           = -(np.array(ts.mdot1[-len(Final_Orbits):])) / M_dot_0

    elif len(ts.pointmasses) == 2:
        Single              = False
        primary, secondary  = ts.pointmasses
        m1, m2              = primary.mass, secondary.mass
        M, G                = m1 + m2, 1.0
        q                   = m2 / m1                                            # mass ratio
        mu                  = m1 * m2 / M                                        # reduced mass
        Torque              = -(np.array(ts.torque_g[-len(Final_Orbits):]) + np.array(ts.torque_a[-len(Final_Orbits):])) / M_dot_0
        Power               = -(np.array(ts.power_g[-len(Final_Orbits):] ) + np.array(ts.power_a[-len(Final_Orbits):] )) / M_dot_0
        Accretion_1         = -(np.array(ts.mdot1[-len(Final_Orbits):])) / M_dot_0
        Accretion_2         = -(np.array(ts.mdot2[-len(Final_Orbits):])) / M_dot_0
        Mdot                = Accretion_1 + Accretion_2                          # total mass accretion rate
        a                   = np.array(ts.semimajor_axis[-len(Final_Orbits):])   # = a_rel (relative orbit semi-major axis)
        e                   = np.array(ts.eccentricity[-len(Final_Orbits):])
        E_Anomaly           = np.array([eccentric_anomaly(2*np.pi*Final_Orbits[ind], e[ind], a[ind], M) for ind in range(len(Final_Orbits))])
        cosE                = np.cos(E_Anomaly)
        ecosE               = e * cosE
        r_binary            = a * (1 - ecosE)                                    # binary separation
        E                   = - G * M * mu / (2*a)
        L                   = mu * np.sqrt(G * M * a * (1 - e**2))
        mudot               = Mdot * mu / M + (1 - q) * (Accretion_2 - q * Accretion_1) * mu / q / M
        v2                  = G * M * (2 / r_binary - 1 / a)
        Edot                = -0.5*mudot*v2 - G*M*mudot/r_binary - G*Mdot*mu / r_binary + Power
        Ldot                = Torque 
        adot                = Mdot / M + mudot / mu - Edot / E
        edot                = ((1-e**2) / (2*e+1e-5)) * (2*Mdot/M + 3*mudot/mu - 2*Ldot/L - Edot/E)

    # ======= Plotting Blocks ========
    if args.Accretion:
        cmap          = plt.cm.hot
        n_lines       = 3

        if Single:
            AccretionRate = Accretion
            fig = plt.figure(figsize=(1.2 * text_width, 1.2 * 0.25*text_width))
            gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.33], hspace=0., wspace=0.)
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax0.plot(Final_Orbits, AccretionRate, label=r'$\dot{M}$', linewidth = 1.0, c = 'tab:blue')


        else:
            AccretionRate = Accretion_1 + Accretion_2
            fig = plt.figure(figsize=(1.2 * text_width, 1.2 * 0.25*text_width))
            gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.33], hspace=0., wspace=0.)
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax0.plot(Final_Orbits, AccretionRate, label='$\dot{M}_\mathrm{t}$'      , linewidth = 1.0, c = cmap(0/n_lines)  )
            ax0.plot(Final_Orbits, Accretion_2  , label=r'$\dot{M}_2$'              , linewidth = 1.0, c = cmap(1.9/n_lines))
            ax0.plot(Final_Orbits, Accretion_1  , label=r'$\dot{M}_1$'              , linewidth = 1.0, c = cmap(0.7/n_lines), alpha = 0.7)
            try:
                FluxOuter = -ts.mdot_flux_outer[-len(Final_Orbits):] / M_dot_0
                ax0.plot(Final_Orbits, FluxOuter, label=r'$\dot{M}(r_\mathrm{buf})$', linewidth=0.8, c='royalblue' , linestyle='dashed')
            except AttributeError:
                pass

        signal        = AccretionRate - np.mean(AccretionRate)
        freq          = np.logspace(-2, 1, 1000)      # cycles / orbit
        omega         = 2 * np.pi * freq              # rad / orbit
        power         = lombscargle(Final_Orbits, signal, omega)
        power        /= np.var(signal)
        savename      = "Accretion"

        print('Mean accretion rate over window:', np.mean(AccretionRate))
        
        ax0.plot(*ComputeBinnedMeans(Final_Orbits, AccretionRate, args.Number_of_Averages), label='Binned Means', linewidth = 1.2, c = cmap(0/n_lines), linestyle='dashed')
        ax0.set_xlabel('Time [P]')
        ax0.set_ylabel(r'$\dot{M}/\langle\dot{M}_0\rangle$')
        ax0.set_title(r'Accretion Timeseries')
        ax0.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize='x-small')

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

    if args.Preferential_Accretion:
        Preferential_Accretion = (ts.mdot2 / ts.mdot1)[-len(Final_Orbits):]
        savename              = "PreferentialAccretion"

        fig, ax = plt.subplots(figsize=[text_width, 0.5*text_width])
        plt.plot(Final_Orbits, Preferential_Accretion, c = 'tab:orange', label = r'$(\dot{M}_2 / \dot{M}_1)$'     , linewidth = 0.8)
        #plt.plot(*ComputeBinnedMeans(Final_Orbits, Preferential_Accretion, args.Number_of_Averages), c = 'purple', label = 'Mean Preferential Accretion', linewidth = 0.5)
        plt.axhline(y=0, c = 'black', linestyle='dashed')
        plt.xlabel('Time [P]')
        plt.ylabel(r'$(\dot{M}_2 / \dot{M}_1)$')
        plt.legend(loc = 'best')

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')

    
    if args.StreamEfficiency:
        sign = 2 * (0.5 - chkpt['model_parameters']['retrograde'])
        cents, means, stds = ComputeBinnedStats(Final_Orbits, sign * Torque / Mdot, args.Number_of_Averages)
        fig, ax = plt.subplots(figsize=[text_width, 0.5*text_width])
        ax.plot(cents, means, c='tab:orange', label='Mean Stream Efficiency')
        ax.fill_between(cents, means - stds, means + stds, color='tab:orange', alpha=0.4)
        ax.axhline(y=0, c='black', linestyle='dashed')
        #ax.set_yscale('symlog', linthresh=0.1)
        ax.set_xlabel('Time [P]')
        ax.set_ylabel(r'$l$')
        ax.axhline(y=1.12, label='Ram Shock Efficiency', linestyle='dashed')
        ax.legend(loc='best')
        savename = "StreamEfficiency"
        
        ax.set_ylim([0.8 * np.min(means), 1.2 * np.max(means)])

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')




    if args.OrbitalEvolution:
        savename = 'OrbitalEvolution'
        fig, ax  = plt.subplots(figsize=[text_width, column_width])
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








    if args.Sweep:
        savename = 'Sweep'
        _blue    = '#2166AC'   # ColorBrewer RdBu deep blue
        _red     = '#B2182B'   # ColorBrewer RdBu deep crimson

        ea, ma, _ = ComputeBinnedStats(e, adot, 0.05)
        ee, me, _ = ComputeBinnedStats(e, edot, 0.05)

        fig, ax = plt.subplots(figsize=[text_width, column_width])
        ax.plot(ea, ma, linewidth=1.8, linestyle='solid', color=_blue, label=r'$\dot{a}/a$', zorder=3)
        ax.plot(ee, me, linewidth=1.8, linestyle='solid', color=_red,  label=r'$\dot{e}$',   zorder=3)
        ax.scatter(ea, ma, color=_blue, label=r'$\dot{a}/a$')
        ax.scatter(ee, me, color=_red,  label=r'$\dot{e}$'  )
        ax.autoscale()
        _ylo, _yhi = ax.get_ylim()

        ax.set_ylim(_ylo, _yhi)   
        ax.axhline(y=0, color='0.5', linewidth=0.8, linestyle='--', zorder=2)

        ax.set_xlabel('Orbital Eccentricity $e$')
        ax.set_ylabel(r'$\dot{a}/a,\ \dot{e}\ [\Omega_0]$')
        ax.set_title(r'Mach 10, $n = 2500$', fontsize=8)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', top=False, right=False)
        ax.minorticks_on()
        ax.tick_params(which='minor', length=2.5, direction='in')
        fig.tight_layout()

        # ax.scatter(0.0, -10.14, color=_blue, marker='x', s=50, zorder=4)
        # ax.scatter(0.3, -13.16, color=_blue, marker='x', s=50, zorder=4)
        # ax.scatter(0.3,  -2.59, color=_red , marker='x', s=50, zorder=4)
        # ax.scatter(0.6, -17.81, color=_blue, marker='x', s=50, zorder=4)
        # ax.scatter(0.6,  -0.84, color=_red , marker='x', s=50, zorder=4)

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(ts.currenttime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(ts.currenttime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')



        # --- EM panels: L(e) line (left) + 2D LS spectrogram P(e, f) (right) ---
        _bands = [
            (ts.infared[-len(Final_Orbits):],  r'$L_\mathrm{IR}$',  r'$L_\mathrm{IR}\ /\ \dot{M}_0$',  '#D95F02', 'Oranges'),
            (ts.optical[-len(Final_Orbits):],  r'$L_\mathrm{opt}$', r'$L_\mathrm{opt}\ /\ \dot{M}_0$', '#1B7837', 'Greens' ),
            (ts.uv[-len(Final_Orbits):],       r'$L_\mathrm{UV}$',  r'$L_\mathrm{UV}\ /\ \dot{M}_0$',  '#762A83', 'Purples'),
        ]

        # eccentricity bins for the 2D spectrogram — one per 0.01 in e
        _n_e      = 100
        _e_edges  = np.linspace(0.0, 1.0, _n_e + 1)
        _e_ctrs   = 0.5 * (_e_edges[:-1] + _e_edges[1:])
        _freq_2d  = np.logspace(-1.5, 1, 500)           # cycles / orbit, 0.1 – 10
        _omega_2d = 2.0 * np.pi * _freq_2d
        _min_samp = 25                                # min points per bin for reliable LS

        # pre-compute L(e) stats and per-bin periodograms for each band
        _band_stats = []
        for lum_raw, band_label, ylabel, color, cmap in _bands:
            lum        = lum_raw / M_dot_0
            ec, lm, ls = ComputeBinnedStats(e, lum, 0.01)
            power_2d   = np.full((_n_e, len(_freq_2d)), np.nan)

            for j in range(_n_e):
                mask = (e >= _e_edges[j]) & (e < _e_edges[j + 1])
                if mask.sum() >= _min_samp:
                    t_bin   = Final_Orbits[mask]
                    sig_bin = lum_raw[mask] - np.mean(lum_raw[mask])
                    var_bin = np.var(sig_bin)
                    if var_bin > 0:
                        p          = lombscargle(t_bin, sig_bin, _omega_2d)
                        p         /= var_bin          # Scargle-normalised LS power
                        med        = np.median(p)
                        if med > 0:
                            p /= med                  # excess above noise floor; stable w.r.t. freq resolution
                        power_2d[j] = p

            _band_stats.append((ec, lm, ls, power_2d, band_label, ylabel, color, cmap))

        fig2 = plt.figure(figsize=[text_width, text_width * 0.75])
        gs   = fig2.add_gridspec(3, 2, width_ratios=[1.0, 1.0],
                                 hspace=0.12, wspace=0.08,
                                 left=0.10, right=0.92, top=0.93, bottom=0.10)

        ax_lum_ref = None
        ax_per_ref = None
        last_im    = None

        for i, (ec, lm, ls, power_2d, band_label, ylabel, color, cmap) in enumerate(_band_stats):

            ax_lum = fig2.add_subplot(gs[i, 0], sharex=ax_lum_ref)
            ax_per = fig2.add_subplot(gs[i, 1], sharex=ax_per_ref)
            if ax_lum_ref is None:
                ax_lum_ref = ax_lum
            if ax_per_ref is None:
                ax_per_ref = ax_per

            # left: mean luminosity ± 1σ vs eccentricity
            ax_lum.fill_between(ec, lm - ls, lm + ls,
                                 color=color, alpha=0.2, linewidth=0, zorder=1)
            ax_lum.plot(ec, lm, linewidth=1.8, color=color, label=band_label, zorder=3)
            ax_lum.set_yscale('log')
            ax_lum.set_ylabel(ylabel, fontsize=7)
            ax_lum.set_xlim(0.0, 1.0)
            ax_lum.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none')
            ax_lum.spines['top'].set_visible(False)
            ax_lum.spines['right'].set_visible(False)
            ax_lum.tick_params(which='both', top=False, right=False)
            ax_lum.minorticks_on()
            ax_lum.tick_params(which='minor', length=2.5, direction='in')
            # if _e_star is not None:
            #     ax_lum.axvline(_e_star, color='0.4', linewidth=0.9, linestyle=':', zorder=2)
            if i < 2:
                ax_lum.tick_params(labelbottom=False)

            # right: 2D LS spectrogram — P(e, f) as pcolormesh
            from matplotlib.colors import LogNorm
            im = ax_per.pcolormesh(_e_ctrs, _freq_2d, power_2d.T,
                                   cmap=cmap, shading='auto', rasterized=True,
                                   norm=LogNorm(vmin=1.0, vmax=10.0))
            last_im = im
            ax_per.set_yscale('log')
            ax_per.set_ylim(_freq_2d[0], _freq_2d[-1])
            ax_per.set_xlim(0.0, 1.0)
            #ax_per.axhline(1.0, color='white', linewidth=0.8, linestyle='--', zorder=3)  # Ω₀
            #ax_per.axhline(2.0, color='white', linewidth=0.6, linestyle=':',  zorder=3)  # 2Ω₀
            # if _e_star is not None:
            #     ax_per.axvline(_e_star, color='white', linewidth=0.9, linestyle=':', zorder=3)
            ax_per.set_ylabel(r'Frequency $[\Omega_0]$', fontsize=7)
            ax_per.yaxis.set_label_position('right')
            ax_per.yaxis.tick_right()
            ax_per.tick_params(which='both', top=False, left=False)
            ax_per.minorticks_on()
            ax_per.tick_params(which='minor', length=2.5, direction='in')

            if i == 0:
                ax_lum.set_title('Averaged Band Luminosities', fontsize=7)
                ax_per.set_title('Band Periodograms (Normalised to Unity)', fontsize=7)

            if i == 2:
                ax_lum.set_xlabel(r'Orbital Eccentricity $e$', fontsize=7)
                ax_per.set_xlabel(r'Orbital Eccentricity $e$', fontsize=7)
            else:
                ax_lum.tick_params(labelbottom=False)
                ax_per.tick_params(labelbottom=False)
        savename2 = 'Sweep-EM'

        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname2 = os.path.join(args.Outputs, f"{savename2}-{int(ts.currenttime * 100):05d}.png")
            fig2.savefig(pngname2, dpi=400, bbox_inches='tight')
        else:
            pngname2 = args.Outputs + savename2 + f"-{int(ts.currenttime * 100):05d}.png"
            fig2.savefig(pngname2, dpi=400, bbox_inches='tight')



    if args.Lightcurves:
        Infared        = ts.infared[-len(Final_Orbits):]
        Optical        = ts.optical[-len(Final_Orbits):]
        UV             = ts.uv[-len(Final_Orbits):]
        XRay           = ts.xray[-len(Final_Orbits):]
        Infared_signal = Infared - np.mean(Infared)
        Optical_signal = Optical - np.mean(Optical)
        UV_signal      = UV      - np.mean(UV)
        XRay_signal    = XRay    - np.mean(XRay)
        freq           = np.logspace(-1, 1, 1000)      # cycles / orbit
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

        #'#D95F02', 'Oranges'
        #'#1B7837', 'Greens' 
        #'#762A83', 'Purples'
        
        
        fig = plt.figure(figsize=(1.2 * text_width, 1.2 * 0.25*text_width))
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.5], hspace=0., wspace=0.0)
        ax0 = fig.add_subplot(gs[0])
        periodogram_grid = gs[1].subgridspec(3, 1, hspace=0.00)

        ax0.plot(Final_Orbits, ts.infared[-len(Final_Orbits):]   , c = '#D95F02', label = 'Infared', linewidth = 0.8)
        ax0.plot(Final_Orbits, ts.optical[-len(Final_Orbits):]   , c = '#1B7837', label = 'Optical', linewidth = 0.8)
        ax0.plot(Final_Orbits, ts.uv[-len(Final_Orbits):]        , c = '#762A83', label = 'UV'     , linewidth = 0.8)
        ax0.set_xlabel('Time [P]')
        ax0.set_title('Emission Timeseries')
        ax0.set_yscale('log')
        ax0.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))

        band_periodograms = [
            ('Infrared', Infared_power, '#D95F02'),
            ('Optical',  Optical_power, '#1B7837'),
            ('UV',       UV_power,     '#762A83'),
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

        print('Current orbital eccentricity is', Eccentricity[-1])
        
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
