import argparse
import pickle
import sys
from sailfish.physics.kepler import OrbitalState

sys.path.insert(1,"/groups/astro/davidon/sailfish/")
import sailfish


sys.path.insert(1, ".")


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as file:
        chkpt = pickle.load(file)

        if require_solver is not None and chkpt["solver"] != require_solver:
            raise ValueError(
                f"checkpoint is from a run with solver {chkpt['solver']}, "
                f"expected {require_solver}"
            )
        return chkpt


def main_srhd_1d():
    import matplotlib.pyplot as plt
    from sailfish.mesh import LogSphericalMesh

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    args = parser.parse_args()

    fig, ax = plt.subplots()

    for filename in args.checkpoints:
        chkpt = load_checkpoint(filename, require_solver="srhd_1d")

        mesh = chkpt["mesh"]
        x = mesh.zone_centers(chkpt["time"])
        rho = chkpt["primitive"][:, 0]
        vel = chkpt["primitive"][:, 1]
        pre = chkpt["primitive"][:, 2]
        ax.plot(x, rho, label=r"$\rho$")
        ax.plot(x, vel, label=r"$\Gamma \beta$")
        ax.plot(x, pre, label=r"$p$")

    if type(mesh) == LogSphericalMesh:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend()
    plt.show()


def main_srhd_2d():
    import matplotlib.pyplot as plt
    import numpy as np
    import sailfish

    fields = {
        "ur": lambda p: p[..., 1],
        "uq": lambda p: p[..., 2],
        "rho": lambda p: p[..., 0],
        "pre": lambda p: p[..., 3],
        "e": lambda p: p[..., 3] / p[..., 0] * 3.0,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="ur",
        choices=fields.keys(),
        help="which field to plot",
    )
    parser.add_argument(
        "--radial-coordinates",
        "-c",
        type=str,
        default="comoving",
        choices=["comoving", "proper"],
        help="plot in comoving or proper (time-independent) radial coordinates",
    )
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )

    args = parser.parse_args()

    for filename in args.checkpoints:
        fig, ax = plt.subplots()

        chkpt = load_checkpoint(filename, require_solver="srhd_2d")
        mesh = chkpt["mesh"]
        prim = chkpt["primitive"]

        t    = chkpt["time"]
        r, q = np.meshgrid(mesh.radial_vertices(t), mesh.polar_vertices)
        z = r * np.cos(q)
        x = r * np.sin(q)
        f = fields[args.field](prim).T

        if args.radial_coordinates == "comoving":
            x[...] /= mesh.scale_factor(t)
            z[...] /= mesh.scale_factor(t)

        if args.log:
            f = np.log10(f)

        cm = ax.pcolormesh(
            x,
            z,
            f,
            edgecolors="none",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap="plasma",
        )

        ax.set_aspect("equal")
        # ax.set_xlim(0, 1.25)
        # ax.set_ylim(0, 1.25)
        fig.colorbar(cm)
        fig.suptitle(filename)

    plt.show()


def main_cbdiso_2d():
    import matplotlib.pyplot as plt
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "torque": None,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="sigma",
        #choices=fields.keys(),
        help="which field to plot",
    )
    parser.add_argument("--poly", type=int, nargs=2, default=None)
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--scale-by-power",
        "-s",
        default=None,
        type=float,
        help="scale the field by the given power",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="colormap name",
    )
    parser.add_argument(
        "--radius",
        default=None,
        type=float,
        help="plot the domain out to this radius",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save PNG files instead of showing a window",
    )
    parser.add_argument(
        "--draw-lindblad31-radius",
        action="store_true",
    )
    parser.add_argument(
        "--vmap",
        action="store_true",
        help="plot velocity vectors",
    )
    parser.add_argument(
        "--print_model_parameters",
        "-params",
        action="store_true",
        help="plot the parameters used for making this checkpoint",
    )
    parser.add_argument("-m", "--print-model-parameters", action="store_true")
    args = parser.parse_args()



    class TorqueCalculation:
        def __init__(self, mesh, masses):
            self.mesh = mesh
            self.masses = masses

        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            dx = mesh.dx
            dy = mesh.dy
            da = dx * dy
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]

            x1 = self.masses[0].position_x
            y1 = self.masses[0].position_y
            x2 = self.masses[1].position_x
            y2 = self.masses[1].position_y
            m1 = self.masses[0].mass
            m2 = self.masses[1].mass
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length

            sigma = primitive[:, :, 0]
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2

            # forces on the gas
            fx1 = -sigma * da * m1 * delx1 / (delx1**2 + dely1**2 + rs1**2) ** 1.5
            fy1 = -sigma * da * m1 * dely1 / (delx1**2 + dely1**2 + rs1**2) ** 1.5
            fx2 = -sigma * da * m2 * delx2 / (delx2**2 + dely2**2 + rs2**2) ** 1.5
            fy2 = -sigma * da * m2 * dely2 / (delx2**2 + dely2**2 + rs2**2) ** 1.5

            t1 = x * fy1 - y * fx1
            t2 = x * fy2 - y * fx2
            t = t1 + t2
            print("total torque:", t.sum())
            return np.abs(t) ** 0.125 * np.sign(t)



    class VelocityQuantities():
        
        def __init__(self, mesh, Vx, Vy):
            self.mesh = mesh
            self.Vx   = Vx
            self.Vy   = Vy


        def Mesh(self):
            mesh = self.mesh
            ni, nj = mesh.shape
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            return x,y

        def VMap(self, Number_of_Vectors=400):
            x, y                 = self.Mesh()
            X, Y                 = np.meshgrid(x, y)
            Sampling             = np.arange(0, len(x)-1, len(x)//Number_of_Vectors)
            
            X_sampled = x[Sampling,0]
            #print(x[Sampling])
            Y_sampled =  y[0,Sampling]
            #print(np.shape(y))

            Vx_sampled = self.Vx[::len(x)//Number_of_Vectors, ::len(x)//Number_of_Vectors] 
            Vy_sampled = self.Vy[::len(y[0])//Number_of_Vectors, ::len(y[0])//Number_of_Vectors]

            plt.quiver(X_sampled, Y_sampled, Vx_sampled, Vy_sampled,width=0.001, color = 'lightcyan', scale=100)

        def Speed(self,t):
            Vx_Relative = self.Vx + 0.5 * np.sin(t)
            Vy_Relative = self.Vy - 0.5 * np.cos(t)

            f = np.sqrt( Vx_Relative**2 + Vy_Relative**2 )

            # Make radial and aximuthal?
            return f

        def Vortensity(self):
            x, y = self.Mesh()

            #x * self.Vy - y * self.Vx
        

            #dx = (mesh.x1 - mesh.x0) / np.shape(fields["vx"](prim).T)[0]
            #dy = (mesh.y1 - mesh.y0) / np.shape(fields["vx"](prim).T)[1]





    for filename in args.checkpoints:
        fig, ax = plt.subplots(figsize=[12, 9])
        chkpt   = load_checkpoint(filename)
        
        mesh             = chkpt["mesh"]
        fields["torque"] = TorqueCalculation(mesh, chkpt["point_masses"])
        

        if chkpt["solver"] == "cbdisodg_2d":
            prim = chkpt["primitive"]
            if args.poly is None:
                prim = chkpt["primitive"]
                f = fields[args.field](prim).T
            else:
                m, n = args.poly
                f = chkpt["solution"][:, :, 0, m, n].T
        else:
            # the cbdiso_2d solver uses primitive data as the solution array
            prim = chkpt["solution"]


        Vx               = fields["vx"](prim).T
        Vy               = fields["vy"](prim).T
        Velocities       = VelocityQuantities(mesh, Vx, Vy)

        if args.field == 'speed':
            f    = Velocities.Speed(chkpt["time"])
            
        elif args.field == 'vortensity':
            pass

        else:
            f = fields[args.field](prim).T


        if args.vmap:
            Velocities.VMap()

        if args.print_model_parameters:
            print('Iteration Number.........',chkpt['iteration'])
            print('Timestep_dt..............',chkpt['timestep_dt'])
            print('cfl_number...............',chkpt['cfl_number'])
            print('Solver options...........',chkpt['solver_options'])
            print('Event states.............',chkpt['event_states'])
            print('------------------Driver------------------')
            print(chkpt['driver'])
            print('-------------Model Parameters-------------')
            print(chkpt["model_parameters"])
            print('---------------Point Masses---------------')
            print(chkpt["point_masses"])
            print('------------------------------------------')



            

        if args.scale_by_power is not None:
            f = f**args.scale_by_power
        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        cm     = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            extent=extent,
        )

        primary, secondary = chkpt['point_masses']

        #ax.scatter(primary.position_x, primary.position_y, marker = 'o', s = 30, c = 'black')
        #ax.scatter(secondary.position_x, secondary.position_y, marker = 'o', s = 30, c = 'black')
        ax.scatter(primary.position_x, primary.position_y, marker = '+', s = 40, c = 'white', label = 'Point Mass')
        ax.scatter(secondary.position_x, secondary.position_y, marker = '+', s = 40, c = 'white')
        ax.legend()
        if args.draw_lindblad31_radius:
            x1 = chkpt["point_masses"][0].position_x
            y1 = chkpt["point_masses"][0].position_y
            t = np.linspace(0, 2 * np.pi, 1000)
            x = x1 + 0.3 * np.cos(t)
            y = y1 + 0.3 * np.sin(t)
            a = 1.0
            q = chkpt["model_parameters"]["mass_ratio"]
            # Eq. 1 in Franchini & Martin (2019; https://arxiv.org/pdf/1908.02776.pdf)
            r_res = 3 ** (-2 / 3) * (1 + q) ** (-1 / 3) * a
            ax.plot(x, y, ls="--", lw=0.75, c="w", alpha=1.0)

        ax.set_aspect("equal")
        if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)
        fig.colorbar(cm)
        fig.suptitle(chkpt["time"]/2/np.pi)
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )
        if args.save:

            import os
            CurrentTime = load_checkpoint(filename)["time"]/ 2 / np.pi
            pngname     = os.getcwd() + f"{'/Outputs/DensityMap'}.{int(100*CurrentTime)}.png"
            print('Figure at time', f"{int(1000*CurrentTime):04d}")
            fig.savefig(pngname, dpi=400)

            if args.field == 'speed':
                fig, ax = plt.subplots(figsize=[12, 9])
                ni, nj  = mesh.shape
                xspace  = np.linspace(mesh.x0,mesh.x1,ni)
                yspace  = np.linspace(mesh.y0,mesh.y1,nj)
                ax.plot(xspace,f[nj//2,:], label = 'horizontal cut')
                ax.plot(yspace,f[:,ni//2], label = 'vertical cut')
                ax.axvline(x = 0.53, linewidth = 0.1, c = 'black')
                ax.axvline(x = 0.47, linewidth = 0.1, c = 'black')
                plt.legend()
                plt.title('Velocity Profile for a Retrograde Disk')
                plt.ylabel(r'$\|v_\mathrm{gas}\|~\left[a\Omega\right]$')
                plt.xlabel(r'$x, y~\left[a_0\right]$')
                plt.xlim([-1,1])
                pngname = os.getcwd() + f"{'/Outputs/VelocityCuts'}.{int(100*CurrentTime):04d}.png"
                fig.savefig(pngname, dpi=400)

    if not args.save:
        plt.show()



def main_cbdisodg_2d():
    main_cbdiso_2d()


def main_cbdgam_2d():
    import matplotlib.pyplot as plt
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pre": lambda p: p[:, :, 3],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="sigma",
        choices=fields.keys(),
        help="which field to plot",
    )
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )

    args = parser.parse_args()

    for filename in args.checkpoints:
        fig, ax = plt.subplots(figsize=[10, 10])
        chkpt = load_checkpoint(filename, require_solver="cbdgam_2d")
        mesh = chkpt["mesh"]
        prim = chkpt["solution"]
        f = fields[args.field](prim).T

        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        cm = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap="magma",
            extent=extent,
        )
        ax.set_aspect("equal")
        fig.colorbar(cm)
        fig.suptitle(filename)

    plt.show()

if __name__ == "__main__":
    for arg in sys.argv:
        if arg.endswith(".pk"):
            chkpt = load_checkpoint(arg)
            
            #prim, sec = chkpt['point_masses']
            import numpy as np
            print('Time',chkpt['time']/2/np.pi)
            print('Semi-Major axis',np.array([s[ 1] for s in chkpt['timeseries']])[-1])
            if chkpt["solver"] == "srhd_1d":
                print("plotting for srhd_1d solver")
                exit(main_srhd_1d())
            if chkpt["solver"] == "srhd_2d":
                print("plotting for srhd_2d solver")
                exit(main_srhd_2d())
            if chkpt["solver"] == "cbdiso_2d":
                print("plotting for cbdiso_2d solver")
                exit(main_cbdiso_2d())
            if chkpt["solver"] == "cbdisodg_2d":
                print("plotting for cbdisodg_2d solver")
                exit(main_cbdisodg_2d())
            if chkpt["solver"] == "cbdgam_2d":
                print("plotting for cbdgam_2d solver")
                exit(main_cbdgam_2d())
            else:
                print(f"Unknown solver {chkpt['solver']}")
        
