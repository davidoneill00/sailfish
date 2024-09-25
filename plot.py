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
        "--Outputs",
        "-o",
        default=None,
	    type=str,
	    help="Where to save the output png files",
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
        "--CorotatingFrame",
        "-cf",
        action="store_true",
        default=False,
        help="plot velocity vectors",
    )
    parser.add_argument(
        "--print_model_parameters",
        "-params",
        action="store_true",
        help="plot the parameters used for making this checkpoint",
    )
    parser.add_argument(
        "--AngularSpeed",
        action="store_true",
        help="plot the orbital speed of a minidisk",
    )
    parser.add_argument("-m", "--print-model-parameters", action="store_true")
    args = parser.parse_args()


    class TorqueCalculation:
        def __init__(self, mesh, masses):
            self.mesh = mesh
            self.masses = masses

        def __call__(self, primitive):
            mesh   = self.mesh
            ni, nj = mesh.shape
            dx     = mesh.dx
            dy     = mesh.dy
            da     = dx * dy
            x      = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y      = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]

            x1  = self.masses[0].position_x
            y1  = self.masses[0].position_y
            x2  = self.masses[1].position_x
            y2  = self.masses[1].position_y
            m1  = self.masses[0].mass
            m2  = self.masses[1].mass
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
        
        def __init__(self, mesh, Vx, Vy, t, Corotating):
            self.mesh       = mesh
            self.Corotating = Corotating
            self.t          = 2*np.pi*t
            
            if self.Corotating:
                self.Vx   = Vx + 0.5 * np.sin(self.t)
                self.Vy   = Vy - 0.5 * np.cos(self.t)
            else:
                self.Vx   = Vx
                self.Vy   = Vy

        def Mesh(self):
            mesh = self.mesh
            ni, nj = mesh.shape

            if self.Corotating:
                x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)]) #+ 0.5 * np.cos(self.t)
                y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)]) #+ 0.5 * np.sin(self.t)
            else:
                x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])
                y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])
            return x,y

        def VMap(self, Number_of_Vectors=40):
            x, y        = self.Mesh()

            try:
                rescaled_x = [ix for ix in x if np.abs(ix) < args.radius]
                xmin, xmax = np.where(x == np.min(rescaled_x))[0][0], np.where(x == np.max(rescaled_x))[0][0]
            except:
                rescaled_x = x
                xmin, xmax = 0,len(x)-1

            Sampling = (xmax-xmin)//Number_of_Vectors

            if len(rescaled_x)//Number_of_Vectors == 0:
                #raise ZeroDivisionError("Too many Vectors in this domain")
                Sampling = 1

            X, Y       = np.meshgrid(x[xmin:xmax:Sampling], y[xmin:xmax:Sampling])

            Vx_sampled = self.Vx[xmin:xmax:Sampling, xmin:xmax:Sampling] 
            Vy_sampled = self.Vy[xmin:xmax:Sampling, xmin:xmax:Sampling]# - 0.5

            #plt.quiver(X, Y, Vx_sampled, Vy_sampled,width=0.001, scale=200)
            plt.quiver(X, Y, Vx_sampled, Vy_sampled,width=0.001, scale=60, color = 'lightblue')

            


        def AngularSpeed(self):
            x, y        = self.Mesh()

            primary, secondary = chkpt['point_masses']
            xprim,yprim        = primary.position_x, primary.position_y
            xsec,ysec          = secondary.position_x, secondary.position_y

            XSecCent  = np.array(x)[:,0] + xsec
            YSecCent  = np.array(y)[0,:] + ysec
            XPrimCent = np.array(x)[:,0] + xsec
            YPrimCent = np.array(y)[0,:] + ysec

            XCent, YCent = np.meshgrid(XPrimCent,YPrimCent)
            #XCent, YCent = np.meshgrid(XSecCent,YSecCent)

            Vx_Relative = self.Vx 
            Vy_Relative = self.Vy 

            f = (XCent * Vy_Relative - YCent * Vx_Relative)/(XCent**2 + YCent**2) # w = (r x v) / r^2
            
            return f



        def Vortensity(self):
            x, y   = self.Mesh()
            dVy_dx = np.gradient(self.Vy, axis=1)  # Partial derivative of Vy with respect to x
            dVx_dy = np.gradient(self.Vx, axis=0)  # Partial derivative of Vx with respect to y
            f      = dVy_dx - dVx_dy

            return f #Ignoring 1/Sigma here





    for filename in args.checkpoints:
        fig, ax     = plt.subplots(figsize=[12, 9])
        chkpt       = load_checkpoint(filename)
        CurrentTime = load_checkpoint(filename)["time"]/ 2 / np.pi
        
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
        Velocities       = VelocityQuantities(mesh, Vx, Vy, t = CurrentTime, Corotating = args.CorotatingFrame)

        if args.field == 'speed':
            f    = Velocities.AngularSpeed()
            
        elif args.field == 'vortensity':
            sigma = fields['sigma'](prim).T
            f     = Velocities.Vortensity()/sigma

        else:
            #if args.CorotatingFrame:
            #    #xprim = chkpt['point']
            #    #yprim = 
            #    f = fields[args.field](prim).T[]
            #else:
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



        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        if args.field == 'vortensity' and args.log:
            cm     = ax.imshow(
                np.log10(f),
                origin="lower",
                vmin=args.vmin,
                vmax=args.vmax,
                cmap='Reds',
                extent=extent,
            )
            cm2     = ax.imshow(
                np.log10(-f),
                origin="lower",
                vmin=args.vmin,
                vmax=args.vmax,
                cmap='Blues',
                extent=extent,
            )
            #cbar_ax  = fig.add_axes([0.76, 0.64, 0.03, 0.24])
            colorbar1      = fig.colorbar(cm , cax=fig.add_axes([0.85, 0.51, 0.03, 0.42]))
            colorbar1.set_label(r'$\log_{10}(\zeta)$', rotation=0, labelpad =30)
            colorbar2      = fig.colorbar(cm2, cax=fig.add_axes([0.85, 0.07, 0.03, 0.42]))
            colorbar2.set_label(r'$\log_{10}(-\zeta)$', rotation=0, labelpad =30)
            colorbar2.ax.invert_yaxis()

        else:
            if args.scale_by_power is not None:
                f = f**args.scale_by_power
            if args.log:
                f = np.log10(f)

            cm     = ax.imshow(
                f,
                origin="lower",
                vmin=args.vmin,
                vmax=args.vmax,
                cmap=args.cmap,
                extent=extent,
            )
            
            fig.colorbar(cm)
                

        primary, secondary = chkpt['point_masses']

        ax.scatter(primary.position_x, primary.position_y, marker = '+', s = 40, c = 'white', label = 'Point Mass')
        #ax.axhline(y=0, linestyle='dashed', c = 'gray', label = 'x cut')
        ax.scatter(secondary.position_x, secondary.position_y, marker = '+', s = 40, c = 'white')
        #ax.legend()
        ax.text(
                0.8, 0.95,  # Relative coordinates (x=5% from left, y=95% from bottom)
                r'$t = %g$'%(int(chkpt["time"]/ 2 / np.pi)),
                fontsize=24,
                transform=ax.transAxes,  
                verticalalignment='top',  
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
            )

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
        fig.suptitle(chkpt["time"]/2/np.pi)
        #fig.suptitle(r'Angular Speed of a Retrograde Minidisk $\log_{10}{\Omega(r)}$')
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )

        #if args.CorotatingFrame:
        #    xmin, xmax = ax.get_xlim()
        #    ymin, ymax = ax.get_ylim()

        #    xprim = primary.position_x
        #    yprim = primary.position_y
        #    ax.set_xlim(xmin + xprim, xmax + xprim)
        #    ax.set_ylim(ymin + yprim, ymax + yprim)

        import os
        try:
            pngname     = args.Outputs + f"{'/DensityMap'}.{int(100*CurrentTime)}.png"
            fig.savefig(pngname, dpi=400)

        except:
            plt.show()



        if args.AngularSpeed:
            x,y = Velocities.Mesh()
            plt.figure()
            plt.plot(x,f[1500,:], label = r'Angular Speed $x =0$ cut')
            plt.plot(x,[1/np.sqrt(2 * np.abs(ix-0.5)**3) for ix in x], label = r'$\Omega_K(r)$')
            plt.ylim([-1,150])
            plt.xlim([0.,1])
            plt.axvline(x = 0.47, linestyle='dashed', c = 'gray', label = 'Sink radius', linewidth = 0.5)
            plt.axvline(x = 0.53, linestyle='dashed', c = 'gray', linewidth = 0.5)
            plt.axvline(x = 0.45, linestyle='dashed', c = 'red', label = 'Minidisk radius', linewidth = 0.5)
            plt.axvline(x = 0.55, linestyle='dashed', c = 'red', linewidth = 0.5)
            plt.legend()
            plt.savefig(args.Outputs + "/AngularSpeed_of_RetrogradeMinidisk.png", dpi = 300)


        
        #with open(args.Outputs + "/SavedData_nu%g_t%g.txt"%(chkpt["model_parameters"]["nu"],int(CurrentTime)), "w") as file:
        #    np.savetxt(file, f.flatten(), fmt="%f")
        #    file.close()



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
            #print('Semi-Major axis',np.array([s[ 1] for s in chkpt['timeseries']])[-1])
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
        
