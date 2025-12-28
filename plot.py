import argparse
import pickle as pk
import sys
import sailfish
from sailfish.physics.kepler import OrbitalState
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sailfish.solvers.scdg_1d import Physics

text_width   = 5.0
column_width = text_width / 2.
def configure_matplotlib():
    plt.rc('xtick' , labelsize=8)
    plt.rc('ytick' , labelsize=8)
    plt.rc('axes'  , labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', family='DejaVu Sans', size=8)
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.color": "black",
        "ytick.color": "black",
    })
    plt.rcParams.update({
        "axes.linewidth": 2,  # default is 0.8
    })
configure_matplotlib()


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = pk.load(file=f)
    return chkpt


def main_srhd_1d():
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
    parser.add_argument(
        "--pressure",
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


        def VMap(self, Number_of_Vectors=20):
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
            plt.quiver(X, Y, Vx_sampled, Vy_sampled,width=0.0025, angles='xy', scale_units='xy', scale=40, color = 'darkgrey')

            


        def AngularSpeed(self):

            Vx_Relative = self.Vx
            Vy_Relative = self.Vy

            TotalSpeed = np.sqrt( Vx_Relative**2 + Vy_Relative**2 )


            fig, ax = plt.subplots(figsize=[text_width, 0.75*text_width])
            ni, nj = mesh.shape
            xspace = np.linspace(mesh.x0,mesh.x1,ni)
            yspace = np.linspace(mesh.y0,mesh.y1,nj)
            ax.plot(xspace,f[nj//2,:], label = 'horizontal cut')
            ax.plot(yspace,f[:,ni//2], label = 'vertical cut')
            plt.legend()
            plt.title('Velocity Profile for a Retrograde Disk')
            plt.ylabel(r'$\|v_\mathrm{gas}\|~\left[a\Omega\right]$')
            plt.xlabel(r'$x, y~\left[a_0\right]$')
            plt.xlim([-2,2])
            pngname     = os.getcwd() + f"{'/Outputs/VelocityCuts'}.{int(np.round(100*CurrentTime,3)):04d}.png"
            fig.savefig(pngname, dpi=400)


            x, y        = self.Mesh()
            TotalSpeed  = self.Vy

            primary, secondary = chkpt['point_masses']
            xprim,yprim        = primary.position_x , primary.position_y
            xsec,ysec          = secondary.position_x, secondary.position_y

            XSecCent  = np.array(x) #+ xsec
            YSecCent  = np.array(y) #+ ysec
            XPrimCent = np.array(x) - xprim
            YPrimCent = np.array(y) #+ yprim

            XCent, YCent = np.meshgrid(XPrimCent,YPrimCent)
            #XCent, YCent = np.meshgrid(XSecCent,YSecCent)

            f = (XCent * self.Vy - YCent * self.Vx)/(XCent**2 + YCent**2) # w = (r x v) / r^2
            
            plt.figure(figsize = (text_width, text_width))
            plt.plot(XPrimCent, TotalSpeed[nj//2,:]/XPrimCent, c = 'black', linewidth = 2, label = r'$\Omega(r)$')
            plt.plot(XPrimCent, [- np.sqrt(np.sign(xpos) / xpos /xpos /xpos) for xpos in XPrimCent], c = 'red', label = r'$\Omega_K(r)$')
            plt.plot(XPrimCent, [np.sqrt(np.sign(xpos) / xpos /xpos /xpos) for xpos in XPrimCent], c = 'red', linestyle='dashed', label = r'$-\Omega_K(r)$')
            plt.xlim([-0.5, 0.5])
            plt.ylim([-200,200])
            plt.axvline(x = primary.softening_length, c ='grey', linestyle = 'dashed')
            plt.axvline(x = -primary.softening_length, c ='grey', linestyle = 'dashed')
            plt.legend()
            plt.xlabel(r'$r~[a_0]$')
            plt.title(r'Angular Speed of Minidisk')
            plt.ylabel(r'$\Omega(r)$', rotation = 0)
            plt.savefig('/home/do2364/sailfish/NEW.png', dpi = 300)

            return f

        def Pressure(self):
            x, y = self.Mesh()

            primary, secondary = chkpt['point_masses']
            xprim,yprim        = primary.position_x, primary.position_y
            xsec,ysec          = secondary.position_x, secondary.position_y

            XSecCent  = np.array(x) + xsec
            YSecCent  = np.array(y) + ysec
            XPrimCent = np.array(x) + xprim
            YPrimCent = np.array(y) + yprim

            XPrim, YPrim = np.meshgrid(XPrimCent,YPrimCent)
            XSec, YSec   = np.meshgrid(XSecCent,YSecCent)

            rs1       = primary.softening_length
            rs2       = secondary.softening_length

            rprim2    = XPrim**2 + YPrim**2
            rsec2     = XSec**2  + YSec**2
            Potential =  - 0.5 / (rprim2 + rs1**2) - 0.5 / (rsec2 + rs2**2)
            #cs2       = - Potential / (MACH**2)

            f =  - Potential / 100
            
            return f




        def Vortensity(self):
            x, y   = self.Mesh()
            dVy_dx = np.gradient(self.Vy, axis=1)  # Partial derivative of Vy with respect to x
            dVx_dy = np.gradient(self.Vx, axis=0)  # Partial derivative of Vx with respect to y

            f      = dVy_dx - dVx_dy

            return f #Ignoring 1/Sigma here

        def Pressure(self):
            x, y = self.Mesh()

            primary, secondary = chkpt['point_masses']
            xprim,yprim        = primary.position_x, primary.position_y
            xsec,ysec          = secondary.position_x, secondary.position_y

            XSecCent  = np.array(x) + xsec
            YSecCent  = np.array(y) + ysec
            XPrimCent = np.array(x) + xprim
            YPrimCent = np.array(y) + yprim

            XPrim, YPrim = np.meshgrid(XPrimCent,YPrimCent)
            XSec, YSec   = np.meshgrid(XSecCent,YSecCent)

            rs1       = primary.softening_length
            rs2       = secondary.softening_length

            rprim2    = XPrim**2 + YPrim**2
            rsec2     = XSec**2  + YSec**2
            Potential =  - 0.5 / (rprim2 + rs1**2) - 0.5 / (rsec2 + rs2**2)
            #cs2       = - Potential / (MACH**2)

            f =  - Potential / 100
            
            return f


    class DensityAverages():
        def __init__(self, mesh):
            self.mesh = mesh

        def Mesh(self):
            mesh = self.mesh
            ni, nj = mesh.shape
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            return x,y

        def MeshBins(self,Sigma,chkpt): #pass chkpt for saving
            x, y = self.Mesh()
            X, Y = np.meshgrid(x, y)

            Radial_Bins = np.linspace(0,10,1000)
            Mesh_Bins   = {}
            for r in Radial_Bins:
                Mesh_Bins[r] = []

            xvals = x[:,0]
            yvals = y[0,:]

            FitEccentricity = 0 
            FitOrientation  = 0
            for i in range(0,len(xvals)-1):
                for j in range(0,len(yvals)-1):
                    Radius = np.sqrt(xvals[i]**2 + yvals[j]**2)
                    Angle  = np.arctan2(yvals[j],xvals[i])
                    xrotate, yrotate = Radius * np.cos(Angle - FitOrientation), Radius * np.sin(Angle - FitOrientation) 

                    Gridded_SemimajorAxis = np.sqrt(xrotate**2 +yrotate**2 / (1-FitEccentricity**2))
                    #Rotate (x,y) with orientation.Mesh
                    #Find Gridded Gridded_SemimajorAxis

                    #radius        = np.sqrt(xvals[i]**2 + yvals[j]**2)
                    closest_index = np.abs(Radial_Bins - Gridded_SemimajorAxis).argmin()
                
                    Mesh_Bins[Radial_Bins[closest_index]].append(Sigma[i,j])
                
            f = []
            for r in Radial_Bins:
                f.append(np.mean(Mesh_Bins[r]))

            #plt.figure(figsize=(12,6))
            #plt.plot(Radial_Bins,f, linewidth = 2, label = 'Azimuthally Averaged Surface Density')
            #plt.xlabel(r'Radius $[a_0]$', fontsize = 16)
            #plt.legend()
            #plt.show()

            #print('density',f)
            if chkpt["model_parameters"]["retrograde"]:
                np.save('/Users/davidoneill/Desktop/Averages/AveragedDensity_Retrograde%g_time%g'%(chkpt['model_parameters']['nu'],chkpt["time"]/ 2 / np.pi), f)
            else:
                np.save('/Users/davidoneill/Desktop/Averages/AveragedDensity_Prograde%g_time%g'%(chkpt['model_parameters']['nu'],chkpt["time"]/ 2 / np.pi), f)


        def Axisymmetry(self,Sigma):
            x,y  = self.Mesh()
            X, Y = np.meshgrid(x, y)

            Radial_Bins = np.linspace(0,10,1000)
            Mesh_Bins   = {}
            for r in Radial_Bins:
                Mesh_Bins[r] = []

            xvals = x[:,0]
            yvals = y[0,:]

            for i in range(0,len(xvals)-1):
                for j in range(0,len(yvals)-1):
                    radius        = np.sqrt(xvals[i]**2 + yvals[j]**2)
                    closest_index = np.abs(Radial_Bins - radius).argmin()
                
                    Mesh_Bins[Radial_Bins[closest_index]].append(Sigma[i,j])
                
            f = []
            for r in Radial_Bins:
                f.append(np.std(Mesh_Bins[r]))

            plt.figure(figsize=(text_width,0.5*text_width))
            plt.plot(Radial_Bins,f, linewidth = 2, label = 'Surface Density Standard Deviation')
            plt.xlabel(r'Radius $[a_0]$')
            plt.legend()
            plt.show()

            print('Standard Deviations',f)


    for filename in args.checkpoints:
        fig, ax     = plt.subplots(figsize=[text_width, 0.75*text_width])
        chkpt       = load_checkpoint(filename)
        CurrentTime = chkpt["time"]/ 2 / np.pi
        
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

        elif args.field == 'pressure':
            sigma = fields['sigma'](prim).T
            f     = sigma * Velocities.Pressure()
            #f     = Velocities.Pressure()

        elif args.field == 'vortensity':
            sigma = fields['sigma'](prim).T
            f     = Velocities.Vortensity()/sigma

        elif args.field == 'AverageDensity':
            AD = DensityAverages(mesh)
            Sigma = fields['sigma'](prim).T
            AD.MeshBins(Sigma, chkpt)
            sys.exit()

        elif args.field == 'Axisymmetry':
            AD = DensityAverages(mesh)
            Sigma = fields['sigma'](prim).T
            AD.Axisymmetry(Sigma)
            sys.exit()

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
            colorbar1.ax.tick_params()

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
                
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        primary, secondary = chkpt['point_masses']

        ax.scatter(primary.position_x, primary.position_y, marker = '+', s = 40, c = 'white', label = 'Point Mass')
        #ax.axhline(y=0, linestyle='dashed', c = 'gray', label = 'x cut')
        ax.scatter(secondary.position_x, secondary.position_y, marker = '+', s = 40, c = 'white')
        #ax.legend()
        #ax.text(
        #        0.8, 0.95,  # Relative coordinates (x=5% from left, y=95% from bottom)
        #        r'$t = %g$'%(int(chkpt["time"]/ 2 / np.pi)),
        #        fontsize=24,

        from matplotlib.patches import Circle
        primarycenter   = (primary.position_x, primary.position_y)
        secondarycenter = (secondary.position_x, secondary.position_y)
        radius          = primary.sink_radius         # Radius of the circle

        # Create a circle
        primarysink   = Circle(primarycenter, radius, color='white', fill=True, alpha=0.7)
        secondarysink = Circle(secondarycenter, radius, color='white', fill=True, alpha=0.7)

        # Add the circle to the axes
        ax.add_patch(primarysink)
        ax.add_patch(secondarysink)

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
            #ax.set_xlim(0.3, 0.6)
            #ax.set_ylim(-0.15, 0.15)
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
            print('Saved at', pngname)

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
    import numpy as np
    import os
    import sailfish.physics.cooling as cooling
    from sailfish.physics.cooling import EffectiveTemperature, cgs, ShakuraSunyaevDisk, PlanckSpectrum 
    from sailfish.physics.kepler import OrbitalState, PointMass
    from matplotlib.patches import Circle
    transparent_black = mcolors.LinearSegmentedColormap.from_list("transparent_black", [(0, (1, 1, 1, 0)), (1, (0, 0, 0, 1))])

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
        default="Sigma",
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
        "--SED",
        default=False,
        action="store_true",
        help="plot spectrum",
    )
    parser.add_argument(
        "--axisymmetry",
        "-asym",
        default=False,
        action="store_true",
        help="plot Mean Absolute Deviation of surface density",
    )
    parser.add_argument(
        "--MinidiskProfile",
        "-mp",
        default=False,
        action="store_true",
        help="plot minidisk velocity profile",
    )
    parser.add_argument(
        "--vmap",
        default=False,
        action="store_true",
        help="plot gas velocities",
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
        "--print_model_parameters",
        "-params",
        action="store_true",
        help="plot the parameters used for making this checkpoint",
    )
    parser.add_argument(
        "--plot_sink",
        action="store_true",
        help="plot the sink properties",
    )
    parser.add_argument(
        "--remap",
        type=bool,
        default=False,
        help="Whether or not to rescale disk properties to target accretion rate",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="colormap name",
    )

    args = parser.parse_args()
    for filename in args.checkpoints:
        fig, ax       = plt.subplots(figsize=[text_width, text_width])
        chkpt         = load_checkpoint(filename, require_solver="cbdgam_2d")
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0)
        
        # ======= Useful checkpoint data ========
        CurrentTime    = chkpt["time"]/ 2 / np.pi
        mesh           = chkpt["mesh"]
        prim           = chkpt["solution"]
        SS73           = chkpt["SS73"]
        gamma          = chkpt['model_parameters']['gamma_law_index']
        Floor_Depth    = chkpt['model_parameters']['OpticalDepthFloor']
        ni, nj         = mesh.shape
        x              = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])
        y              = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])
        X, Y           = np.meshgrid(x, y, indexing="xy")
        
        # ======== Calculate desired field ========
        Mdrop           = (SS73.Mdrop if args.remap else 1.0)
        Sigma           = fields["sigma"](prim).T * Mdrop**(3./5.)
        Pressure        = fields["pre"](prim).T   * Mdrop
        Vx, Vy          = chkpt['solution'][:, :, 1].T, chkpt['solution'][:, :, 2].T
        optical_depth   = (Sigma * SS73.kappa_code)
        mask_values     = (optical_depth > Floor_Depth)
        Midplane_T      = ((Pressure / Sigma) * (SS73.mp_code / SS73.kb_code))
        Teff            = EffectiveTemperature(optical_depth, Midplane_T) 
        r_g             = cgs['G'] * chkpt['model_parameters']['central_mass_msun'] * cgs['msun'] / cgs['c'] / cgs['c']
        length_scale_pc = r_g * chkpt['model_parameters']['init_separation_rg'] / cgs['pc']

        def SaveBinStats(dictname, keys, args):
            for (key, arg) in zip(keys, args):
                dictname[key].append(arg)
        
        if args.field == 't':
            f             = Teff
            title         = 'Effective Temperature [K] '
            cmap          = 'inferno'
            savename      = 'TemperatureMap'
            if args.log:
                ColourbarLabel = r'$\log_{10}T_\mathrm{eff}$'
            else:
                ColourbarLabel = r'$T_\mathrm{eff}$'

        elif args.field == 'pressure':
            f             = Pressure
            title         = 'Pressure '
            cmap          = 'inferno'
            savename      = 'PressureMap'
            if args.log:
                ColourbarLabel = r'$\log_{10}P$'
            else:
                ColourbarLabel = r'$P$'

        elif args.field == 't4':
            f             = (Teff)**4
            title         = 'Emitted Flux [$\mathrm{K^4}$] '
            cmap          = 'inferno'
            savename      = 'FluxMap'
            if args.log:
                ColourbarLabel = r'$\log_{10}T^4$'
            else:
                ColourbarLabel = r'$T^4$'

        elif args.field == 'tau':
            f             = optical_depth
            title         = 'Optical Depth'
            savename      = 'TauMap'
            cmap          = 'cividis'
            if args.log:
                ColourbarLabel = r'$\log_{10}\tau$'
            else:
                ColourbarLabel = r'$\tau$'

        elif args.field == 'viscosity':
            primary, secondary = chkpt['point_masses']
            xprim, yprim       = primary.position_x  , primary.position_y
            xsec, ysec         = secondary.position_x, secondary.position_y
            R_1, R_2           = np.sqrt((X-xprim)**2 + (Y-yprim)**2), np.sqrt((X-xsec)**2  + (Y-ysec)**2)
            omega              = np.sqrt(primary.mass / (R_1**3 + 1e-12) + secondary.mass / (R_2**3 + 1e-12))
            cs                 = (gamma * Pressure / Sigma)**0.5
            H                  = cs / omega
            nu                 = chkpt['model_parameters']['alpha'] * cs * H
            f                  = nu
            title              = 'Viscosity'
            savename           = 'ViscosityMap'
            cmap               = 'cividis'
            if args.log:
                ColourbarLabel = r'$\log_{10}\nu$'
            else:
                ColourbarLabel = r'$\nu$'

        elif args.field == 'speed':
            speed   = np.sqrt(Vx**2 + Vy**2)
            print('Max speed is', np.max(speed))
            f       = speed
            title   = 'Speed Map'
            savename= 'SpeedMap'
            cmap    = 'inferno'
            if args.log:
                ColourbarLabel = r'$\log_{10}|\mathbf{v}|$'
            else:
                ColourbarLabel = r'$|\mathbf{v}|$'

        elif args.field == 'dt':
            cs      = np.sqrt(gamma * Pressure / Sigma)
            speed_x = np.maximum(np.abs(Vx - cs), np.abs(Vx + cs))
            speed_y = np.maximum(np.abs(Vy - cs), np.abs(Vy + cs))
            max_sp  = np.maximum(speed_x, speed_y)
            dx      = mesh.min_spacing(CurrentTime)
            cfl     = chkpt['cfl_number']
            f       = (dx * cfl / max_sp)
            cmap    = 'magma_r'
            if args.log:
                ColourbarLabel = r'$\log_{10}dt$'
            else:
                ColourbarLabel = r'$dt$'

            title     = 'Time Step Map'
            savename  = 'Timestepping'
            dt_global = dx * cfl / np.max(max_sp)
            print(f"Global timestep from max speed: {dt_global:.5e}")

        elif args.field == 'mach':
            cs     = (gamma * Pressure / Sigma)**0.5
            ni, nj = mesh.shape
            cmap   = 'magma'
            
            # ===== Calculate orbital velocity ========
            primary, secondary = chkpt['point_masses']
            xprim, yprim       = primary.position_x  , primary.position_y
            xsec, ysec         = secondary.position_x, secondary.position_y
            speed              = np.sqrt(Vx**2 + Vy**2)
            f                  = (speed / cs)
            title              = 'Mach Number'
            savename           = 'MachMap'
            MachNumber_a       = chkpt["model_parameters"]["mach_number_a"] * Mdrop**(-1./5.)
            if args.log:
                ColourbarLabel = r'$\log_{10}\mathcal{M}$'
            else:
                ColourbarLabel = r'$\mathcal{M}$'

            # ===== Plot midplane cuts ========
            plt.figure(figsize = (column_width,3*column_width/4))
            plt.title('Mach Number Profile')
            plt.plot(np.linspace(mesh.x0, mesh.x1, mesh.shape[0]), f[mesh.shape[1]//2,:], label = 'horizontal cut', c = 'tab:red', linewidth = 2)
            plt.plot(np.linspace(mesh.x0, mesh.x1, mesh.shape[0]), f[:,mesh.shape[0]//2], label = 'vertical cut'  , c = 'tab:blue', linewidth = 2)
            plt.plot(np.linspace(mesh.x0, mesh.x1, mesh.shape[0]), [SS73.mach_profile(np.abs(r)) for r in np.linspace(mesh.x0, mesh.x1, mesh.shape[0])], label = 'SS73 Mach Profile', linestyle = 'dashed', c = 'black')
            plt.ylim([0 ,2*MachNumber_a])
            plt.legend()
            plt.savefig(args.Outputs + "/MidplaneMach.png", dpi = 300)

        elif args.field == 'eccentricity':
            r        = np.sqrt(X**2+Y**2)
            v_dot_v  = Vx * Vx + Vy * Vy
            v_dot_r  = Vx * X  + Vy * Y
            ex       = (v_dot_v * X - v_dot_r * Vx) / 1 - X / r
            ey       = (v_dot_v * Y - v_dot_r * Vy) / 1 - Y / r
            f        = np.sqrt(ex**2 + ey**2)
            title    = 'Disk Eccentricity'
            savename = 'DiskEccentricityMap'
            cmap     = 'plasma'
            if args.log:
                ColourbarLabel = r'$\log_{10}e$'
            else:
                ColourbarLabel = r'$e$'
            
        elif args.field == 'Sigma':
            f        = Sigma
            title    = 'Surface Density'
            savename = 'DensityMap'
            cmap     = 'magma'
            if args.log:
                ColourbarLabel = r'$\log_{10}\Sigma$'
            else:
                ColourbarLabel = r'$\Sigma$'

        if args.log:
            f = np.log10(f)

        
        # ========== Plotting ============
        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        cm     = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=cmap,
            extent=extent,
        )
        cbar = fig.colorbar(cm, ax=ax, shrink=0.805, aspect=20, pad=0.05)
        cbar.ax.set_title(ColourbarLabel, pad=6)          # puts text above the bar
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        ax.set_aspect("equal")
        ax.set_title(title + r' at time $t = $ %g $\mathrm{[2\pi\Omega_0^{-1}]}$'%(np.round(chkpt["time"]/2/np.pi,3)))
    
        if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)
            mask_x , mask_y  = (x >= -args.radius) & (x <= args.radius), (y >= -args.radius) & (y <= args.radius)
            x_bound, y_bound = x[mask_x], y[mask_y]
        else:
            mask_x , mask_y  = np.ones_like(x, dtype=bool), np.ones_like(y, dtype=bool)
            x_bound, y_bound = x[mask_x], y[mask_y]

        if args.vmap:
            VectorN  = 40
            Vx_bound = Vx[mask_y][:, mask_x]
            Vy_bound = Vy[mask_y][:, mask_x]
            stride_x = max(1, len(x_bound) // VectorN) # downsample safely
            stride_y = max(1, len(y_bound) // VectorN) # downsample safely
            Xv, Yv   = np.meshgrid(x_bound[::stride_x], y_bound[::stride_y])

            plt.quiver(
                Xv, Yv,
                Vx_bound[::stride_y, ::stride_x], Vy_bound[::stride_y, ::stride_x],
                width=0.002, angles='xy', scale_units='xy', scale=20,
                color='darkgrey', headwidth=4
            )

        if args.plot_sink:
            primary, secondary = chkpt['point_masses']
            Primary            = PointMass(primary.mass  , primary.position_x  , primary.position_y  , primary.velocity_x  , primary.velocity_y)
            Secondary          = PointMass(secondary.mass, secondary.position_x, secondary.position_y, secondary.velocity_x, secondary.velocity_y)

            ax.scatter(primary.position_x  , primary.position_y  , marker = '+', s = 40, c = 'white', label = 'Point Masses')
            ax.scatter(secondary.position_x, secondary.position_y, marker = '+', s = 40, c = 'white')
            primarycenter   = (primary.position_x, primary.position_y)
            secondarycenter = (secondary.position_x, secondary.position_y)
            primarysink     = Circle(primarycenter  , primary.sink_radius  , color='grey', fill=True, alpha=0.8)        # Radius of the circle
            secondarysink   = Circle(secondarycenter, secondary.sink_radius, color='grey', fill=True, alpha=0.8)       # Radius of the circle
            ax.add_patch(primarysink)
            ax.add_patch(secondarysink)

            def Position(t, a, e, q):
                a1 = a * q / (1 + q)
                a2 = a / (1 + q)
                return (
                    (a1 * np.cos(t) - a1 * e, a1 * np.sqrt(1 - e**2) * np.sin(t)), 
                    (a2 * np.cos(t) - a2 * e, a2 * np.sqrt(1 - e**2) * np.sin(t))
                    )
            
            eccentr      = chkpt['timeseries'][-1][ 2] 
            semimaj      = chkpt['timeseries'][-1][ 1] 
            mass_ratio   = chkpt['model_parameters']['mass_ratio']
            Orbital_Path = np.array([Position(t, semimaj, eccentr, mass_ratio) for t in np.linspace(0,2*np.pi,1000)])
            plt.plot(Orbital_Path[:,0,0], Orbital_Path[:,0,1], linestyle = 'dashed', c = 'gray')
            plt.plot(Orbital_Path[:,1,0], Orbital_Path[:,1,1], linestyle = 'dashed', c = 'gray')


        if args.Outputs is None:
            plt.show()
        elif args.Outputs == ".":
            pngname = os.path.join(args.Outputs, f"{savename}-{int(CurrentTime * 100):05d}.png")
            fig.savefig(pngname, dpi=400, bbox_inches='tight')
        else:
            pngname = args.Outputs + savename + f"-{int(CurrentTime * 100):05d}.png"
            fig.savefig(pngname, dpi=400, bbox_inches='tight')


    if args.SED:
        E_low, E_high            = 1e-1 * cgs['ev'], 5e6 * cgs['ev']
        E_array                  = np.logspace(np.log10(E_low/(1000*cgs['ev'])), np.log10(E_high/(1000*cgs['ev'])), 100)
        freq_low, freq_high      = E_low / cgs['h'], E_high / cgs['h']
        freq_space               = np.logspace(np.log10(freq_low), np.log10(freq_high), len(E_array))
        Masked_Cell_Temperatures = Teff * (mask_values)
        Cell_Spectra             = np.array([PlanckSpectrum(freq, Masked_Cell_Temperatures) for freq in freq_space]) # Note: This can be large if remap is False!
        Spectrum                 = np.sum(Cell_Spectra, axis=(1,2)) * (length_scale_pc*cgs['pc']*mesh.dx)**2
        integral                 = np.trapz(Spectrum, x=freq_space)

        E_Xray_low   , E_Xray_high    = cooling.E_Xray_low/(1000*cgs['ev'])   , cooling.E_Xray_high/(1000*cgs['ev'])
        E_UV_low     , E_UV_high      = cooling.E_UV_low/(1000*cgs['ev'])     , cooling.E_UV_high/(1000*cgs['ev'])
        E_optical_low, E_optical_high = cooling.E_optical_low/(1000*cgs['ev']), cooling.E_optical_high/(1000*cgs['ev'])
        E_infared_low, E_infared_high = cooling.E_infared_low/(1000*cgs['ev']), cooling.E_infared_high/(1000*cgs['ev'])


        plt.figure(figsize=(column_width, 0.75*column_width))
        plt.xscale('log')
        plt.yscale('log')
       
        # ================ Add band filters ================
        plt.axvspan(E_Xray_low   , E_Xray_high   , color='gray', alpha=0.15, label='X-ray band') 
        plt.axvspan(E_UV_low     , E_UV_high     , color='gray', alpha=0.15, label='UV band') 
        plt.axvspan(E_optical_low, E_optical_high, color='gray', alpha=0.15, label='Optical band') 
        plt.axvspan(E_infared_low, E_infared_high, color='gray', alpha=0.15, label='Infrared band') 
        ax        = plt.gca()
        x_xray    = np.sqrt(E_Xray_low * E_Xray_high)
        x_uv      = np.sqrt(E_UV_low * E_UV_high)
        x_optical = np.sqrt(E_optical_low * E_optical_high)
        x_infared = np.sqrt(E_infared_low * E_infared_high)
        ax.text(x_xray   , 0.5, 'X-ray'   , color='gray', fontweight='bold', ha='center', va='center', rotation=90, alpha=1.0, transform=ax.get_xaxis_transform(), bbox=dict(facecolor='gray', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(x_uv     , 0.5, 'UV'      , color='gray', fontweight='bold', ha='center', va='center', rotation=90, alpha=1.0, transform=ax.get_xaxis_transform(), bbox=dict(facecolor='gray', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(x_optical, 0.5, 'Optical' , color='gray', fontweight='bold', ha='center', va='center', rotation=90, alpha=1.0, transform=ax.get_xaxis_transform(), bbox=dict(facecolor='gray', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(x_infared, 0.5, 'Infrared', color='gray', fontweight='bold', ha='center', va='center', rotation=90, alpha=1.0, transform=ax.get_xaxis_transform(), bbox=dict(facecolor='gray', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.3'))


        # ========== Add "Integrated Spectrum" text ==========
        plt.axhline(y = integral, linestyle='dashed', c = 'black')
        xmin, xmax = ax.get_xlim()
        xmid       = np.sqrt(xmin * xmax)
        ax.text(x=xmax*5, y=integral, s=r'$\int d\nu \nu L_\nu$', color='black', fontweight='bold', ha='center', va='center', rotation=0, alpha=1.0, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))

        # ================ Plot Spectrum ================
        plt.plot(E_array, freq_space*Spectrum, c = 'black', label = 'SED')

        # ============= Labels, Titles Limits and Saves =============
        plt.xlabel(r'$E = h\nu~[\mathrm{keV}]$')
        plt.ylabel(r'$\nu L_\nu~[\mathrm{erg~s^{-1}}]$')
        plt.title('Spectral Energy Distribution at t = %g'%(chkpt["time"]/ 2 / np.pi))
        plt.ylim([1e40, 30 * integral])
        plt.xlim([E_infared_low, E_Xray_high])
        Savename = os.path.join(args.Outputs, f"SED_{chkpt['time'] / 2 / np.pi:.2f}.png")
        plt.savefig(Savename, dpi=400, bbox_inches='tight')

    if args.MinidiskProfile:
        RMinidisk          = 0.3
        primary, secondary = chkpt['point_masses']
        SinkRadius         = (primary.sink_radius     , secondary.sink_radius)
        SoftRadius         = (primary.softening_length, secondary.softening_length)
        Nbins              = int((RMinidisk/(mesh.x1-mesh.x0)) * ni)
        RadialBins         = np.linspace(0,RMinidisk,Nbins)
        GMu_primary        = primary.mass     # GM = 1.0 normalised in code units
        GMu_secondary      = secondary.mass
        r_primary          = [X  - primary.position_x  , Y  - primary.position_y  ]
        r_secondary        = [X  - secondary.position_x, Y  - secondary.position_y]
        V_primary          = [Vx - primary.velocity_x  , Vy - primary.velocity_y  ]
        V_secondary        = [Vx - secondary.velocity_x, Vy - secondary.velocity_y]

        primary_speed , secondary_speed  = np.sqrt(V_primary[0]**2 + V_primary[1]**2), np.sqrt(V_secondary[0]**2 + V_secondary[1]**2)
        primary_radius, secondary_radius = np.sqrt(r_primary[0]**2 + r_primary[1]**2), np.sqrt(r_secondary[0]**2 + r_secondary[1]**2)

        v_dot_v_primary   = primary_speed**2
        v_dot_r_primary   = V_primary[0] * r_primary[0] + V_primary[1] * r_primary[1]
        ex_primary        = (v_dot_v_primary * r_primary[0] - v_dot_r_primary * V_primary[0]) / GMu_primary - r_primary[0] / primary_radius
        ey_primary        = (v_dot_v_primary * r_primary[1] - v_dot_r_primary * V_primary[1]) / GMu_primary - r_primary[1] / primary_radius
        omega_primary     = np.arctan2(ey_primary, ex_primary)
        e_primary         = np.sqrt(ex_primary**2 + ey_primary**2)
        v_dot_v_secondary = secondary_speed**2
        v_dot_r_secondary = V_secondary[0] * r_secondary[0] + V_secondary[1] * r_secondary[1]
        ex_secondary      = (v_dot_v_secondary * r_secondary[0] - v_dot_r_secondary * V_secondary[0]) / GMu_secondary - r_secondary[0] / secondary_radius
        ey_secondary      = (v_dot_v_secondary * r_secondary[1] - v_dot_r_secondary * V_secondary[1]) / GMu_secondary - r_secondary[1] / secondary_radius
        omega_secondary   = np.arctan2(ey_secondary, ex_secondary)
        e_secondary       = np.sqrt(ex_secondary**2 + ey_secondary**2)
        PrimaryMiniDisk   = {'MeanV': [], 'MinV': [], 'MaxV': [], 'MeanE': [], 'MinE': [], 'MaxE': [], 'MeanD': [], 'MinD': [], 'MaxD': [], 'MeanW': [], 'MinW': [], 'MaxW': []}
        SecondaryMiniDisk = {'MeanV': [], 'MinV': [], 'MaxV': [], 'MeanE': [], 'MinE': [], 'MaxE': [], 'MeanD': [], 'MinD': [], 'MaxD': [], 'MeanW': [], 'MinW': [], 'MaxW': []}

        for i in range(len(RadialBins)-1):
            primary_mask   = (RadialBins[i] < primary_radius) & (primary_radius < RadialBins[i+1])
            secondary_mask = (RadialBins[i] < secondary_radius) & (secondary_radius < RadialBins[i+1])
            primary_N      = np.sum(primary_mask)
            secondary_N    = np.sum(secondary_mask)

            primary_speed_mask     = primary_speed[primary_mask]
            secondary_speed_mask   = secondary_speed[secondary_mask]
            primary_e_mask         = e_primary[primary_mask]
            secondary_e_mask       = e_secondary[secondary_mask]
            primary_density_mask   = Sigma[primary_mask]
            secondary_density_mask = Sigma[secondary_mask]
            primary_phase_mask     = omega_primary[primary_mask]
            secondary_phase_mask   = omega_secondary[secondary_mask]

            # plt.figure()
            # plt.imshow(primary_density_mask, origin='lower', cmap='magma')
            # plt.colorbar(label='Sigma')
            # plt.title(f'Primary density, bin {i}')
            # plt.savefig(os.path.join(args.Outputs, f"PrimaryMinidiskDensityBin{i}.png"), dpi=300, bbox_inches='tight')


            SaveBinStats(
                PrimaryMiniDisk,
                ['MeanV', 'MinV', 'MaxV', 'MeanE', 'MinE', 'MaxE', 'MeanD', 'MinD', 'MaxD', 'MeanW', 'MinW', 'MaxW'], 
                [np.mean(primary_speed_mask)  , np.min(primary_speed_mask)  , np.max(primary_speed_mask),
                 np.mean(primary_e_mask)      , np.min(primary_e_mask)      , np.max(primary_e_mask),
                 np.mean(primary_density_mask), np.min(primary_density_mask), np.max(primary_density_mask),
                 np.mean(primary_phase_mask)  , np.min(primary_phase_mask)  , np.max(primary_phase_mask)]
            )
            SaveBinStats(
                SecondaryMiniDisk,
                ['MeanV', 'MinV', 'MaxV', 'MeanE', 'MinE', 'MaxE', 'MeanD', 'MinD', 'MaxD', 'MeanW', 'MinW', 'MaxW'], 
                [np.mean(secondary_speed_mask)  , np.min(secondary_speed_mask)  , np.max(secondary_speed_mask),
                 np.mean(secondary_e_mask)      , np.min(secondary_e_mask)      , np.max(secondary_e_mask),
                 np.mean(secondary_density_mask), np.min(secondary_density_mask), np.max(secondary_density_mask),
                 np.mean(secondary_phase_mask)  , np.min(secondary_phase_mask)  , np.max(secondary_phase_mask)]
            )

        vlim            = 1.1 * np.nanmax([np.nanmax(PrimaryMiniDisk['MaxV']), np.nanmax(SecondaryMiniDisk['MaxV'])])
        dlim            = 3   * np.nanmax([np.max(PrimaryMiniDisk['MaxD'])   , np.max(SecondaryMiniDisk['MaxD'])])
        PrimaryRadius   = RadialBins[1:] / SinkRadius[0]
        SecondaryRadius = RadialBins[1:] / SinkRadius[1]

        fig = plt.figure(figsize=(1.0 * text_width, 1.0 * text_width))
        gs  = fig.add_gridspec(4, 2, height_ratios=[1, 0.5, 0.6, 0.6], hspace=0.1, wspace=0.1)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax_ = fig.add_subplot(gs[1, :])
        ax2 = fig.add_subplot(gs[2, :]) 
        ax3 = fig.add_subplot(gs[3, :]) 

        ax0.set_title('Primary Minidisk')
        ax0.set_xlim([0, PrimaryRadius[-1]])
        ax0.set_ylim([0, vlim])
        ax0.plot(PrimaryRadius, PrimaryMiniDisk['MeanV']  , label = 'Primary Mean'  , c = 'red')
        ax0.fill_between(PrimaryRadius, PrimaryMiniDisk['MinV'], PrimaryMiniDisk['MaxV'], color='red', alpha=0.3, label = 'Primary Range')
        ax0.plot(PrimaryRadius, [np.sqrt(primary.mass/r) for r in RadialBins[1:]], linestyle='dashed', c = 'black', label = 'Keplerian Profile')
        ax0.set_xlabel(r'Distance $[r_\mathrm{sink}]$')#; ax0.set_xscale('log')
        ax0.set_ylabel(r'Velocity $[a_0\Omega_0]$')#; ax0.set_yscale('log')
        ax0.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black', label = 'Softening Radius')
        

        ax1.set_title('Secondary Minidisk')
        ax1.set_xlim([0, SecondaryRadius[-1]])
        ax1.set_ylim([0, vlim])
        ax1.plot(SecondaryRadius, SecondaryMiniDisk['MeanV'], label='Secondary Mean', c = 'blue'  )
        ax1.fill_between(SecondaryRadius, SecondaryMiniDisk['MinV'], SecondaryMiniDisk['MaxV'], color='blue', alpha=0.3, label = 'Secondary Range')
        ax1.plot(SecondaryRadius, [np.sqrt(secondary.mass/r) for r in RadialBins[1:]], linestyle='dashed', c = 'black', label = 'Keplerian Profile')
        ax1.tick_params(axis='y', colors='white')#; ax1.set_yscale('log')
        ax1.set_xlabel(r'Distance $[r_\mathrm{sink}]$')#; ax1.set_xscale('log')
        ax1.axvline(x = SoftRadius[1]/SinkRadius[1], linestyle='dotted', c = 'black', label = 'Softening Radius')
        #ax1.legend()

        ax_.axis('off')

        # ax_.plot(PrimaryRadius  , PrimaryMiniDisk['MeanW'], color='red' , alpha=0.8, label = r'Primary $\omega$')
        # ax_.plot(SecondaryRadius, SecondaryMiniDisk['MeanW'], color='blue', alpha=0.8, label = r'Secondary $\omega$')
        # ax_.set_xlim([0, SecondaryRadius[-1]])
        # ax_.set_ylim([-np.pi, np.pi])
        # ax_.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black', alpha=0.6)
        # ax_.set_xticks([])
        # ax_.set_ylabel(r'$\omega$ [rad]')
        # ax_.legend(loc='upper right')

        ax2.plot(PrimaryRadius  , PrimaryMiniDisk['MeanE']  , color='red' , alpha=0.8, label = 'Primary Mean')
        ax2.plot(SecondaryRadius, SecondaryMiniDisk['MeanE'], color='blue', alpha=0.8, label = 'Secondary Mean')
        ax2.fill_between(PrimaryRadius  , PrimaryMiniDisk['MinE']  , PrimaryMiniDisk['MaxE']  , color='red', alpha=0.1, label = 'Min/Max')
        ax2.fill_between(SecondaryRadius, SecondaryMiniDisk['MinE'], SecondaryMiniDisk['MaxE'], color='blue', alpha=0.1, label = 'Min/Max')
        ax2.set_xlim([0, SecondaryRadius[-1]])
        ax2.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black', label = 'Softening Radius')
        ax2.set_xticks([])
        ax2.set_ylabel(r'Eccentricity')

        SS73_coeff   = PrimaryMiniDisk['MeanD'][Nbins//2]/PrimaryRadius[Nbins//2]**(-3/5)
        SS73_profile = [SS73_coeff * r**(-3./5.) for r in PrimaryRadius]
        ax3.plot(PrimaryRadius  , PrimaryMiniDisk['MeanD']  , color='red' , alpha=0.8)
        ax3.plot(SecondaryRadius, SecondaryMiniDisk['MeanD'], color='blue', alpha=0.8)
        ax3.plot(PrimaryRadius  , SS73_profile              , color='peru', linestyle='dashed', label = r'$r^{-3/5}$')
        ax3.fill_between(PrimaryRadius  , PrimaryMiniDisk['MinD']  , PrimaryMiniDisk['MaxD']  , color='red', alpha=0.1)
        ax3.fill_between(SecondaryRadius, SecondaryMiniDisk['MinD'], SecondaryMiniDisk['MaxD'], color='blue', alpha=0.1)
        ax3.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black')
        ax3.set_xlim([0, SecondaryRadius[-1]]); ax3.set_xlabel(r'Distance $[r_\mathrm{sink}]$')
        ax3.set_ylim([dlim*1e-4,dlim]); ax3.set_ylabel(r'$\langle\Sigma\rangle_\phi$')
        ax3.set_yscale('log')
        ax3.legend(loc='upper right')

        # Include phase!
        
        #plt.suptitle(r'\textbf{Minidisk Profiles with} $r_\mathrm{sink}$ = %g'%(SinkRadius[0]), y=0.95, fontweight="heavy")
        handles = []
        labels  = []

        for ax in [ax0, ax1]:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:   # avoid duplicates
                    handles.append(hi)
                    labels.append(li)
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
        plt.savefig(f"MinidiskProfiles{chkpt['time'] / 2 / np.pi:.2f}.png", dpi=400, bbox_inches='tight')

    if args.axisymmetry:
        primary, secondary = chkpt['point_masses']
        SinkRadius         = (primary.sink_radius     , secondary.sink_radius)
        SoftRadius         = (primary.softening_length, secondary.softening_length)
        Nbins              = int(ni/2)
        RadialBins         = np.linspace(0,mesh.x1,Nbins)
        GMu_primary        = primary.mass     # GM = 1.0 normalised in code units
        GMu_secondary      = secondary.mass
        
        radius             = np.sqrt( X**2 +  Y**2)
        speed              = np.sqrt(Vx**2 + Vy**2)
        v_dot_v            = speed**2
        v_dot_r            = Vx*X + Vy*Y
        ex                 = (v_dot_v * X - v_dot_r * Vx) / (GMu_primary + GMu_secondary) - X / radius
        ey                 = (v_dot_v * Y - v_dot_r * Vy) / (GMu_primary + GMu_secondary) - Y / radius
        omega              = np.arctan2(ey, ex)
        e                  = np.sqrt(ex**2 + ey**2)
        DiskStats          = {'MeanV': [], 'MinV': [], 'MaxV': [], 'MeanE': [], 'MinE': [], 'MaxE': [], 'MeanD': [], 'MinD': [], 'MaxD': [], 'MeanW': [], 'MinW': [], 'MaxW': []}
        
        

        for i in range(len(RadialBins)-1):
            mask   = (RadialBins[i] < radius) & (radius < RadialBins[i+1])
            N      = np.sum(mask)

            speed_mask     = speed[mask]
            e_mask         = e[mask]
            density_mask   = Sigma[mask]
            phase_mask     = omega[mask]

            SaveBinStats(
                DiskStats,
                ['MeanV', 'MinV', 'MaxV', 'MeanE', 'MinE', 'MaxE', 'MeanD', 'MinD', 'MaxD', 'MeanW', 'MinW', 'MaxW'], 
                [np.mean(speed_mask)  , np.min(speed_mask)  , np.max(speed_mask),
                 np.mean(e_mask)      , np.min(e_mask)      , np.max(e_mask),
                 np.mean(density_mask), np.min(density_mask), np.max(density_mask),
                 np.mean(phase_mask)  , np.min(phase_mask)  , np.max(phase_mask)]
            )


        vlim     = min([1.1 * np.nanmax(DiskStats['MaxV']), 5])
        dlim     = min([3   * np.nanmax(DiskStats['MaxD']), 1e-3])
        Radius   = RadialBins[1:] / SinkRadius[0]

        fig = plt.figure(figsize=(1.0 * text_width, 1.0 * text_width))
        gs  = fig.add_gridspec(4, 2, height_ratios=[1, 0.5, 0.6, 0.6], hspace=0.1, wspace=0.1)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax_ = fig.add_subplot(gs[1, :])
        ax2 = fig.add_subplot(gs[2, :]) 
        ax3 = fig.add_subplot(gs[3, :]) 

        ax0.set_title('Disk')
        ax0.set_xlim([0, Radius[-1]])
        ax0.set_ylim([0, vlim])
        ax0.plot(Radius, DiskStats['MeanV']  , label = 'Mean Velocity'  , c = 'red')
        ax0.fill_between(Radius, DiskStats['MinV'], DiskStats['MaxV'], color='red', alpha=0.3, label = 'Range')
        ax0.plot(Radius, [np.sqrt(1/r) for r in RadialBins[1:]], linestyle='dashed', c = 'black', label = 'Keplerian Profile')
        ax0.set_xlabel(r'Distance $[r_\mathrm{sink}]$')#; ax0.set_xscale('log')
        ax0.set_ylabel(r'Velocity $[a_0\Omega_0]$')#; ax0.set_yscale('log')
        ax0.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black', label = 'Softening Radius')
        
        ax_.axis('off')

        ax2.plot(Radius  , DiskStats['MeanE']  , color='red' , alpha=0.8, label = 'Primary Mean')
        ax2.fill_between(PrimaryRadius  , PrimaryMiniDisk['MinE']  , PrimaryMiniDisk['MaxE']  , color='red', alpha=0.1, label = 'Min/Max')        
        ax2.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black', label = 'Softening Radius')
        ax2.set_xticks([])
        ax2.set_ylabel(r'Eccentricity')

        SS73_coeff   = DiskStats['MeanD'][Nbins//2]/Radius[Nbins//2]**(-3/5)
        SS73_profile = [SS73_coeff * r**(-3./5.) for r in Radius]
        ax3.plot(Radius  , DiskStats['MeanD']  , color='red' , alpha=0.8)
        ax3.plot(Radius  , SS73_profile              , color='peru', linestyle='dashed', label = r'$r^{-3/5}$')
        ax3.fill_between(Radius  , DiskStats['MinD']  , DiskStats['MaxD']  , color='red', alpha=0.1)
        ax3.axvline(x = SoftRadius[0]/SinkRadius[0], linestyle='dotted', c = 'black')
        ax3.set_xlim([0, Radius[-1]]); ax3.set_xlabel(r'Distance $[r_\mathrm{sink}]$')
        ax3.set_ylim([dlim*1e-4,dlim]); ax3.set_ylabel(r'$\langle\Sigma\rangle_\phi$')
        ax3.set_yscale('log')
        ax3.legend(loc='upper right')

        
        #plt.suptitle(r'\textbf{Minidisk Profiles with} $r_\mathrm{sink}$ = %g'%(SinkRadius[0]), y=0.95, fontweight="heavy")
        handles = []
        labels  = []

        for ax in [ax0, ax1]:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:   # avoid duplicates
                    handles.append(hi)
                    labels.append(li)
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
        plt.savefig(f"Axisymmetry_{chkpt['time'] / 2 / np.pi:.2f}.png", dpi=400, bbox_inches='tight')




    if args.print_model_parameters:
        print('Iteration Number.........',chkpt['iteration'])
        print('Timestep_dt..............',chkpt['timestep_dt'])
        print('cfl_number...............',chkpt['cfl_number'])
        print('Solver options...........',chkpt['solver_options'])
        print('Event states.............',chkpt['event_states'])

        print('----------------Driver--------------------')
        print(chkpt['driver'])
        print('-------------Model Parameters-------------')
        print(chkpt["model_parameters"])
        print('-------------Solver Parameters-------------')
        print(chkpt["SS73"])
        print('---------------Point Masses---------------')
        print(chkpt["point_masses"])
        print('-------------Timestep dt-------------------')
        print(chkpt['timestep_dt'])






if __name__ == "__main__":
    for arg in sys.argv:
        if arg.endswith(".pk"):
            chkpt = load_checkpoint(arg)
            
            if chkpt["solver"] == "srhd_1d":
                print(f"plotting for srhd_1d solver at time {chkpt['time']/2/3.14159:.2f}")
                exit(main_srhd_1d())
            if chkpt["solver"] == "srhd_2d":
                print(f"plotting for srhd_2d solver at time {chkpt['time']/2/3.14159:.2f}")
                exit(main_srhd_2d())
            if chkpt["solver"] == "cbdiso_2d":
                print(f"plotting for cbdiso_2d solver at time {chkpt['time']/2/3.14159:.2f}")
                exit(main_cbdiso_2d())
            if chkpt["solver"] == "cbdisodg_2d":
                print(f"plotting for cbdisodg_2d solver at time {chkpt['time']/2/3.14159:.2f}")
                exit(main_cbdisodg_2d())
            if chkpt["solver"] == "cbdgam_2d":
                print(f"plotting for cbdgam_2d solver at time {chkpt['time']/2/3.14159:.2f}")
                exit(main_cbdgam_2d())
            else:
                print(f"Unknown solver {chkpt['solver']} at time {chkpt['time']/2/3.14159:.2f}")
