import argparse
import pickle as pk
import sys
from sailfish.physics.kepler import OrbitalState
import matplotlib.pyplot as plt

sys.path.insert(1,"/groups/astro/davidon/sailfish/")
import sailfish


sys.path.insert(1, ".")

class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)



def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
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

            #plt.quiver(X, Y, Vx_sampled, Vy_sampled,width=0.001, scale=200)
            #plt.quiver(X, Y, Vx_sampled, Vy_sampled,width=0.003, angles='xy', scale_units='xy', scale=200, color = 'white')
            plt.quiver(X, Y, Vx_sampled, Vy_sampled,width=0.0025, angles='xy', scale_units='xy', scale=40, color = 'darkgrey')

            


        def AngularSpeed(self):

            Vx_Relative = self.Vx
            Vy_Relative = self.Vy

            TotalSpeed = np.sqrt( Vx_Relative**2 + Vy_Relative**2 )


            fig, ax = plt.subplots(figsize=[12, 9])
            #ni, nj = mesh.shape
            #xspace = np.linspace(mesh.x0,mesh.x1,ni)
            #yspace = np.linspace(mesh.y0,mesh.y1,nj)
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
            #TotalSpeed = np.sqrt( self.Vx**2 + self.Vy**2 )
            TotalSpeed = self.Vy


            print('Keys', chkpt.keys())


            print('Keys', chkpt.keys())

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
            
            plt.figure(figsize = (6,6))
            plt.plot(XPrimCent, TotalSpeed[nj//2,:]/XPrimCent, c = 'black', linewidth = 2, label = r'$\Omega(r)$')
            plt.plot(XPrimCent, [- np.sqrt(np.sign(xpos) / xpos /xpos /xpos) for xpos in XPrimCent], c = 'red', label = r'$\Omega_K(r)$')
            plt.plot(XPrimCent, [np.sqrt(np.sign(xpos) / xpos /xpos /xpos) for xpos in XPrimCent], c = 'red', linestyle='dashed', label = r'$-\Omega_K(r)$')
            plt.xlim([-0.5, 0.5])
            plt.ylim([-200,200])
            plt.axvline(x = primary.softening_length, c ='grey', linestyle = 'dashed')
            plt.axvline(x = -primary.softening_length, c ='grey', linestyle = 'dashed')
            plt.legend()
            plt.xlabel(r'$r~[a_0]$', fontsize = 12)
            plt.title(r'Angular Speed of Minidisk', fontsize = 12)
            plt.ylabel(r'$\Omega(r)$', fontsize = 12, rotation = 0)
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

            plt.figure(figsize=(12,6))
            plt.plot(Radial_Bins,f, linewidth = 2, label = 'Surface Density Standard Deviation')
            plt.xlabel(r'Radius $[a_0]$', fontsize = 16)
            plt.legend()
            plt.show()

            print('Standard Deviations',f)


    for filename in args.checkpoints:
        fig, ax     = plt.subplots(figsize=[12, 9])
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
            colorbar1.ax.tick_params(labelsize=16)

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
                
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
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
        "--cmap",
        default="magma",
        help="colormap name",
    )


    args = parser.parse_args()

    for filename in args.checkpoints:
        fig, ax     = plt.subplots(figsize=[10, 10])
        chkpt       = load_checkpoint(filename, require_solver="cbdgam_2d")
        CurrentTime = chkpt["time"]/ 2 / np.pi
        mesh        = chkpt["mesh"]
        prim        = chkpt["solution"]

        
        import cooling
        from cooling import gamma_law_index, EffectiveTemperature, cgs
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

        if args.field == 't':
            Sigma    = fields["sigma"](prim)
            Pressure = fields["pre"](prim)

            kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code    = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

            Midplane_T    = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK

            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (10/EddingtonFrac) ** 0.25
            f             = (RescaledTemp * mask_values).T

        elif args.field == 'mach':
            Sigma    = fields["sigma"](prim)
            Pressure = fields["pre"](prim)

            cs     = (gamma * Pressure / Sigma)**0.5
            ni, nj = mesh.shape
            xspace = np.linspace(mesh.x0, mesh.x1,ni)
            yspace = np.linspace(mesh.y0, mesh.y1,nj)
            X, Y   = np.meshgrid(xspace, yspace)
            primary, secondary = chkpt['point_masses']
            xprim, yprim       = primary.position_x, primary.position_y
            xsec, ysec         = secondary.position_x, secondary.position_y

            R_1     = np.sqrt((X-xprim)**2 + (Y-yprim)**2)
            R_2     = np.sqrt((X-xsec)**2  + (Y-ysec)**2)
            romega  = np.sqrt(0.5 / (R_1 + 1e-12) + 0.5 / (R_2 + 1e-12))
 
            Mach    = romega / cs
            f       = Mach.T

            plt.figure(figsize = (4,4))
            plt.title('Mach Number Profile')
            plt.plot(np.linspace(mesh.x0, mesh.x1, mesh.shape[0]), f[mesh.shape[1]//2,:], label = 'horizontal cut', c = 'tab:red', linewidth = 2)
            plt.plot(np.linspace(mesh.x0, mesh.x1, mesh.shape[0]), f[:,mesh.shape[0]//2], label = 'vertical cut'  , c = 'tab:blue', linewidth = 2)
            plt.legend()
            plt.ylim([5,20])

            #plt.plot(np.linspace(mesh.x0, mesh.x1, mesh.shape[0]), [SS73.mach_profile(r/3) for r in np.linspace(mesh.x0, mesh.x1, mesh.shape[0])], label = 'SS73 Mach Profile', linestyle = 'dashed')
            plt.savefig(args.Outputs + "/MidplaneMach.png", dpi = 300)


        elif args.field == 'tau':
            Sigma    = fields["sigma"](prim) 
            #Pressure = fields["pre"](prim)

            kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code    = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

            #Midplane_T    = ((Pressure / Sigma) * (mp_code / kb_code)).T
            #optical_depth = Sigma * kappa_code
            #Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            #EddingtonFrac = SS73._eddington_fraction
            #RescaledTemp  = Teff * (10/EddingtonFrac) ** 0.25

            f         = Sigma * kappa_code * 1.0 / (10 / SS73._eddington_fraction)

            print('Minimum Density', 10 / kappa_code)
            #Tmid = 10 **-2 * mp_code/kb_code
            #kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            #Teff = cooling.EffectiveTemperature(1e-10, kappa_code, Tmid)
            #EmittingTemp = Teff * (10/SS73._eddington_fraction)** 0.25
            #print('Emitting Temp',EmittingTemp)
            #print('Optical depth', kappa_code * 1e-10)
            

 
        import cooling

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

        if args.field == 't':
            Sigma    = fields["sigma"](prim)
            Pressure = fields["pre"](prim)

            kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code    = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

            Midplane_T    = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK

            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (10/EddingtonFrac) ** 0.25
            print('Remapping factor', 1/SS73._eddington_fraction)
            f             = (RescaledTemp * mask_values).T

        elif args.field == 'mach':
             Sigma    = fields["sigma"](prim)
             Pressure = fields["pre"](prim)
 
             cs     = (gamma * Pressure / Sigma)**0.5
             ni, nj = mesh.shape
             xspace = np.linspace(mesh.x0, mesh.x1,ni)
             yspace = np.linspace(mesh.y0, mesh.y1,nj)
             X, Y   = np.meshgrid(xspace, yspace)
             primary, secondary = chkpt['point_masses']
             xprim, yprim = primary.position_x, primary.position_y
             xsec, ysec   = secondary.position_x, secondary.position_y

             R_1     = np.sqrt((X-xprim)**2 + (Y-yprim)**2)
             R_2     = np.sqrt((X-xsec)**2  + (Y-ysec)**2)
             Omega_1 = np.sqrt(0.5 / (R_1**3 + 1e-12)) #Assuming a Keplerian disk
             Omega_2 = np.sqrt(0.5 / (R_2**3 + 1e-12))  #Assuming a Keplerian disk
             
             ROmega  = np.sqrt((R_1 * Omega_1)**2 + (R_2 * Omega_2)**2) #Assuming a Keplerian disk
 
             Mach    = ROmega / cs
             f       = Mach.T


        elif args.field == 'tau':
            Sigma    = fields["sigma"](prim) 
            #Pressure = fields["pre"](prim)

            kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code    = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

            #Midplane_T    = ((Pressure / Sigma) * (mp_code / kb_code)).T
            #optical_depth = Sigma * kappa_code
            #Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            #EddingtonFrac = SS73._eddington_fraction
            #RescaledTemp  = Teff * (10/EddingtonFrac) ** 0.25

            f         = Sigma * kappa_code * 1.0 / (10 / SS73._eddington_fraction)

            print('Minimum Density', 10 / kappa_code)
            #Tmid = 10 **-2 * mp_code/kb_code
            #kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            #Teff = cooling.EffectiveTemperature(1e-10, kappa_code, Tmid)
            #EmittingTemp = Teff * (10/SS73._eddington_fraction)** 0.25
            #print('Emitting Temp',EmittingTemp)
            #print('Optical depth', kappa_code * 1e-10)
            

        else:
            f = fields[args.field](prim).T

        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1

        if np.percentile(f, 0)> -9: 
            if np.percentile(f, 0) < -7:
                new_vmin = np.percentile(f, 0)  
            else:
                new_vmin = -7
        else:
            new_vmin = -9

        cm = ax.imshow(
            f,
            origin="lower",
            vmin=new_vmin,
            vmax=-4,
            cmap=args.cmap,
            extent=extent,
        )
        fig.colorbar(cm)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_aspect("equal")
        fig.suptitle(chkpt["time"]/2/np.pi)

        fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )

<<<<<<< HEAD
    if args.SED:
        kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
        mp_code    = cgs['mp'] / (SS73._mass)
        kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

        Sigma           = fields["sigma"](prim) 
        Pressure        = fields["pre"](prim)
        optical_depth   = Sigma * kappa_code
        Remapping_Value = chkpt['model_parameters']['target_accretion_rate'] / SS73._eddington_fraction
        mask_values     = optical_depth >= (10.0 / Remapping_Value)
        T               = np.maximum((Pressure / Sigma) * (mp_code / kb_code), 1) * mask_values

        Teff                  = EffectiveTemperature(optical_depth, T)
        RescaledTemp          = Teff * Remapping_Value ** 0.25
        mask_values           = optical_depth >= (10.0 / Remapping_Value)
        
        ev            = 1.6e-12
        E_low         = 1e-1 * ev
        E_high        = 5e4 * ev
        freq_low      = E_low  / cgs['h']
        freq_high     = E_high / cgs['h']
        E_Xray_low    = cgs['h'] * cgs['c'] / (1e-6) / 1000/ ev
        E_Xray_high   = cgs['h'] * cgs['c'] / (1e-9) / 1000/ ev

        Ev_array      = np.logspace(np.log10(E_low/1000/ev), np.log10(E_high/1000/ev), 100)  # Energy range in eV
        #freq_space    = np.logspace(np.log10(freq_low), np.log10(freq_high), len(Ev_array))  # Frequency range in Hz
        freq_space    = np.logspace(np.log10(freq_low), np.log10(freq_high), len(Ev_array))
        ni, nj        = np.shape(RescaledTemp)
        dx            = mesh.dx

        Cell_Spectra  = np.array([cooling.PlanckSpectrum(freq, RescaledTemp, length_scale_pc * cgs['pc'] * dx) for freq in freq_space])
        Spectrum      = np.sum(np.sum(Cell_Spectra, axis=1), axis=1) 
        integral      = np.trapz(Spectrum, np.log(freq_space)) 

        plt.figure(figsize=(4, 3))
        plt.plot(Ev_array, Spectrum, label = 'SED')
        plt.xscale('log')
        plt.yscale('log') 
        plt.axvline(x = E_Xray_low, linestyle='dashed', c = 'red', label = 'X-ray band')   
        plt.axvline(x = E_Xray_high, linestyle='dashed', c = 'red')        
        plt.axhline(y = integral, linestyle='dashed', c = 'black', label = 'Total Luminosity')
        plt.legend()
        plt.xlabel(r'$h\nu~[\mathrm{kev}]$')
        plt.ylabel(r'$2\pi\nu B_\nu(T)~[\mathrm{erg/s}]$')
        plt.title('Spectral Energy Distribution at t = %g'%(chkpt["time"]/ 2 / np.pi))
        plt.ylim([1e35, 2 * integral])
        plt.savefig(args.Outputs + "/SED_%g.png"%(chkpt["time"]/ 2 / np.pi), dpi=300, bbox_inches='tight')

    if args.vmap:
            Number_of_Vectors = 15
<<<<<<< HEAD
=======
    if args.vmap:
            Number_of_Vectors = 20
>>>>>>> f3c6636 (Clean diagnostic plotting. Include vmap for plot.py)
=======
>>>>>>> dac8f5d (Plot mach number in disk)
            ni, nj            = mesh.shape
            x                 = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y                 = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            Vx                = chkpt['solution'][:,:,1].T
            Vy                = chkpt['solution'][:,:,2].T

            try:
                rescaled_x = [ix for ix in x if np.abs(ix) < args.radius]
                xmin, xmax = np.where(x == np.min(rescaled_x))[0][0], np.where(x == np.max(rescaled_x))[0][0]
            except:
                rescaled_x = x
                xmin, xmax = 0,len(x)-1

            Sampling   = (xmax-xmin)//Number_of_Vectors
            X, Y       = np.meshgrid(x[xmin:xmax:Sampling, :], y[:, xmin:xmax:Sampling])
            Vx_sampled = Vx[xmin:xmax:Sampling, xmin:xmax:Sampling] 
            Vy_sampled = Vy[xmin:xmax:Sampling, xmin:xmax:Sampling]# - 0.5
            #print(np.shape(y[:, xmin:xmax:Sampling]))
            #print(np.shape(x[xmin:xmax:Sampling, :]))
            plt.quiver(
                X, Y,
                Vx_sampled, Vy_sampled,
                width=0.0025, angles='xy', scale_units='xy', scale=20, color = 'darkgrey', headwidth=4
            )



    if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)

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

    if args.plot_sink:
        from sailfish.physics.kepler import OrbitalState, PointMass
        from matplotlib.patches import Circle

        primary, secondary = chkpt['point_masses']
        Primary   = PointMass(primary.mass  , primary.position_x  , primary.position_y  , primary.velocity_x  , primary.velocity_y)
        Secondary = PointMass(secondary.mass, secondary.position_x, secondary.position_y, secondary.velocity_x, secondary.velocity_y)

        orbital_state = OrbitalState(Primary, Secondary)

        ax.scatter(primary.position_x, primary.position_y, marker = '+', s = 40, c = 'white', label = 'Point Mass')
        ax.scatter(secondary.position_x, secondary.position_y, marker = '+', s = 40, c = 'white')

        
        primarycenter   = (primary.position_x, primary.position_y)
        secondarycenter = (secondary.position_x, secondary.position_y)
        radius          = primary.sink_radius         # Radius of the circle

        primarysink   = Circle(primarycenter, radius, color='grey', fill=True, alpha=0.8)
        secondarysink = Circle(secondarycenter, radius, color='grey', fill=True, alpha=0.8)
        ax.add_patch(primarysink)
        ax.add_patch(secondarysink)

        def Position(t, a, e):
            return 0.5 * a * np.cos(t) - 0.5 * a * e, 0.5 * a * np.sqrt(1 - e**2) * np.sin(t)
        
        eccentr = chkpt['timeseries'][-1][ 2] 
        semimaj = chkpt['timeseries'][-1][ 1] 
        #eccentr = chkpt['timeseries'].eccentricity[-1]
        Orbital_Path = np.array([Position(t, semimaj, eccentr) for t in np.linspace(0,2*np.pi,1000)])
        # USE  TIMESERIES DATA TO GET A AND E AND USE THIS TO PLOT
        plt.plot( Orbital_Path[:,0], Orbital_Path[:,1], linestyle = 'dashed', c = 'grey')
        plt.plot(-Orbital_Path[:,0], Orbital_Path[:,1], linestyle = 'dashed', c = 'grey')

    if args.Outputs is None:
        plt.show()
    else:
        pngname     = args.Outputs + f"/DensityMap-{int(CurrentTime * 100):05d}.png"
        
        fig.savefig(pngname, dpi=400)


text_width   = 7.1
column_width = text_width / 2.
def configure_matplotlib():
    plt.rc('xtick' , labelsize=8)
    plt.rc('ytick' , labelsize=8)
    plt.rc('axes'  , labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', family='DejaVu Sans', size=8)
    plt.rc('text', usetex=True)
configure_matplotlib()

if __name__ == "__main__":
    for arg in sys.argv:
        if arg.endswith(".pk"):
            chkpt = load_checkpoint(arg)
            
            import numpy as np
            print('Time',chkpt['time']/2/np.pi)
            print(chkpt.keys())
            
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
        
