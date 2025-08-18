"""
Energy-conserving solver for the binary accretion problem in 2D.
"""

from typing import NamedTuple
from logging import getLogger
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.physics.circumbinary import (
    Physics,
    EquationOfState,
    ViscosityModel,
    Diagnostic,
)
from sailfish.solver_base import SolverBase
from sailfish.subdivide import subdivide, to_host, concat_on_host, lazy_reduce
from cooling import OpticalEmission, InfaredEmission, UVEmission, XrayEmission, cgs, EffectiveTemperature
import numpy as np
import warnings


logger = getLogger(__name__)


class Options(NamedTuple):
    pressure_floor: float = 1e-12
    density_floor: float = 1e-10
    velocity_ceiling: float = 1e16
    mach_ceiling: float = 1e5
    sink_emission: bool = True


def initial_condition(setup, mesh, time):
    """
    Generate a 2D array of primitive data from a mesh and a setup.
    """
    import numpy as np

    ni, nj = mesh.shape
    primitive = np.zeros([ni, nj, 4])

    for i in range(ni):
        for j in range(nj):
            setup.primitive(time, mesh.cell_coordinates(i, j), primitive[i, j])

    return primitive


class Patch:
    """
    Holds the array buffer state for the solution on a subset of the
    solution domain.
    """

    def __init__(
        self,
        time,
        primitive,
        mesh,
        index_range,
        physics,
        options,
        buffer_outer_radius,
        buffer_surface_density,
        buffer_surface_pressure,
        lib,
        xp,
        execution_context,
    ):
        i0, i1 = index_range
        ni, nj = i1 - i0, mesh.shape[1]
        self.lib = lib
        self.mesh = mesh
        self.xp = xp
        self.execution_context = execution_context
        self.time = self.time0 = time
        self.shape = (i1 - i0, nj)  # not including guard zones
        self.physics = physics
        self.options = options
        self.xl, self.yl = mesh.vertex_coordinates(i0, 0)
        self.xr, self.yr = mesh.vertex_coordinates(i1, nj)
        self.buffer_outer_radius = buffer_outer_radius
        self.buffer_surface_density = buffer_surface_density
        self.buffer_surface_pressure = buffer_surface_pressure
        self.retrograde = physics.retrograde

        def dynamic_cooling(_self):
            return (2e-9 * xp.exp(self.time / 10)) * self.physics.base_coefficient
            #return _self._base_cooling_coefficient * (2e-9 * self.xp.exp(self.time / 10))

        # Inject it as a property into the *physics instance only*
        self.physics.__class__.cooling_coefficient = property(dynamic_cooling)


        with self.execution_context:
            x0 = self.xl + 0.5 * mesh.dx
            x1 = self.xr - 0.5 * mesh.dx
            y0 = self.yl + 0.5 * mesh.dy
            y1 = self.yr - 0.5 * mesh.dy
            self.coordinate_array_x = xp.linspace(x0, x1, ni)[:, None]
            self.coordinate_array_y = xp.linspace(y0, y1, nj)[None, :]
            self.wavespeeds = self.xp.zeros(primitive.shape[:2])
            self.primitive1 = self.xp.array(primitive)
            self.primitive2 = self.xp.array(primitive)
            self.conserved0 = self.xp.zeros(primitive.shape)

    @property
    def cell_center_coordinate_arrays(self):
        """
        Return two 2d arrays, one with the cell-center X coordinates, and the
        other with the cell-center Y coordinates. The arrays are either numpy
        or cupy arrays, allocated for the device this patch is assigned to.
        """
        return self.coordinate_array_x, self.coordinate_array_y


    def point_mass_source_term(self, which_mass, gravity=False, accretion=False):
        ng = 2  # number of guard cells
        if which_mass not in (1, 2):
            raise ValueError("the mass must be either 1 or 2")

        m1, m2 = self.physics.point_masses(self.time)

        with self.execution_context:
            cons_rate = self.xp.zeros_like(self.conserved0)

            self.lib.cbdgam_2d_point_mass_source_term[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass * gravity,
                m1.softening_length,
                m1.sink_rate * accretion,
                m1.sink_radius,
                m1.sink_model.value,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass * gravity,
                m2.softening_length,
                m2.sink_rate * accretion,
                m2.sink_radius,
                m2.sink_model.value,
                which_mass,
                self.primitive1,
                cons_rate,
                int(self.physics.constant_softening),
                self.physics.gamma_law_index,
            )
            return cons_rate[ng:-ng, ng:-ng]

    def maximum_wavespeed(self):
        with self.execution_context:
            self.lib.cbdgam_2d_wavespeed[self.shape](
                self.primitive1,
                self.wavespeeds,
                self.physics.gamma_law_index,
            )
            return self.wavespeeds.max()

    def recompute_conserved(self):
        with self.execution_context:
            return self.lib.cbdgam_2d_primitive_to_conserved[self.shape](
                self.primitive1,
                self.conserved0,
                self.physics.gamma_law_index,
            )

    def advance_rk(self, rk_param, dt):
        m1, m2 = self.physics.point_masses(self.time)
        buffer_central_mass = m1.mass + m2.mass
        buffer_surface_density = self.buffer_surface_density
        buffer_surface_pressure = self.buffer_surface_pressure

        with self.execution_context:
            self.lib.cbdgam_2d_advance_rk[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.conserved0,
                self.primitive1,
                self.primitive2,
                self.physics.gamma_law_index,
                buffer_surface_density,
                buffer_surface_pressure,
                buffer_central_mass,
                self.physics.buffer_driving_rate,
                self.buffer_outer_radius,
                self.physics.buffer_onset_width,
                int(self.physics.buffer_is_enabled),
                int(self.retrograde),
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass,
                m1.softening_length,
                m1.sink_rate,
                m1.sink_radius,
                m1.sink_model.value,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass,
                m2.softening_length,
                m2.sink_rate,
                m2.sink_radius,
                m2.sink_model.value,
                self.physics.alpha,
                rk_param,
                dt,
                self.options.velocity_ceiling,
                self.physics.cooling_coefficient,
                self.options.mach_ceiling,
                self.options.density_floor,
                self.options.pressure_floor,
                int(self.physics.constant_softening),
            )

        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.primitive1, self.primitive2 = self.primitive2, self.primitive1

    def new_iteration(self):
        self.time0 = self.time
        self.recompute_conserved()

    @property
    def primitive(self):
        return self.primitive1


class Solver(SolverBase):
    """
    Adapter class to drive the cbdgam_2d C extension module.
    """

    def __init__(
        self,
        setup=None,
        mesh=None,
        time=0.0,
        solution=None,
        num_patches=1,
        mode="cpu",
        physics=dict(),
        options=dict(),
    ):
        import numpy as np

        physics["diagnostics"] = [
            Diagnostic(**v) for v in physics.get("diagnostics", [])
        ]

        self._physics = physics = Physics(**physics)
        self._options = options = Options(**options)

        if type(mesh) is not PlanarCartesian2DMesh:
            raise ValueError("solver only supports 2D cartesian mesh")

        if setup.boundary_condition != "outflow":
            raise ValueError("solver only supports outflow boundary condition")

        if physics.viscosity_model not in (
            ViscosityModel.NONE,
            ViscosityModel.CONSTANT_ALPHA,
        ):
            raise ValueError("solver only supports constant-alpha viscosity")

        if physics.eos_type != EquationOfState.GAMMA_LAW:
            raise ValueError("solver only supports isothermal equation of states")

        xp = get_array_module(mode)
        ng = 2  # number of guard zones
        nq = 4  # number of conserved quantities
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()
        lib = Library(code, mode=mode, debug=False)

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")
        logger.info(f"boundary condition is outflow")

        self.mesh = mesh
        self.setup = setup
        self.num_guard = ng
        self.num_cons = nq
        self.xp = xp
        self.patches = []
        ni, nj = mesh.shape
        self.domain_radius = self.mesh.x1
        self.buffer_onset_width = 0.1
        self.infared_cache = None
        self.optical_cache = None
        self.uv_cache      = None
        self.xray_cache    = None

        if solution is None:
            primitive = initial_condition(setup, mesh, time)
        else:
            primitive = solution

        if physics.buffer_is_enabled:
            # Here we sample the initial condition at the buffer onset radius
            # to determine the disk surface density at the radius where the
            # buffer begins to ramp up. This procedure makes sense as long as
            # the initial condition is axisymmetric.
            buffer_prim = [0.0] * 4
            buffer_outer_radius = mesh.x1  # this assumes the mesh is a centered squared
            buffer_onset_radius = buffer_outer_radius - physics.buffer_onset_width
            setup.primitive(time, [buffer_onset_radius, 0.0], buffer_prim)
            buffer_surface_density = buffer_prim[0]
            buffer_surface_pressure = buffer_prim[3]
        else:
            buffer_outer_radius = 0.0
            buffer_surface_density = 0.0
            buffer_surface_pressure = 0.0

        for n, (a, b) in enumerate(subdivide(ni, num_patches)):
            prim = np.zeros([b - a + 2 * ng, nj + 2 * ng, nq])
            prim[ng:-ng, ng:-ng] = primitive[a:b]
            patch = Patch(
                time,
                prim,
                mesh,
                (a, b),
                physics,
                options,
                buffer_outer_radius,
                buffer_surface_density,
                buffer_surface_pressure,
                lib,
                xp,
                execution_context(mode, device_id=n % num_devices(mode)),
            )
            self.patches.append(patch)


    @property
    def solution(self):
        return concat_on_host(
            [p.primitive for p in self.patches], (self.num_guard, self.num_guard)
        )

    @property
    def primitive(self):
        """
        This solver uses primitive data as the solution array.
        """
        return None

    @property
    def Length_Scale_CGS(self):
         return self.setup.length_scale_pc * cgs['pc']

    @property
    def kb_code(self):
        return cgs['kb'] / (self.setup.SS73._mass * self.setup.SS73._length**2 / self.setup.SS73._time**2)

    @property
    def mp_code(self):
        return cgs['mp'] / (self.setup.SS73._mass)

    @property
    def kappa_code(self):
        return cgs['kappa'] / (self.setup.SS73._length**2 / self.setup.SS73._mass)
    
    @property
    def point_mass_distance(self, patch):
        x_, y_           = patch.cell_center_coordinate_arrays
        x, y             = x_[-1,0]*self.xp.ones(np.shape(x_)[0]+4) ,  y_[0,-1]*self.xp.ones(np.shape(y_)[1]+4)
        x[2:-2]          = x_[:,0]
        y[2:-2]          = y_[0,:]
        X, Y             = np.meshgrid(x, y, indexing='ij')

        m1, m2  = patch.physics.point_masses(patch.time)
        r1      = ((X-m1.position_x)**2 + (Y-m1.position_y)**2) 
        r2      = ((X-m2.position_x)**2 + (Y-m2.position_y)**2) 
        return r1, r2

    @property
    def Precompute_Band_Luminosities(self):
        logT_low  = 0
        logT_high = 10

        Temperature_Range    = np.logspace(logT_low,logT_high,int(1e6)) 
        Log_Temperature_Diff = np.diff(np.log10(Temperature_Range))[0]
        #Length_scale_cgs     = setup.length_scale_pc * cgs['pc']
        if self.optical_cache is None:
            optical_emission   = OpticalEmission(Temperature_Range, self.Length_Scale_CGS)
            self.optical_cache = optical_emission
        else:
            pass

        if self.infared_cache is None:
            infared_emission   = InfaredEmission(Temperature_Range, self.Length_Scale_CGS)
            self.infared_cache = infared_emission
        else:
            pass

        if self.uv_cache is None:
            uv_emission   = UVEmission(Temperature_Range, self.Length_Scale_CGS)
            self.uv_cache = uv_emission
        else:
            pass

        if self.xray_cache is None:
            xray_emission   = XrayEmission(Temperature_Range, self.Length_Scale_CGS)
            self.xray_cache = xray_emission
        else:
            pass

        return [Temperature_Range, Log_Temperature_Diff, np.asarray(self.optical_cache), np.asarray(self.infared_cache), np.asarray(self.uv_cache), np.asarray(self.xray_cache)]


    def detect_density_floor(self, patch):
        rho  = patch.primitive1[:, :, 0]
        mask = rho <= patch.options.density_floor * 1.01
        return mask.sum() ## Return and check this sum over patches. Is it multiplied by da? If not is it a bottleneck?
    
    def detect_pressure_floor(self, patch):
        pressure = patch.primitive1[:, :, 0]
        mask     = pressure <= patch.options.pressure_floor * 1.01
        return mask.sum() ## Return and check this sum over patches. Is it multiplied by da? If not is it a bottleneck?


    def detect_density_floor(self, patch):
        rho  = patch.primitive1[:, :, 0]
        mask = rho <= patch.options.density_floor * 1.01
        return mask.sum() ## Return and check this sum over patches. Is it multiplied by da? If not is it a bottleneck?
    
    def detect_pressure_floor(self, patch):
        pressure = patch.primitive1[:, :, 0]
        mask     = pressure <= patch.options.pressure_floor * 1.01
        return mask.sum() ## Return and check this sum over patches. Is it multiplied by da? If not is it a bottleneck?


    def Interpolate_Band_Luminosity(self, patch):
        Precomputed      = list(self.Precompute_Band_Luminosities)
        Precomputed[2]   = self.xp.asarray(Precomputed[2])
        Precomputed[3]   = self.xp.asarray(Precomputed[3])
        Precomputed[4]   = self.xp.asarray(Precomputed[4])
        Precomputed[5]   = self.xp.asarray(Precomputed[5])
        Precomputed_low  = self.xp.log10(Precomputed[0][0])
        Precomputed_high = self.xp.log10(Precomputed[0][-1])


        Sigma            = patch.primitive[:, :, 0]
        T                = self.xp.maximum((patch.primitive[:, :, 3] / Sigma) * (self.mp_code / self.kb_code), 10**Precomputed_low)
        optical_depth    = Sigma * self.kappa_code


        T                = np.maximum((patch.primitive[:, :, 3] / Sigma) * (self.mp_code / self.kb_code), Precomputed_low)


        Sigma            = patch.primitive[:, :, 0]
        T                = self.xp.maximum((patch.primitive[:, :, 3] / Sigma) * (self.mp_code / self.kb_code), 10**Precomputed_low)
        optical_depth    = Sigma * self.kappa_code

        # Note that the temperature mapping only occurs on the effective temperature, not the actual temperature. Hence
        # the code unit optical depth is used to compute the surface temperature before the mapping is applied
        Teff                  = EffectiveTemperature(optical_depth, T)
        RescaledTemp          = Teff * self.setup.Mdrop ** 0.25
        RescaledDepth         = optical_depth * self.setup.Mdrop ** (7./10.)
        Bolometric_Luminosity = 2 * cgs['sigmab'] * RescaledTemp ** 4 * self.Length_Scale_CGS**2



        if not patch.options.sink_emission:
            x_, y_           = patch.cell_center_coordinate_arrays
            x, y             = x_[-1,0]*self.xp.ones(np.shape(x_)[0]+4) ,  y_[0,-1]*self.xp.ones(np.shape(y_)[1]+4)
            x[2:-2]          = x_[:,0]
            y[2:-2]          = y_[0,:]
            X, Y             = np.meshgrid(x, y, indexing='ij')
            m1, m2  = patch.physics.point_masses(patch.time)
            r1_mask = ((X-m1.position_x)**2 + (Y-m1.position_y)**2) > m1.sink_radius**2
            r2_mask = ((X-m2.position_x)**2 + (Y-m2.position_y)**2) > m2.sink_radius**2

        else:
            r1_mask = 1
            r2_mask = 1


        transparent_mask = (RescaledDepth >= self.setup.OpticalDepthFloor)
        mask_interp_min  = (RescaledTemp * transparent_mask >= 1.01 * 10**Precomputed_low) # Rescaled Temperatures should not be below interpolation minimum
        mask             = r1_mask * r2_mask * mask_interp_min 

        N0               = self.xp.floor(Progress).astype(int)
        Progress      = (np.log10(RescaledTemp) - Precomputed_low)/ Precomputed[1]

        
        if np.min(Progress) < 0:
            raise IndexError("Interpolated temperature range limit needs to be lower in cbdgam_2d.py. Current value is logT_min = %g"%(np.log10(self.Precompute_Band_Luminosities[0][0])), 
                "while the temperature dropped down to a value of logT = %g"%(np.log10(np.min(RescaledTemp))))

        Teff                  = EffectiveTemperature(optical_depth, T)
        RescaledTemp          = Teff * self.setup.AccretionRateRescaling ** 0.25
        Bolometric_Luminosity = 2 * cgs['sigmab'] * RescaledTemp ** 4 * self.Length_Scale_CGS**2

        transparent_mask = (optical_depth >= 1.0)#.astype(self.xp.float64)
        mask_all_vals_if = (RescaledTemp * transparent_mask >= 1.01)#.astype(self.xp.float64)
        # High temp cutoff??



        if self.xp.min(Progress) < 0:
            raise IndexError(
            f"Interpolated temperature range limit needs to be lower in cbdgam_2d.py. "
            f"Current value is logT_min = {self.xp.log10(self.Precompute_Band_Luminosities[0][0])}, "
            f"while the temperature dropped down to a value of logT = {self.xp.log10(self.xp.min(RescaledTemp))}"
        )

        N0            = self.xp.floor(Progress).astype(int)
        Bracket_N0_N1 = Progress - N0

        Progress         = (self.xp.log10(RescaledTemp) - Precomputed_low)/ Precomputed[1]
        N0               = self.xp.floor(Progress).astype(int)
        Bracket_N0_N1    = Progress - N0


        try:
            Optical_N0 = self.xp.take(Precomputed[2],N0, axis=0)
            Optical_N1 = self.xp.take(Precomputed[2],N0+1,axis=0)
            Infared_N0 = self.xp.take(Precomputed[3],N0  ,axis=0)
            Infared_N1 = self.xp.take(Precomputed[3],N0+1,axis=0)
            UV_N0      = self.xp.take(Precomputed[4],N0  ,axis=0)
            UV_N1      = self.xp.take(Precomputed[4],N0+1,axis=0)
            Xray_N0    = self.xp.take(Precomputed[5],N0  ,axis=0)
            Xray_N1    = self.xp.take(Precomputed[5],N0+1,axis=0)

            Optical_N0 = self.xp.take(Precomputed[2], N0, axis=0)
            Optical_N1 = self.xp.take(Precomputed[2],N0+1,axis=0)
            Infared_N0 = self.xp.take(Precomputed[3],N0  ,axis=0)
            Infared_N1 = self.xp.take(Precomputed[3],N0+1,axis=0)

            Interpolated_Optical = Optical_N0 + Bracket_N0_N1 * (Optical_N1-Optical_N0)
            Interpolated_Infared = Infared_N0 + Bracket_N0_N1 * (Infared_N1-Infared_N0)
            Interpolated_UV      = UV_N0      + Bracket_N0_N1 * (UV_N1-UV_N0)
            Interpolated_Xray    = Xray_N0    + Bracket_N0_N1 * (Xray_N1-Xray_N0)

            Interpolated_Optical  *= mask
            Interpolated_Infared  *= mask
            Interpolated_UV       *= mask
            Interpolated_Xray     *= mask
            Bolometric_Luminosity *= mask

            #return Interpolated_Optical, Interpolated_Infared, Bolometric_Luminosity, sum(~mask_all_vals_if)
            return Interpolated_Infared, Interpolated_Optical, Interpolated_UV, Interpolated_Xray, Bolometric_Luminosity, sum(~mask_interp_min), self.xp.max(RescaledTemp)

            Interpolated_Optical  *= mask_all_vals_if
            Interpolated_Infared  *= mask_all_vals_if
            Bolometric_Luminosity *= mask_all_vals_if

            return Interpolated_Optical, Interpolated_Infared, Bolometric_Luminosity, sum(~mask_all_vals_if)

        
        except IndexError as e:


            if self.xp.max(RescaledTemp) > self.Precompute_Band_Luminosities[0][-2]:
                raise IndexError("Interpolated temperature range limit needs to be higher in cbdgam_2d.py. Current value is logT_max = %g"%(self.xp.log10(self.Precompute_Band_Luminosities[0][-1])),
                    "while the temperature reached a value of logT = %g"%(self.xp.log10(self.xp.max(RescaledTemp))))
            elif np.min(Sigma) == 0.0:
                logger.info(f"Lightcurve reductions failed at time={self.time:0.4f} due to zero surface density")
                warnings.warn(f"Lightcurve reductions failed at time={self.time:0.4f} due to zero surface density")
                return self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), sum(~mask_interp_min), 0
            else:
                print('SOMETHING ELSE WENT WRONG, FIGURE IT OUT.')
            

            warnings.warn("Interpolated temperature range needs to be higher in cbdgam_2d.py. Current value is logT = %g"%(np.log10(self.Precompute_Band_Luminosities[0][-1])), UserWarning)
            return 0., 0.
        except TypeError as e:
            warnings.warn("The rescaled, effective temperature inside a cell was %g"%(RescaledTemp), UserWarning)
            return 0., 0.
            if RescaledTemp.any() > self.Precompute_Band_Luminosities[0][-1]:

            if np.max(RescaledTemp) > self.Precompute_Band_Luminosities[0][-2]:

                raise IndexError("Interpolated temperature range limit needs to be higher in cbdgam_2d.py. Current value is logT_min = %g"%(np.log10(self.Precompute_Band_Luminosities[0][-1])),
                    "while the temperature reached a value of logT = %g"%(np.log10(np.min(RescaledTemp))))

                raise IndexError("Interpolated temperature range limit needs to be higher in cbdgam_2d.py. Current value is logT_max = %g"%(np.log10(self.Precompute_Band_Luminosities[0][-1])),
                    "while the temperature reached a value of logT = %g"%(np.log10(np.max(RescaledTemp))))

            elif np.min(Sigma) == 0.0:
                logger.info(f"Lightcurve reductions failed at time={self.time:0.4f} due to zero surface density")
                warnings.warn(f"Lightcurve reductions failed at time={self.time:0.4f} due to zero surface density")
                return np.zeros_like(Sigma), np.zeros_like(Sigma)


            elif RescaledTemp.any() < self.Precompute_Band_Luminosities[0][0]:
                raise IndexError("Interpolated temperature range limit needs to be lower in cbdgam_2d.py. Current value is logT = %g"%(np.log10(self.Precompute_Band_Luminosities[0][0])))


            if self.xp.max(RescaledTemp) > self.Precompute_Band_Luminosities[0][-2]:
                raise IndexError("Interpolated temperature range limit needs to be higher in cbdgam_2d.py. Current value is logT_max = %g"%(self.xp.log10(self.Precompute_Band_Luminosities[0][-1])),
                    "while the temperature reached a value of logT = %g"%(self.xp.log10(self.xp.max(RescaledTemp))))
            elif np.min(Sigma) == 0.0:
                logger.info(f"Lightcurve reductions failed at time={self.time:0.4f} due to zero surface density")
                warnings.warn(f"Lightcurve reductions failed at time={self.time:0.4f} due to zero surface density")
                return self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), self.xp.zeros_like(Sigma), sum(~mask_all_vals_if), 0
            else:
                print('SOMETHING ELSE WENT WRONG, FIGURE IT OUT.')

            

    def optical_luminosity(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[0]

    def infared_luminosity(self,patch):
        return 2 * self.Interpolate_Band_Luminosity(patch)[0]
    
    def optical_luminosity(self,patch):
        return 2 * self.Interpolate_Band_Luminosity(patch)[1]
    
    def uv_luminosity(self,patch):
        return 2 * self.Interpolate_Band_Luminosity(patch)[2]
    
    def xray_luminosity(self,patch):
        return 2 * self.Interpolate_Band_Luminosity(patch)[3]
    
    def bolometric_luminosity(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[4]
    def Uncounted_Cells(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[5].sum()
    
    def MaxTemperature(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[6] 

    
    def bolometric_luminosity(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[2]
    
    def Uncounted_Cells(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[3].sum()


    
        return self.Interpolate_Band_Luminosity(patch)[2]
    
    def Uncounted_Cells(self,patch):
        return self.Interpolate_Band_Luminosity(patch)[3].sum()

    def runtime_reductions(self):
        """
        Generate runtime reductions on the solution data for time series.
        """
        diagnostics = self._physics.diagnostics
        udots1_acc = [p.point_mass_source_term(1, accretion=True) for p in self.patches]
        udots2_acc = [p.point_mass_source_term(2, accretion=True) for p in self.patches]
        udots1_grv = [p.point_mass_source_term(1, gravity=True) for p in self.patches]
        udots2_grv = [p.point_mass_source_term(2, gravity=True) for p in self.patches]
        da = self.mesh.dx * self.mesh.dy

        def get_field(patch, quantity, cut, mass, gravity=False, accretion=False, buffer=False):
            """
            Return one of the udot fields: for a particular patch, conserved
            variable quantity, radial cut (optional), and point mass (either
            1, 2, or 'both'), term (either 'acc' or 'grv').
            """
            x, y = patch.cell_center_coordinate_arrays
                if cut is not None:
                    r0, r1 = cut
                    return f * (r0 < r) * (r < r1)
                else:
                    return f

            if quantity == "mdot":
                return get_field(patch, 0, cut, mass, gravity, accretion, buffer)

            if quantity == "torque":
                fx = get_field(patch, 1, cut, mass, gravity, accretion, buffer)
                fy = get_field(patch, 2, cut, mass, gravity, accretion, buffer)
                return x * fy - y * fx

            if quantity == "sigma_m1":
                sigma = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 0])
                cos_phi = x / r
                sin_phi = y / r
                return sigma * (cos_phi + 1.0j * sin_phi)

            if quantity == "eccentricity_vector":
                sigma = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 0])
                vx = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 1])
                vy = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 2])
                GM = 1.0
                v_dot_v = vx * vx + vy * vy
                v_dot_r = vx * x + vy * y
                ex = (v_dot_v * x - v_dot_r * vx) / GM - x / r
                ey = (v_dot_v * y - v_dot_r * vy) / GM - y / r
                return sigma * (ex + 1.0j * ey)

            if quantity == "angular_momentum":
                sigma = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 0])
                vx = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 1])
                vy = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 2])
                return sigma * (x * vy - y * vx)

            if quantity == "buffer_torque":
                fx = get_field(patch, 1, cut, mass, False, False, buffer=True)
                fy = get_field(patch, 2, cut, mass, False, False, buffer=True)
                return x * fy - y * fx

            if quantity == "buffer_mass_rate":
                return get_field(patch, 0, cut, mass, False, False, buffer=True)

            if quantity == "power":
                fx = get_field(patch, 1, cut, mass, gravity, accretion, buffer)
                fy = get_field(patch, 2, cut, mass, gravity, accretion, buffer)
                if mass == 1:
                    m1, m2 = self._physics.point_masses(self.time)
                    vx1, vy1 = m1.velocity_x, m1.velocity_y
                    return vx1 * fx + vy1 * fy
                elif mass == 2:
                    m1, m2 = self._physics.point_masses(self.time)
                    vx2, vy2 = m2.velocity_x, m2.velocity_y
                    return vx2 * fx + vy2 * fy
                else:
                    raise ValueError("Mass option for 'power' must be 1 or 2.")


            if quantity == "Accreted_energy":
                return get_field(patch, 3, cut, mass="both", gravity=False, accretion=True, buffer=False)

            if quantity == "optical":
                return self.optical_luminosity(patch)

            if quantity == "infared":
                return self.infared_luminosity(patch)
     
            if quantity == "floor":
                return self.detect_density_floor(patch)



            if quantity == "energy":
                Energy = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 3])
                return Energy


            if quantity == "Accreted_energy":
                return get_field(patch, 3, cut, mass="both", gravity=False, accretion=True, buffer=False)

            if quantity == "optical":
                return self.optical_luminosity(patch)

            if quantity == "infared":
                return self.infared_luminosity(patch)
     
            if quantity == "floor":
                return self.detect_density_floor(patch)


            q = quantity
            i = self.patches.index(patch)

            if accretion:
                udots1 = udots1_acc
                udots2 = udots2_acc
            elif gravity:
                udots1 = udots1_grv
                udots2 = udots2_grv

            if mass == "both":
                f = udots1[i][..., q] + udots2[i][..., q]
            elif mass == 1:
                f = udots1[i][..., q]
            elif mass == 2:
                f = udots2[i][..., q]

            return apply_radial_cut(f)

        def get_sum_fields(d):
            result = []
            for p in self.patches:
                with p.execution_context:

                    if d.quantity == "optical":
                        f = self.optical_luminosity(p)
                        result.append(f.sum())
                    
                    elif d.quantity == "infared":
                        f = self.infared_luminosity(p)
                        result.append(f.sum())


                    elif d.quantity == "uv":
                        f = self.uv_luminosity(p)
                        result.append(f.sum())

                    elif d.quantity == "xray":
                        f = self.xray_luminosity(p)
                        result.append(f.sum())


                    elif d.quantity == "bolometric":
                        f = self.bolometric_luminosity(p)
                        result.append(f.sum())

                    #elif d.quantity == "uncounted_cells_in_lc":
                    #    f = self.Uncounted_Cells(p)
                    #    result.append(f.sum())

                    else:
                        f = get_field(
                            p,
                            d.quantity,
                            d.radial_cut,
                            d.which_mass,
                            gravity=d.gravity,
                            accretion=d.accretion,
                        )
                        result.append(f.sum())
            return result

        pass1 = []
        pass2 = []


        import sailfish.physics.kepler as kepler
        m1, m2 = self._physics.point_masses(self.time)                                             # These are different PointMass structs with
        m1 = kepler.PointMass(m1.mass, m1.position_x, m1.position_y, m1.velocity_x, m1.velocity_y) # different attributes....
        m2 = kepler.PointMass(m2.mass, m2.position_x, m2.position_y, m2.velocity_x, m2.velocity_y) 
        orbital_state = kepler.OrbitalState(primary=m1, secondary=m2)

        for d in diagnostics:
            if d.quantity == "time":
                pass1.append(self.time / self.setup.reference_time_scale)
            elif d.quantity == "semimajor-axis":
                pass1.append(orbital_state.semimajor_axis)
            elif d.quantity == "eccentricity":
                pass1.append(orbital_state.eccentricity)

            elif d.quantity == 'density_floor':
                pass1.append(float(sum(self.detect_density_floor(p) for p in self.patches))) # double check again
            elif d.quantity == 'pressure_floor':
                pass1.append(float(sum(self.detect_pressure_floor(p) for p in self.patches))) # double check again
            elif d.quantity == 'uncounted_cells_in_lc':
                pass1.append(float(sum(self.Uncounted_Cells(p) for p in self.patches))) # double check again

            elif d.quantity == 'max_temperature':
                pass1.append(float(max(self.MaxTemperature(p) for p in self.patches)))


            elif d.quantity == 'floor':
                return int(sum(self.detect_density_floor(p) for p in self.patches))
            elif d.quantity == "optical":
                return sum(self.optical_luminosity(p) for p in self.patches)
            elif d.quantity == "infared":
                return sum(self.infared_luminosity(p) for p in self.patches)

            

            else:
                pass1.append(get_sum_fields(d))

        for item in pass1:
            if type(item) is not float:
                pass2.append(sum(to_host(x) for x in item) * da) # weight by area area in code units
            else:
                pass2.append(item)

        return pass2


    @property
    def time(self):
        return self.patches[0].time

    @property
    def options(self):
        return self._options._asdict()

    @property
    def physics(self):
        return self._physics._asdict()

    @property
    def recommended_cfl(self):
        return 0.1

    @property
    def maximum_cfl(self):
        return 0.4

    def maximum_wavespeed(self):
        return lazy_reduce(
            max,
            float,
            (patch.maximum_wavespeed for patch in self.patches),
            (patch.execution_context for patch in self.patches),
        )

    def advance(self, dt):
        self.new_iteration()
        self.advance_rk(0.0, dt)
        self.advance_rk(0.5, dt)

    def advance_rk(self, rk_param, dt):
        self.set_bc("primitive1")
        for patch in self.patches:
            patch.advance_rk(rk_param, dt)



    def set_bc(self, array):
        ng = self.num_guard
        num_patches = len(self.patches)
        for i0 in range(num_patches):
            il = (i0 + num_patches - 1) % num_patches
            ir = (i0 + num_patches + 1) % num_patches
            pl = getattr(self.patches[il], array)
            pc = getattr(self.patches[i0], array)
            pr = getattr(self.patches[ir], array)
            self.set_bc_patch(pl, pc, pr, i0)

    def set_bc_patch(self, pl, pc, pr, patch_index):
        ni, nj = self.mesh.shape
        ng = self.num_guard

        with self.patches[patch_index].execution_context:
            # 1. write to the guard zones of pc, the internal BC
            pc[:+ng] = pl[-2 * ng : -ng]
            pc[-ng:] = pr[+ng : +2 * ng]

            # 2. Set outflow BC on the left/right patch edges
            if patch_index == 0:
                for i in range(ng):
                    pc[i] = pc[ng]
            if patch_index == len(self.patches) - 1:
                for i in range(pc.shape[0] - ng, pc.shape[0]):
                    pc[i] = pc[-ng - 1]

            # 3. Set outflow BC on bottom and top edges
            for i in range(ng):
                pc[:, i] = pc[:, ng]

            for i in range(pc.shape[1] - ng, pc.shape[1]):
                pc[:, i] = pc[:, -ng - 1]

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
