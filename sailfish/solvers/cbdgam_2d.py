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
from sailfish.physics.cooling import OpticalEmission, InfaredEmission, UVEmission, XrayEmission, cgs, EffectiveTemperature
import numpy as np
#from cupy import fuse

try:
    import cupy as cp
    from cupy import fuse
    @fuse()
    def interpolate_band(N0, N1, frac):
        return N0 + frac * (N1 - N0)
except Exception:
    def interpolate_band(N0, N1, frac):
        return N0 + frac * (N1 - N0)



logger = getLogger(__name__)


class Options(NamedTuple):
    pressure_floor   : float = 1e-12
    density_floor    : float = 1e-10
    velocity_ceiling : float = 1e16
    mach_ceiling     : float = 1e5
    sink_emission    : bool  = False


def initial_condition(setup, mesh, time):
    """
    Generate a 2D array of primitive data from a mesh and a setup.
    """

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
        Mdot_inf,
        buffer_surface_density_onset,
        buffer_pressure_onset,
        surface_density_powerlaw,
        pressure_powerlaw,
        buffer_ell0,
        lib,
        xp,
        execution_context,
    ):
        i0, i1                             = index_range
        ni, nj                             = i1 - i0, mesh.shape[1]
        self.lib                           = lib
        self.mesh                          = mesh
        self.xp                            = xp
        self.execution_context             = execution_context
        self.time                          = self.time0 = time
        self.shape                         = (i1 - i0, nj)  # not including guard zones
        self.physics                       = physics
        self.options                       = options
        self.xl, self.yl                   = mesh.vertex_coordinates(i0, 0)
        self.xr, self.yr                   = mesh.vertex_coordinates(i1, nj)
        self.retrograde                    = physics.retrograde
        self.buffer_outer_radius           = buffer_outer_radius
        self.buffer_surface_density_onset  = buffer_surface_density_onset
        self.buffer_pressure_onset         = buffer_pressure_onset
        self.surface_density_powerlaw      = surface_density_powerlaw
        self.pressure_powerlaw             = pressure_powerlaw
        self.Mdot_inf                      = Mdot_inf
        self.buffer_ell0                   = buffer_ell0
        self._iteration                    = 0

        # option to vary the cooling coefficient dynamically
        def dynamic_cooling(_self):
            #return (2e-9 * xp.exp(self.time / 10)) * self.physics.dynamic_cooling_base
            return self.physics.dynamic_cooling_base 
            
        # Inject it back into physics as a property
        self.physics.__class__.cooling_coefficient = property(dynamic_cooling)
        self.physics.__class__.dynamic_cooling     = (self.physics.cooling_coefficient==self.physics.dynamic_cooling_base )


        with self.execution_context:
            x0 = self.xl + 0.5 * mesh.dx
            x1 = self.xr - 0.5 * mesh.dx
            y0 = self.yl + 0.5 * mesh.dy
            y1 = self.yr - 0.5 * mesh.dy
            self.coordinate_array_x = xp.linspace(x0, x1, ni)[:, None]
            self.coordinate_array_y = xp.linspace(y0, y1, nj)[None, :]
            self.r                  = xp.sqrt(self.coordinate_array_x**2 + self.coordinate_array_y**2)
            self.wavespeeds = self.xp.zeros(primitive.shape[:2])
            self.primitive1 = self.xp.array(primitive)
            self.primitive2 = self.xp.array(primitive)
            self.conserved0 = self.xp.zeros(primitive.shape)

            import numpy as np
            if not np.isfinite(self.primitive1).all():
                raise RuntimeError(
                    f"[Patch.__init__] Non-finite primitive1 immediately after GPU upload "
                    f"on device {self.execution_context.id}.\n"
                    f"  CPU Sigma range: {primitive[..., 0].min()} – {primitive[..., 0].max()}\n"
                    f"  CPU Pressure range: {primitive[..., 3].min()} – {primitive[..., 3].max()}\n"
                    f"  GPU Sigma range: {self.primitive1[..., 0].min()} – {self.primitive1[..., 0].max()}\n"
                    f"  GPU Pressure range: {self.primitive1[..., 3].min()} – {self.primitive1[..., 3].max()}"
                )

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


    def buffer_source_term(self):
        """
        (Adopted from C. Tiede for isothermal)
        Return an array of the rates of conserved quantities, resulting from
        the application of the buffer source terms near the outer boundary
        """
        ng                  = 2
        m1, m2              = self.physics.point_masses(self.time)
        buffer_central_mass = m1.mass + m2.mass

        with self.execution_context:
            conserved1 = self.xp.zeros_like(self.conserved0)
            cons_rate  = self.xp.zeros_like(self.conserved0)

            self.lib.cbdgam_2d_primitive_to_conserved[self.shape](
                self.primitive1,
                conserved1,
                self.physics.gamma_law_index
            )


            params = self.xp.ascontiguousarray(self.xp.array([
                self.xl, self.xr, self.yl, self.yr,
                self.physics.gamma_law_index,
                self.buffer_surface_density_onset,
                self.buffer_pressure_onset,
                self.surface_density_powerlaw,
                self.pressure_powerlaw,
                buffer_central_mass,
                self.physics.buffer_driving_rate,
                self.buffer_outer_radius,
                self.physics.buffer_onset_width,
                self.Mdot_inf,
                self.buffer_ell0,
                float(int(self.physics.buffer_is_enabled)),
                float(int(self.retrograde)),
            ], dtype=self.xp.float64))

            self.lib.cbdgam_2d_buffer_source_term[self.shape](
                params,
                conserved1,
                cons_rate
            )

        return cons_rate[ng:-ng, ng:-ng]

    def maximum_wavespeed(self):
        with self.execution_context:
            self.lib.cbdgam_2d_wavespeed[self.shape](
                self.primitive1,
                self.wavespeeds,
                self.physics.gamma_law_index,
            )
            if self._iteration % 1000 == 0:
                if not self.xp.isfinite(self.wavespeeds).all():
                    raise RuntimeError(
                        f"[Patch.maximum_wavespeed] Non-finite values in wavespeeds "
                        f"on device {self.execution_context.id}."
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
        m1, m2              =  self.physics.point_masses(self.time)
        buffer_central_mass = m1.mass + m2.mass

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
                self.buffer_surface_density_onset,
                self.buffer_pressure_onset,
                self.surface_density_powerlaw,
                self.pressure_powerlaw,
                buffer_central_mass,
                self.physics.buffer_driving_rate,
                self.buffer_outer_radius,
                self.physics.buffer_onset_width,
                self.Mdot_inf,
                self.buffer_ell0,
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
                self.physics.viscosity_coefficient,
                self.physics.viscosity_model.value,
                rk_param,
                dt,
                self.options.velocity_ceiling,
                self.physics.cooling_coefficient,
                self.options.mach_ceiling,
                self.options.density_floor,
                self.options.pressure_floor,
                int(self.physics.constant_softening)
            )

            if self._iteration % 1000 == 0:
                rho = self.primitive2[..., 0]
                pre = self.primitive2[..., 3]
                if not self.xp.isfinite(rho).all() or not self.xp.isfinite(pre).all():
                    raise RuntimeError(
                        f"[Patch.advance_rk] Non-finite values in primitive2 after advance_rk "
                        f"on device {self.execution_context.id}, time={self.time}, rk_param={rk_param}, dt={dt}.\n"
                        f"  Sigma range: {float(rho.min())} – {float(rho.max())}\n"
                        f"  Pressure range: {float(pre.min())} – {float(pre.max())}"
                    )

        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.primitive1, self.primitive2 = self.primitive2, self.primitive1

    def new_iteration(self):
        self.time0 = self.time
        self._iteration += 1
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
            ViscosityModel.CONSTANT_NU,
        ):
            raise ValueError(
                "solver only supports constant-alpha or constant-nu viscosity"
            )

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

        self.mesh                  = mesh
        self.setup                 = setup
        self.ng                    = ng
        self.num_cons              = nq
        self.xp                    = xp
        self.patches               = []
        ni, nj                     = mesh.shape
        self.domain_radius         = self.mesh.x1
        self.Mdot_inf              = self.setup.SS73.Mdot_inf
        self.buffer_onset_radius   = self.domain_radius - physics.buffer_onset_width
        self.live_buffer           = self.setup.live_buffer
        self.live_buffer_cadence   = self.setup.live_buffer_cadence
        x                          = self.xp.array([self.mesh.cell_coordinates(i, 0)[0] for i in range(ni)])
        y                          = self.xp.array([self.mesh.cell_coordinates(0, j)[1] for j in range(nj)])
        self.X, self.Y             = self.xp.meshgrid(x, y, indexing="xy")
        if solution is None:
            primitive = initial_condition(setup, mesh, time)
        else:
            primitive = solution

        if physics.buffer_is_enabled:
            buffer_prim                        = [0.0] * 4
            buffer_outer_radius                = mesh.x1  # this assumes the mesh is a centered squared
            buffer_onset_radius                = buffer_outer_radius - physics.buffer_onset_width
            setup.primitive(time, [buffer_onset_radius, 0.0], buffer_prim)
            surface_density_powerlaw           = setup.surface_density_powerlaw
            pressure_powerlaw                  = setup.pressure_powerlaw
            # These values are only at initialization and can be overwritten in driver.append_timeseries
            buffer_surface_density_onset       = buffer_prim[0]
            buffer_pressure_onset              = buffer_prim[3]
            buffer_ell0                        = getattr(setup, 'ell0', 0.0)
        else:
            buffer_outer_radius                = 0.0
            buffer_surface_density_onset       = 0.0
            buffer_pressure_onset              = 0.0
            surface_density_powerlaw           = 0.0
            pressure_powerlaw                  = 0.0
            buffer_ell0                        = 0.0

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
                self.Mdot_inf,
                buffer_surface_density_onset,
                buffer_pressure_onset,
                surface_density_powerlaw,
                pressure_powerlaw,
                buffer_ell0,
                lib,
                xp,
                execution_context(mode, device_id=n % num_devices(mode)),
            )
            self.patches.append(patch)



    @property
    def solution(self):
        return concat_on_host(
            [p.primitive for p in self.patches], (self.ng, self.ng)
        )

    @property
    def primitive(self):
        """
        This solver uses primitive data as the solution array.
        """
        return None
    
    def detect_density_floor(self, patch):
        with patch.execution_context:
            rho  = patch.primitive1[:, :, 0]
            mask = rho <= patch.options.density_floor * 1.01
            s = self.xp.sum(mask)
            return float(s.get() if hasattr(s, "get") else s)

    def detect_pressure_floor(self, patch):
        with patch.execution_context:
            pressure = patch.primitive1[:, :, 3]
            mask     = pressure <= patch.options.pressure_floor * 1.01
            s        = self.xp.sum(mask)
            return float(s.get() if hasattr(s, "get") else s)


    def Band_Luminosity(self, patch, return_teff=False):
        with patch.execution_context:
            ng     = self.ng
            dev_id = int(getattr(patch.execution_context, "id", 0))
            if not hasattr(self, "_EmissionTable_cache"):
                self._EmissionTable_cache = {}
            if dev_id not in self._EmissionTable_cache:
                self._EmissionTable_cache[dev_id] = [ self.xp.array(band) for band in self.setup.EmissionTable ]
                logger.info(f"Precomputed emission tables for temperature range 10^{self.xp.round(self.xp.log10(self.setup.Temperature[0]))} K to 10^{self.xp.round(self.xp.log10(self.setup.Temperature[-1]))} K")
            

            # ============ We need to do remapping for diagnostics ============
            Mdrop           = self.setup.SS73.Mdrop
            Sigma           = self.xp.maximum(patch.primitive[ng:-ng, ng:-ng, 0] * Mdrop**(3./5.), 1e-30)
            Pressure        = patch.primitive[ng:-ng, ng:-ng, 3] * Mdrop          
            Precomputed_T   = self.setup.Temperature
            T               = self.xp.maximum((Pressure / Sigma) * (self.setup.SS73.mp_code / self.setup.SS73.kb_code), Precomputed_T[0])
            m1, m2          = patch.physics.point_masses(patch.time)
            X               = patch.coordinate_array_x
            Y               = patch.coordinate_array_y
            R_1, R_2        = self.xp.sqrt((X-m1.position_x)**2 + (Y-m1.position_y)**2), self.xp.sqrt((X-m2.position_x)**2  + (Y-m2.position_y)**2)
            cs              = (patch.physics.gamma_law_index * Pressure / Sigma)**0.5
            omega           = self.xp.sqrt(m1.mass / (R_1**3 + 1e-12) + m2.mass / (R_2**3 + 1e-12))
            H               = cs / omega
            rho             = Sigma / (2 * H)

            # ============ absorption, scattering and effective optical depths ============
            Z                    = 1.0
            gaunt_r              = 1.0
            alpha_ff             = self.setup.SS73.ff_absorption_code * T **(-7/2) * Z**2 * rho**2 * gaunt_r 
            tau_ff               = alpha_ff * H 
            tau_es               = Sigma    * self.setup.SS73.kappa_code
            tau_effective        = self.xp.sqrt(tau_ff * (tau_ff + tau_es))
            tau                  = tau_es + tau_ff
            Teff                 = EffectiveTemperature(tau, T)
            BolometricLuminosity = 2 * cgs['sigmab'] * Teff ** 4 # report emission in cgs

            if not patch.options.sink_emission:
                r1_mask = ((X-m1.position_x)**2 + (Y-m1.position_y)**2) > m1.sink_radius**2
                r2_mask = ((X-m2.position_x)**2 + (Y-m2.position_y)**2) > m2.sink_radius**2
            else:
                r1_mask = 1
                r2_mask = 1

            tau_mask    = (tau_effective >= self.setup.OpticalDepthFloor)
            floor_mask  = (Sigma >= patch.options.density_floor * Mdrop**(3./5.)) & (Pressure >= patch.options.pressure_floor * Mdrop)
            mask        = r1_mask & r2_mask & tau_mask & floor_mask
            #mask        = mask.astype(float)
            
            dlogT         = float(np.diff(np.log10(Precomputed_T))[0])
            T0            = float(Precomputed_T[0])
            Teff_clamped  = self.xp.maximum(Teff, T0)
            interpolate   = self.xp.log10(Teff_clamped / T0) / dlogT
            N0            = self.xp.floor(interpolate).astype(int)
            Bracket_N0_N1 = interpolate - N0
            
            EmissionTable = self._EmissionTable_cache[dev_id]
            Optical_N0    = self.xp.take(EmissionTable[0],N0, axis=0)
            Optical_N1    = self.xp.take(EmissionTable[0],N0+1,axis=0)
            Infared_N0    = self.xp.take(EmissionTable[1],N0  ,axis=0)
            Infared_N1    = self.xp.take(EmissionTable[1],N0+1,axis=0)
            UV_N0         = self.xp.take(EmissionTable[2],N0  ,axis=0)
            UV_N1         = self.xp.take(EmissionTable[2],N0+1,axis=0)
            Xray_N0       = self.xp.take(EmissionTable[3],N0  ,axis=0)
            Xray_N1       = self.xp.take(EmissionTable[3],N0+1,axis=0)

            Interpolated_Optical = interpolate_band(Optical_N0, Optical_N1, Bracket_N0_N1)
            Interpolated_Infared = interpolate_band(Infared_N0, Infared_N1, Bracket_N0_N1)
            Interpolated_UV      = interpolate_band(UV_N0     , UV_N1     , Bracket_N0_N1)
            Interpolated_Xray    = interpolate_band(Xray_N0   , Xray_N1   , Bracket_N0_N1)

            Interpolated_Optical *= (mask * self.setup.SS73.Length_Scale_CGS**2)
            Interpolated_Infared *= (mask * self.setup.SS73.Length_Scale_CGS**2)
            Interpolated_UV      *= (mask * self.setup.SS73.Length_Scale_CGS**2)
            Interpolated_Xray    *= (mask * self.setup.SS73.Length_Scale_CGS**2)
            BolometricLuminosity *= (mask * self.setup.SS73.Length_Scale_CGS**2)

            if return_teff:
                Teff_cpu = Teff.get() if hasattr(Teff, "get") else np.array(Teff)
                mask_cpu = mask.get() if hasattr(mask, "get") else np.array(mask)
                return 2*Interpolated_Infared, 2*Interpolated_Optical, 2*Interpolated_UV, 2*Interpolated_Xray, 2*BolometricLuminosity, self.xp.sum(~mask), self.xp.max(Teff*mask), Teff_cpu, mask_cpu

            return 2*Interpolated_Infared, 2*Interpolated_Optical, 2*Interpolated_UV, 2*Interpolated_Xray, 2*BolometricLuminosity, self.xp.sum(~mask), self.xp.max(Teff*mask)

    def timeseries_sed(self, freq_space):
        from sailfish.physics.cooling import PlanckSpectrum
        spectrum = np.zeros(len(freq_space))
        area     = (self.mesh.dx * self.setup.SS73.Length_Scale_CGS) ** 2
        f_col    = freq_space[:, np.newaxis]
        for patch in self.patches:
            Teff_cpu, mask_cpu = self._teff_cache[patch]
            Teff_masked = np.where(mask_cpu, Teff_cpu, 1.0)
            T_flat  = Teff_masked.ravel()[np.newaxis, :]
            spectrum += PlanckSpectrum(f_col, T_flat).sum(axis=1) * area * 2
        return spectrum

    def reductions(self):
        """
        Generate runtime reductions on the solution data for time series.
        Fully GPU-parallelized version with all diagnostics preserved.
        """

        xp = self.xp
        da = self.mesh.dx * self.mesh.dy
        ng = self.ng
        diagnostics = self._physics.diagnostics
        gpu_results = []

        # Helper to make sure we can always stack safely
        def to_gpu_array(x):
            if isinstance(x, (float, int)):
                return xp.array(x)
            return xp.asarray(x)
        

        patch_idx = {p: i for i, p in enumerate(self.patches)}

        udots_cache = {}
        def get_udots(which_mass, term):
            key = (which_mass, term)
            if key not in udots_cache:
                if term == "acc":
                    udots_cache[key] = [p.point_mass_source_term(which_mass, accretion=True) for p in self.patches]
                elif term == "grv":
                    udots_cache[key] = [p.point_mass_source_term(which_mass, gravity=True) for p in self.patches]
                elif term == "buf":
                    udots_cache[key] = [p.buffer_source_term() for p in self.patches]
                else:
                    raise ValueError("Invalid source term")
            return udots_cache[key]

        # Precompute emission results once per patch, caching Teff if SED recording is on
        if getattr(self.setup, 'record_sed_timeseries', False):
            _full        = {p: self.Band_Luminosity(p, return_teff=True) for p in self.patches}
            band_cache   = {p: v[:-2] for p, v in _full.items()}
            self._teff_cache = {p: v[-2:] for p, v in _full.items()}
        else:
            band_cache       = {p: self.Band_Luminosity(p) for p in self.patches}
            self._teff_cache = {}

        import sailfish.physics.kepler as kepler
        m1, m2 = self._physics.point_masses(self.time)
        m1 = kepler.PointMass(m1.mass, m1.position_x, m1.position_y, m1.velocity_x, m1.velocity_y)
        m2 = kepler.PointMass(m2.mass, m2.position_x, m2.position_y, m2.velocity_x, m2.velocity_y)
        orbital_state = kepler.OrbitalState(primary=m1, secondary=m2)

        # Utility for hydrodynamic quantities 
        def get_field(patch, quantity, cut, mass, gravity=False, accretion=False, buffer=False):
            x, y, r = patch.coordinate_array_x, patch.coordinate_array_y, patch.r

            def apply_radial_cut(f):
                if cut is not None:
                    r0, r1 = cut
                    return f * (r0 < r) * (r < r1)
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

            if quantity == "total_angular_momentum":
                sigma = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 0])
                vx = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 1])
                vy = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 2])
                return sigma * (x * vy - y * vx)

            # generalise this to mass 0, 1 and 2? ie. outer buffer and two inner buffers?
            if quantity == "buffer_torque":
                fx = get_field(patch, 1, cut=(self.buffer_onset_radius, self.domain_radius), mass=0, gravity=False, accretion=False, buffer=True)
                fy = get_field(patch, 2, cut=(self.buffer_onset_radius, self.domain_radius), mass=0, gravity=False, accretion=False, buffer=True)
                return x * fy - y * fx

            if quantity == "buffer_torque_dynamical":
                # Buffer torque excluding angular momentum advection from mass source
                fx   = get_field(patch, 1, cut=(self.buffer_onset_radius, self.domain_radius), mass=0, gravity=False, accretion=False, buffer=True)
                fy   = get_field(patch, 2, cut=(self.buffer_onset_radius, self.domain_radius), mass=0, gravity=False, accretion=False, buffer=True)
                mdot = get_field(patch, 0, cut=(self.buffer_onset_radius, self.domain_radius), mass=0, gravity=False, accretion=False, buffer=True)
                
                # Angular momentum per unit mass at each location
                GM         = 1.0
                omega      = self.xp.sqrt(GM / (r + 1e-12)**3)
                l_specific = r * r * omega
                
                # Subtract angular momentum carried by mass source
                # For retrograde: omega and velocities are negative
                sign         = -1.0 if patch.physics.retrograde else 1.0
                fx_corrected = fx - sign * mdot * (-y / (r**2 + 1e-12)) * l_specific
                fy_corrected = fy - sign * mdot * (+x / (r**2 + 1e-12)) * l_specific
                
                return x * fy_corrected - y * fx_corrected

            if quantity == "buffer_mass_rate":
                return get_field(patch, 0, cut=(self.buffer_onset_radius, self.domain_radius), mass=0, gravity=False,  accretion=False, buffer=True)

            if quantity == "radial_mass_flux":
                r0, r1 = cut if cut is not None else (0.0, 1e10)
                sigma = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 0])
                vx    = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 1])
                vy    = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 2])
                vr    = (vx * x + vy * y) / (r + 1e-12)
                return sigma * vr / (r1 - r0)

            if quantity == "angular_momentum_flux":
                r0, r1 = cut if cut is not None else (0.0, 1e10)
                pres = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 3])
                vx   = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 1])
                vy   = apply_radial_cut(patch.primitive[ng:-ng, ng:-ng, 2])
                vphi = (-y * vx + x * vy) / (r + 1e-12)
                return (1.5 * self._physics.alpha * self._physics.gamma_law_index) * pres * r**1.5 * vphi / (r1 - r0)

            if quantity == "power":
                fx = get_field(patch, 1, cut, mass, gravity, accretion, buffer)
                fy = get_field(patch, 2, cut, mass, gravity, accretion, buffer)
                m1, m2 = self._physics.point_masses(self.time)
                if mass == 1:
                    vx1, vy1 = m1.velocity_x, m1.velocity_y
                    return vx1 * fx + vy1 * fy
                elif mass == 2:
                    vx2, vy2 = m2.velocity_x, m2.velocity_y
                    return vx2 * fx + vy2 * fy
                else:
                    raise ValueError("Mass option for 'power' must be 1 or 2.")

            if quantity == "Accreted_energy":
                return get_field(patch, 3, cut, mass="both", gravity=False, accretion=True, buffer=False)

            q = quantity
            i = patch_idx[patch]

            if accretion:
                udots1 = get_udots(1, "acc")
                udots2 = get_udots(2, "acc")
            elif gravity:
                udots1 = get_udots(1, "grv")
                udots2 = get_udots(2, "grv")

            if mass == "both":
                f = udots1[i][..., q] + udots2[i][..., q]
            elif mass == 1:
                f = udots1[i][..., q]
            elif mass == 2:
                f = udots2[i][..., q]
            elif mass == 0:
                udots = get_udots(0, "buf")
                f = udots[i][..., q]
            else:
                raise ValueError("Invalid mass specifier")

            return apply_radial_cut(f)

        for d in diagnostics:
            q = d.quantity

            # --- Orbital quantities ---
            if q == "time":
                gpu_results.append(xp.array(self.time / self.setup.reference_time_scale))
                continue
            elif q == "semimajor-axis":
                gpu_results.append(xp.array(orbital_state.semimajor_axis))
                continue
            elif q == "eccentricity":
                gpu_results.append(xp.array(orbital_state.eccentricity))
                continue

            # --- Floor diagnostics ---
            elif q == "density_floor":
                gpu_results.append(xp.array(sum(self.detect_density_floor(p) for p in self.patches)))
                continue
            elif q == "pressure_floor":
                gpu_results.append(xp.array(sum(self.detect_pressure_floor(p) for p in self.patches)))
                continue

            # --- Radiative luminosities ---
            if q in ("infared", "optical", "uv", "xray", "bolometric"):
                idx = dict(infared=0, optical=1, uv=2, xray=3, bolometric=4)[q]
                gpu_sum = xp.sum(xp.stack([band_cache[p][idx] for p in self.patches]))
                gpu_results.append(gpu_sum * da)
                continue

            elif q == "uncounted_cells_in_lc":
                vals = [to_gpu_array(band_cache[p][5]) for p in self.patches]
                gpu_sum = xp.sum(xp.stack(vals))
                gpu_results.append(gpu_sum)
                continue

            elif q == "max_temperature":
                vals = [to_gpu_array(band_cache[p][6]) for p in self.patches]
                gpu_max = xp.max(xp.stack(vals))
                gpu_results.append(gpu_max)
                continue

            # --- Hydrodynamic quantities ---
            field_sums = []
            for p in self.patches:
                with p.execution_context:
                    f = get_field(
                        p,
                        d.quantity,
                        d.radial_cut,
                        d.which_mass,
                        gravity=d.gravity,
                        accretion=d.accretion,
                    )
                    field_sums.append(xp.sum(f))
            gpu_sum = xp.sum(xp.stack([to_gpu_array(f) for f in field_sums]))
            gpu_results.append(gpu_sum * da)

        gpu_results = xp.stack([xp.asarray(r) for r in gpu_results])
        host_results = gpu_results.get() if hasattr(gpu_results, "get") else gpu_results
        return host_results.tolist()

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
            # advance timestep
            patch.advance_rk(rk_param, dt)



    def set_bc(self, array):
        ng = self.ng
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
        ng = self.ng

        with self.patches[patch_index].execution_context:
            xp = self.xp

            # --- copy left boundary from left neighbor via host ---
            left_host  = to_host(pl[-2 * ng : -ng])    # numpy array
            right_host = to_host(pr[+ng : +2 * ng])    # numpy array

            left  = xp.asarray(left_host)   # now on this patch's device
            right = xp.asarray(right_host)

            # 1. internal BC in i-direction using the re-uploaded slices
            pc[:+ng] = left
            pc[-ng:] = right

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
