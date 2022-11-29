from collections.abc import MutableMapping
from collections import ChainMap
from loguru import logger
from numpy import linspace, meshgrid, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import kernel, perf_time_sequence, configure_kernel_module, device
from lib_euler import prim_to_cons, cons_to_prim, riemann
from configuration import configurable, all_schemas


@device()
def plm_minmod(yl: float, yc: float, yr: float, plm_theta: float):
    R"""
    #define min2(a, b) ((a) < (b) ? (a) : (b))
    #define min3(a, b, c) min2(a, min2(b, c))
    #define sign(x) copysign(1.0, x)
    #define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

    DEVICE double plm_minmod(
        double yl,
        double yc,
        double yr,
        double plm_theta)
    {
        double a = (yc - yl) * plm_theta;
        double b = (yr - yl) * 0.5;
        double c = (yr - yc) * plm_theta;
        return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
    }
    """


@kernel(device_funcs=[cons_to_prim], define_macros=dict(DIM=1))
def cons_to_prim_array(u: NDArray[float], p: NDArray[float], ni: int = None):
    R"""
    //
    // Convert an array of conserved data to an array of primitive data
    //
    KERNEL void cons_to_prim_array(double *u, double *p, int ni)
    {
        FOR_RANGE_1D(1, ni - 1)
        {
            cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
        }
    }
    """
    return u.size // 3, (u, p, u.size // 3)


@kernel(device_funcs=[prim_to_cons], define_macros=dict(DIM=1))
def prim_to_cons_array(p: NDArray[float], u: NDArray[float], ni: int = None):
    R"""
    //
    // Convert an array of primitive data to an array of conserved data
    //
    KERNEL void prim_to_cons_array(double *p, double *u, int ni)
    {
        FOR_RANGE_1D(1, ni - 1)
        {
            prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
        }
    }
    """
    return p.size // 3, (p, u, p.size // 3)


@kernel()
def conservative_update(
    u: NDArray[float],
    f: NDArray[float],
    dt: float,
    dx: float,
    ni: int = None,
):
    R"""
    //
    // Conservative difference an array of fluxes to update an array of conserved
    // densities.
    //
    // Does not modify the first-2 or final-2 zones in the conserved variable
    // array.
    //
    KERNEL void conservative_update(
        double *u,
        double *f,
        double dt,
        double dx,
        int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
        {
            double *uc = &u[3 * (i + 0)];
            double *fm = &f[3 * (i + 0)];
            double *fp = &f[3 * (i + 1)];

            for (int q = 0; q < 3; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
            }
        }
    }
    """
    return u.size // 3, (u, f, dt, dx, u.size // 3)


@kernel()
def conservative_update_rk(
    u: NDArray[float],
    f: NDArray[float],
    urk: NDArray[float],
    dt: float,
    dx: float,
    rk: float,
    ni: int = None,
):
    R"""
    //
    // Conservative difference an array of fluxes to update an array of conserved
    // densities.
    //
    // Does not modify the first-2 or final-2 zones in the conserved variable
    // array.
    //
    KERNEL void conservative_update_rk(
        double *u,
        double *f,
        double *urk,
        double dt,
        double dx,
        double rk,
        int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
        {
            double *uc = &u[3 * (i + 0)];
            double *fm = &f[3 * (i + 0)];
            double *fp = &f[3 * (i + 1)];

            for (int q = 0; q < 3; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
                uc[q] *= (1.0 - rk);
                uc[q] += rk * urk[3 * i + q];
            }
        }
    }
    """
    return u.size // 3, (u, f, urk, dt, dx, rk, u.size // 3)


@kernel(device_funcs=[prim_to_cons, cons_to_prim, riemann], define_macros=dict(DIM=1))
def update_prim_fwd_pcm(
    prd: NDArray[float],
    pwr: NDArray[float],
    dt: float,
    dx: float,
    ni: int = None,
):
    R"""
    //
    // A single-step first-order update using flux-per-zone.
    //
    // The first-2 and final-2 elements of the primitive array are not modified.
    //
    KERNEL void update_prim_fwd_pcm(double *prd, double *pwr, double dt, double dx, int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
        {
            double uc[NCONS];
            double fm[NCONS];
            double fp[NCONS];

            double *pl = &prd[NCONS * (i - 1)];
            double *pc = &prd[NCONS * (i + 0)];
            double *pr = &prd[NCONS * (i + 1)];

            prim_to_cons(pc, uc);
            riemann(pl, pc, fm, 1);
            riemann(pc, pr, fp, 1);

            for (int q = 0; q < NCONS; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
            }
            cons_to_prim(uc, &pwr[NCONS * i]);
        }
    }
    """
    return prd.shape[0], (prd, pwr, dt, dx, prd.shape[0])


@kernel(device_funcs=[prim_to_cons, cons_to_prim, riemann], define_macros=dict(DIM=1))
def update_prim_rkn_pcm(
    prd: NDArray[float],  # read-from primitive
    pwr: NDArray[float],  # write-to-primitive
    urk: NDArray[float],  # u at time-level n
    dt: float,  # time step dt
    dx: float,  # grid spacing dx
    rk: float,  # RK parameter
    ni: int = None,  # number of grid zones (includes guard)
):
    R"""
    //
    // A single-step first-order update using flux-per-zone.
    //
    // The first-2 and final-2 elements of the primitive array are not modified.
    //
    KERNEL void update_prim_rkn_pcm(double *prd, double *pwr, double *urk, double dt, double dx, double rk, int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
        {
            double uc[NCONS];
            double fm[NCONS];
            double fp[NCONS];

            double *pl = &prd[NCONS * (i - 1)];
            double *pc = &prd[NCONS * (i + 0)];
            double *pr = &prd[NCONS * (i + 1)];

            prim_to_cons(pc, uc);
            riemann(pl, pc, fm, 1);
            riemann(pc, pr, fp, 1);

            for (int q = 0; q < NCONS; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
                uc[q] *= (1.0 - rk);
                uc[q] += rk * urk[NCONS * i + q];
            }
            cons_to_prim(uc, &pwr[NCONS * i]);
        }
    }
    """
    return prd.shape[0], (prd, pwr, urk, dt, dx, rk, prd.shape[0])


@kernel(
    device_funcs=[prim_to_cons, cons_to_prim, riemann, plm_minmod],
    define_macros=dict(DIM=1),
)
def update_prim_fwd_plm(
    prd: NDArray[float],
    pwr: NDArray[float],
    dt: float,
    dx: float,
    plm_theta: float,
    ni: int = None,
):
    R"""
    //
    // A second-order-in-space update using flux-per-zone and PLM.
    //
    // The first-2 and final-2 elements of the primitive array are not modified.
    //
    KERNEL void update_prim_fwd_plm(double *prd, double *pwr, double dt, double dx, double plm_theta, int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
        {
            double uc[NCONS];
            double plp[NCONS];
            double pcm[NCONS];
            double pcp[NCONS];
            double prm[NCONS];
            double fm[NCONS];
            double fp[NCONS];

            double *pk = &prd[NCONS * (i - 2)];
            double *pl = &prd[NCONS * (i - 1)];
            double *pc = &prd[NCONS * (i + 0)];
            double *pr = &prd[NCONS * (i + 1)];
            double *ps = &prd[NCONS * (i + 2)];

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = plm_minmod(pk[q], pl[q], pc[q], plm_theta);
                double gc = plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                double gr = plm_minmod(pc[q], pr[q], ps[q], plm_theta);

                plp[q] = pl[q] + 0.5 * gl;
                pcm[q] = pc[q] - 0.5 * gc;
                pcp[q] = pc[q] + 0.5 * gc;
                prm[q] = pr[q] - 0.5 * gr;
            }
            riemann(plp, pcm, fm, 1);
            riemann(pcp, prm, fp, 1);
            prim_to_cons(pc, uc);

            for (int q = 0; q < NCONS; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
            }
            cons_to_prim(uc, &pwr[NCONS * i]);
        }
    }
    """
    return prd.shape[0], (prd, pwr, dt, dx, plm_theta, prd.shape[0])


@kernel(
    device_funcs=[prim_to_cons, cons_to_prim, riemann, plm_minmod],
    define_macros=dict(DIM=1),
)
def update_prim_rkn_plm(
    prd: NDArray[float],  # read-from primitive
    pwr: NDArray[float],  # write-to-primitive
    urk: NDArray[float],  # u at time-level n
    dt: float,  # time step dt
    dx: float,  # grid spacing dx
    rk: float,  # RK parameter
    plm_theta: float,
    ni: int = None,
):
    R"""
    //
    // A second-order-in-space update using flux-per-zone and PLM.
    //
    // The first-2 and final-2 elements of the primitive array are not modified.
    //
    KERNEL void update_prim_rkn_plm(
        double *prd,
        double *pwr,
        double *urk,
        double dt,
        double dx,
        double rk,
        double plm_theta,
        int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
        {
            double uc[NCONS];
            double plp[NCONS];
            double pcm[NCONS];
            double pcp[NCONS];
            double prm[NCONS];
            double fm[NCONS];
            double fp[NCONS];

            double *pk = &prd[NCONS * (i - 2)];
            double *pl = &prd[NCONS * (i - 1)];
            double *pc = &prd[NCONS * (i + 0)];
            double *pr = &prd[NCONS * (i + 1)];
            double *ps = &prd[NCONS * (i + 2)];

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = plm_minmod(pk[q], pl[q], pc[q], plm_theta);
                double gc = plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                double gr = plm_minmod(pc[q], pr[q], ps[q], plm_theta);

                plp[q] = pl[q] + 0.5 * gl;
                pcm[q] = pc[q] - 0.5 * gc;
                pcp[q] = pc[q] + 0.5 * gc;
                prm[q] = pr[q] - 0.5 * gr;
            }
            riemann(plp, pcm, fm, 1);
            riemann(pcp, prm, fp, 1);
            prim_to_cons(pc, uc);

            for (int q = 0; q < NCONS; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
                uc[q] *= (1.0 - rk);
                uc[q] += rk * urk[NCONS * i + q];
            }
            cons_to_prim(uc, &pwr[NCONS * i]);
        }
    }
    """
    return prd.shape[0], (prd, pwr, urk, dt, dx, rk, plm_theta, prd.shape[0])


@kernel(device_funcs=[riemann], define_macros=dict(DIM=1))
def compute_godunov_fluxes_pcm(p: NDArray[float], f: NDArray[float], ni: int = None):
    R"""
    //
    // Compute an array of Godunov fluxes using HLLE Riemann solver.
    //
    // The first and final elements of the flux array are not modified.
    //
    KERNEL void compute_godunov_fluxes_pcm(double *p, double *f, int ni)
    {
        FOR_RANGE_1D(0, ni - 1)
        {
            double *pc = &p[NCONS * (i + 0)];
            double *pr = &p[NCONS * (i + 1)];
            double *fp = &f[NCONS * (i + 1)];

            riemann(pc, pr, fp, 1);
        }
    }
    """
    return p.shape[0], (p, f, p.shape[0])


@kernel(device_funcs=[riemann, plm_minmod], define_macros=dict(DIM=1))
def compute_godunov_fluxes_plm(
    p: NDArray[float],
    f: NDArray[float],
    plm_theta: float,
    ni: int = None,
):
    R"""
    //
    // Compute an array of Godunov fluxes using HLLE Riemann solver and PLM
    // reconstruction.
    //
    // The first-1 and final-2 elements of the flux array are not modified.
    //
    KERNEL void compute_godunov_fluxes_plm(double *p, double *f, double plm_theta, int ni)
    {
        FOR_RANGE_1D(1, ni - 2)
        {
            double pm[NCONS];
            double pp[NCONS];

            double *pl = &p[NCONS * (i - 1)];
            double *pc = &p[NCONS * (i + 0)];
            double *pr = &p[NCONS * (i + 1)];
            double *ps = &p[NCONS * (i + 2)];
            double *fh = &f[NCONS * (i + 1)];

            for (int q = 0; q < NCONS; ++q)
            {
                pm[q] = pc[q] + 0.5 * plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                pp[q] = pr[q] - 0.5 * plm_minmod(pc[q], pr[q], ps[q], plm_theta);
            }
            riemann(pm, pp, fh, 1);
        }
    }
    """
    return p.shape[0], (p, f, plm_theta, p.shape[0])


def update_prim_fwd(prd, pwr, dt, dx, plm_theta, reconstruction):
    if reconstruction == "pcm":
        update_prim_fwd_pcm(prd, pwr, dt, dx)
    elif reconstruction == "plm":
        update_prim_fwd_plm(prd, pwr, dt, dx, plm_theta)
    else:
        raise ValueError(f"reconstruction must be [pcm|plm], got {reconstruction}")


def update_prim_rkn(prd, pwr, urk, dt, dx, rk, plm_theta, reconstruction):
    if reconstruction == "pcm":
        update_prim_rkn_pcm(prd, pwr, urk, dt, dx, rk)
    elif reconstruction == "plm":
        update_prim_rkn_plm(prd, pwr, urk, dt, dx, rk, plm_theta)
    else:
        raise ValueError(f"reconstruction must be [pcm|plm], got {reconstruction}")


def compute_godunov_fluxes(p, f, plm_theta, reconstruction):
    if reconstruction == "pcm":
        compute_godunov_fluxes_pcm(p, f)
    elif reconstruction == "plm":
        compute_godunov_fluxes_plm(p, f, plm_theta)
    else:
        raise ValueError(f"reconstruction must be [pcm|plm], got {reconstruction}")


def update_prim(
    p,
    dt,
    dx,
    strategy="flux_per_zone",
    reconstruction="pcm",
    plm_theta=2.0,
    time_integration="fwd",
    exec_mode="cpu",
):
    """
    Drives a first-order update of a primitive array
    """
    xp = numpy_or_cupy(exec_mode)

    if time_integration == "fwd":
        pass
    elif time_integration == "rk1":
        rks = [0.0]
    elif time_integration == "rk2":
        rks = [0.0, 0.5]
    elif time_integration == "rk3":
        rks = [0.0, 3.0 / 4.0, 1.0 / 3.0]
    else:
        raise ValueError(
            f"time_integration must be [fwd|r1k|rk2|rk3], got {time_integration}"
        )

    if strategy == "flux_per_face":
        f = xp.empty_like(p)
        u = xp.empty_like(p)

        if time_integration == "fwd":
            compute_godunov_fluxes(p, f, plm_theta, reconstruction)
            prim_to_cons_array(p, u)
            conservative_update(u, f, dt, dx)
            cons_to_prim_array(u, p)
        else:
            prim_to_cons_array(p, u)
            urk = u.copy()

            for rk in rks:
                compute_godunov_fluxes(p, f, plm_theta, reconstruction)
                conservative_update_rk(u, f, urk, dt, dx, rk)
                cons_to_prim_array(u, p)
        return p

    elif strategy == "flux_per_zone":
        prd = p
        pwr = p.copy()

        if time_integration == "fwd":
            update_prim_fwd(prd, pwr, dt, dx, plm_theta, reconstruction)
            prd, pwr = pwr, prd

        else:
            urk = xp.empty_like(prd)  # does this cause bug? was zeros_like last checked
            prim_to_cons_array(prd, urk)

            # BUG HERE ... no updates with RK1 (plm or pcm)
            for rk in rks:
                update_prim_rkn(prd, pwr, urk, dt, dx, rk, plm_theta, reconstruction)
                prd, pwr = pwr, prd

        return prd
    else:
        raise ValueError(f"unknown strategy {strategy}")


def cell_centers_1d(ni):
    from numpy import linspace

    xv = linspace(0.0, 1.0, ni)
    xc = 0.5 * (xv[1:] + xv[:-1])
    return xc


def linear_shocktube(x):
    """
    A linear shocktube setup
    """

    from numpy import array, zeros, logical_not

    l = x < 0.5
    r = logical_not(l)
    p = zeros(x.shape + (3,))
    p[l, :] = [1.0, 0.0, 1.000]
    p[r, :] = [0.1, 0.0, 0.125]
    return p


def numpy_or_cupy(exec_mode):
    if exec_mode == "gpu":
        import cupy

        return cupy

    if exec_mode == "cpu":
        import numpy

        return numpy


class State:
    def __init__(self, n, t, p):
        self._n = n
        self._t = t
        self._p = p

    @property
    def iteration(self):
        return self._n

    @property
    def time(self):
        return self._t

    @property
    def primitive(self):
        try:
            return self._p.get()
        except AttributeError:
            return self._p

    @property
    def total_zones(self):
        return self._p.shape[0]


def simulation(exec_mode, resolution, strategy, reconstruction, time_integration):
    from functools import partial

    xp = numpy_or_cupy(exec_mode)
    nz = resolution
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)
    p = linear_shocktube(x)
    t = 0.0
    n = 0
    p = xp.array(p)
    iteration = 0

    advance = partial(
        update_prim,
        dx=dx,
        strategy=strategy,
        reconstruction=reconstruction,
        time_integration=time_integration,
        exec_mode=exec_mode,
    )
    yield State(n, t, p)

    while True:
        p = advance(p, dt)
        t += dt
        n += 1
        yield State(n, t, p)


@configurable
def driver(
    app_config,
    exec_mode: str = "cpu",
    resolution: int = 10000,
    tfinal: float = 0.1,
    strategy: str = "flux_per_zone",
    reconstruction: str = "pcm",
    plm_theta: float = 1.5,
    time_integration: str = "fwd",
    fold: int = 100,
    plot: bool = False,
):
    """
    Configuration
    -------------

    exec_mode:        execution mode [cpu|gpu]
    resolution:       number of grid zones
    tfinal:           time to end the simulation
    strategy:         solver strategy [flux_per_zone|flux_per_face]
    reconstruction:   first or second-order reconstruction [pcm|plm]
    plm_theta:        PLM parameter [1.0, 2.0]
    time_integration: Runge-Kutta order [fwd|rk1|rk2|rk3]
    fold:             number of iterations between iteration message
    plot:             whether to show a plot of the solution
    """
    from reporting import terminal, iteration_msg

    configure_kernel_module(default_exec_mode=exec_mode)
    term = terminal(logger)

    logger.info("start simulation")
    perf_timer = perf_time_sequence(mode=exec_mode)

    for state in simulation(
        exec_mode, resolution, strategy, reconstruction, time_integration
    ):

        # if tasks.checkpoint.is_due(state.time):
        #     write_checkpoint(state, timeseries, tasks)

        # if tasks.timeseries.is_due(state.time):
        #     for name, info in diagnostics.items():
        #         timeseries[name].append(state.diagnostic(info))

        if state.time >= tfinal:
            break

        if state.iteration % fold == 0:
            zps = state.total_zones / next(perf_timer) * fold
            term(iteration_msg(state.iteration, state.time, zps=zps))

    if plot:
        from matplotlib import pyplot as plt

        plt.plot(state.primitive[:, 0], "-o", mfc="none", label=strategy)
        plt.show()


def flatten_dict(
    d: MutableMapping,
    parent_key: str = "",
    sep: str = ".",
) -> MutableMapping:
    """
    Create a flattened dictionary e from d, with e['a.b.c'] = d['a']['b']['c'].
    """
    items = list()
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_section(d: MutableMapping, section: str):
    """
    From a map with keys like section.b.c, return a dict with keys like b.c.
    """
    return {k[k.index(".") + 1 :]: v for k, v in d.items() if k.startswith(section)}


def load_config(config):
    """
    Attempt to load configuration data from a file: either JSON or YAML.
    """
    if config.endswith(".json"):
        from json import load

        with open(config, "r") as infile:
            return load(infile)

    elif config.endswith(".yaml"):
        from yaml import load, CLoader

        with open(config, "r") as infile:
            return load(infile, Loader=CLoader)

    else:
        raise ValueError(f"unknown configuration file {config}")


def short_help(args):
    args.parser.print_usage()


def run(args):
    app_config = ChainMap(
        {k: v for k, v in vars(args).items() if v is not None and "." in k}
    )
    app_config.maps.extend(flatten_dict(load_config(c)) for c in reversed(args.configs))
    driver_args = dict_section(app_config, "driver")
    driver.schema.validate(**driver_args)
    driver.schema.print_schema(
        args.term,
        config=driver_args,
        newline=True,
    )
    driver(app_config, **driver_args)


def show_config(args):
    if args.defaults:
        for schema in all_schemas():
            schema.print_schema(args.term)

    else:
        app_cfg = {s.component_name: s.defaults_dict() for s in all_schemas()}

        if args.format == "json":
            from json import dumps

            print(dumps(app_cfg, indent=4))

        if args.format == "yaml":
            try:
                from yaml import dump, CDumper

                print(dump(app_cfg, Dumper=CDumper))

            except ImportError as e:
                print(e)


@logger.catch
def main():
    from argparse import ArgumentParser
    from reporting import add_logging_arguments, terminal, configure_logger

    parser = ArgumentParser()
    parser.set_defaults(func=short_help)
    parser.set_defaults(term=terminal(logger))
    parser.set_defaults(parser=parser)
    parser.set_defaults(log_level="info")
    subparsers = parser.add_subparsers()

    show_config_parser = subparsers.add_parser(
        "show-config",
        help="show global configuration data",
    )
    show_config_parser.set_defaults(func=show_config)
    group = show_config_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "yaml"],
        help="output format for the configuration data",
    )
    group.add_argument(
        "--defaults",
        action="store_true",
        help="print defaults and help messages for configurable components",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="run a simulation",
    )
    run_parser.set_defaults(func=run)
    run_parser.add_argument(
        "configs",
        nargs="*",
        help="sequence of presets, configuration files, or checkpoints",
    )
    driver.schema.argument_parser(run_parser, dest_prefix="driver")
    add_logging_arguments(run_parser)

    args = parser.parse_args()
    configure_logger(logger, log_level=args.log_level)

    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.success("ctrl-c interrupt")
