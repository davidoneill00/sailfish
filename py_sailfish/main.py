#!/usr/bin/env python3


class ConfigurationError(Exception):
    """An invalid runtime configuration"""


class RecurringTask:
    def __init__(self, name_or_dict):
        if type(name_or_dict) is dict:
            self.name = name_or_dict["name"]
            self.interval = name_or_dict["interval"]
            self.last_time = name_or_dict["last_time"]
            self.number = name_or_dict["number"]
        else:
            self.name = name_or_dict
            self.interval = None
            self.last_time = None
            self.number = 0

    def next_time(self, time):
        if self.last_time is None:
            return time
        else:
            return self.last_time + self.interval

    def is_due(self, time):
        return time >= self.next_time(time)

    def next(self, time):
        self.last_time = self.next_time(time)
        self.number += 1


def first_rest(a):
    if len(a) > 1:
        return a[0], a[1:]
    else:
        return a[0], []


def update_namespace(new_args, old_args, frozen=[]):
    for key in vars(new_args):
        new_val = getattr(new_args, key)
        old_val = getattr(old_args, key)
        if old_val is not None:
            if new_val is None:
                setattr(new_args, key, old_val)
            elif key in frozen and new_val != old_val:
                raise ConfigurationError(f"{key} cannot be changed")


def parse_parameters(item_list):
    parameters = dict()
    for item in item_list:
        try:
            key, val = item.split("=")
            parameters[key] = eval(val)
        except NameError:
            raise ConfigurationError(f"badly formed model parameter {item}")
    return parameters


def write_checkpoint(number, logger=None, **kwargs):
    import pickle

    with open(f"chkpt.{number:04d}.pk", "wb") as chkpt:
        if logger is not None:
            logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(kwargs, chkpt)


def load_checkpoint(chkpt_name):
    import pickle

    try:
        with open(chkpt_name, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise ConfigurationError(f"could not open checkpoint file {chkpt_name}")


def initial_condition(setup, num_zones, domain):
    import math
    import numpy as np

    xcells = np.linspace(domain[0], domain[1], num_zones)
    primitive = np.zeros([num_zones, 4])

    for x, p in zip(xcells, primitive):
        setup.initial_primitive(x, p)

    return primitive


def main(args):
    from time import perf_counter
    from logging import getLogger
    from sailfish.solvers import srhd_1d
    from sailfish.setup import Setup
    from sailfish import system

    logger = getLogger("driver")
    setup_or_checkpoint, parameter_list = first_rest(args.command.split(":"))

    if setup_or_checkpoint.endswith(".pk"):
        chkpt_name = setup_or_checkpoint
        logger.info(f"load checkpoint {chkpt_name}")

        chkpt = load_checkpoint(chkpt_name)
        chkpt["parameters"].update(parse_parameters(parameter_list))
        update_namespace(args, chkpt["args"], frozen=["resolution"])

        setup_name = chkpt["setup_name"]
        parameters = chkpt["parameters"]
        setup = Setup.find_setup(setup_name)(**parameters)
        iteration = chkpt["iteration"]
        time = chkpt["time"]
        checkpoint_task = RecurringTask(chkpt["tasks"]["checkpoint"])
        initial = chkpt["primitive"]

    else:
        if args.resolution is None:
            args.resolution = 10000

        logger.info(f"generate initial data for setup {setup_or_checkpoint}")

        setup_name = setup_or_checkpoint
        parameters = parse_parameters(parameter_list)
        setup = Setup.find_setup(setup_name)(**parameters)
        iteration = 0
        time = 0.0
        checkpoint_task = RecurringTask("checkpoint")
        initial = initial_condition(setup, args.resolution, setup.domain)

    num_zones = args.resolution
    mode = args.mode or "cpu"
    fold = args.fold or 10
    cfl_number = 0.6
    dt = 1.0 / num_zones * cfl_number
    checkpoint_task.interval = args.checkpoint or 0.1

    system.log_system_info(mode)
    system.configure_build()

    solver = srhd_1d.Solver(
        initial=initial,
        time=time,
        domain=setup.domain,
        num_patches=4,
        boundary_condition=setup.boundary_condition,
        mode=mode,
    )
    logger.info("start simulation")

    end_time = args.end_time if args.end_time is not None else 0.4

    def checkpoint():
        checkpoint_task.next(solver.time)
        write_checkpoint(
            checkpoint_task.number - 1,
            logger=logger,
            time=solver.time,
            iteration=iteration,
            primitive=solver.primitive,
            args=args,
            tasks=dict(checkpoint=vars(checkpoint_task)),
            parameters=parameters,
            setup_name=setup_name,
            domain=setup.domain,
        )

    while end_time is None or end_time > solver.time:
        if checkpoint_task.is_due(solver.time):
            checkpoint()

        start = perf_counter()
        for _ in range(fold):
            solver.new_timestep()
            solver.advance_rk(0.0, dt)
            solver.advance_rk(0.5, dt)
            iteration += 1
        stop = perf_counter()
        Mzps = num_zones / (stop - start) * 1e-6 * fold

        print(f"[{iteration:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    checkpoint()


if __name__ == "__main__":
    import argparse
    import logging
    import textwrap
    from sailfish.setup import Setup, SetupError

    logging.basicConfig(level=logging.INFO, format="-> %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="sailfish",
        usage="%(prog)s <command> [options]",
        description="gpu-accelerated astrophysical gasdynamics code",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="setup name or restart file",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print a description of the setup and exit",
    )
    parser.add_argument(
        "--resolution",
        "-n",
        metavar="N",
        type=int,
        help="grid resolution",
    )
    parser.add_argument(
        "--fold",
        "-f",
        metavar="F",
        type=int,
        help="iterations between messages and side effects",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        metavar="C",
        type=float,
        help="how often to write a checkpoint file",
    )
    parser.add_argument(
        "--end-time",
        "-e",
        metavar="E",
        type=float,
        help="when to end the simulation",
    )
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument(
        "--mode",
        help="execution mode",
        choices=["cpu", "omp", "gpu"],
    )
    exec_group.add_argument(
        "--use-omp",
        "-p",
        dest="mode",
        action="store_const",
        const="omp",
        help="multi-core with OpenMP",
    )
    exec_group.add_argument(
        "--use-gpu",
        "-g",
        dest="mode",
        action="store_const",
        const="gpu",
        help="gpu acceleration",
    )

    try:
        args = parser.parse_args()

        if args.describe and args.command is not None:
            setup_name = args.command
            setup = Setup.find_setup(setup_name)
            print(f"setup: {setup_name}")
            print()
            print(textwrap.dedent(setup.__doc__).strip())
            print()
            print("model parameters:")
            for name, default, about in setup.model_parameters():
                print(f"{name:.<16s} {default:<5} {about}")
            print()

        elif args.command is None:
            print("specify setup:")
            for setup in Setup.__subclasses__():
                print(f"    {setup.dash_case_class_name()}")

        else:
            main(args)

    except ConfigurationError as e:
        print(f"bad configuration: {e}")

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except SetupError as e:
        print(f"error: {e}")

    except KeyboardInterrupt:
        print("")
