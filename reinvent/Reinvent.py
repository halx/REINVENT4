#!/usr/bin/env python
"""Main entry point into Reinvent."""

from __future__ import annotations
import logging
import os
import sys

from dotenv import load_dotenv, find_dotenv
import platform
import getpass
import datetime
from typing import Any

from rdkit import rdBase
import rdkit
import torch

import pydantic

from reinvent.utils import (
    parse_command_line,
    set_seed,
    extract_sections,
    write_json_config,
    enable_rdkit_log,
    setup_responder,
    config_parse,
    get_cuda_driver_version,
    report_resource_usage,
    get_accelerator,
    report_hardware,
    fatal,
    format_validation_error,
)

from reinvent import version, runmodes
from reinvent.utils import setup_logger
from reinvent.runmodes.utils import set_torch_device
from reinvent.runmodes.handler import StageInterruptedControlled
from .validation import ReinventConfig

rdBase.DisableLog("rdApp.*")


def main(args: Any):
    """Simple entry point into Reinvent's run modes.

    :param args: arguments object, can be argparse.Namespace or any other class
    """

    logger = setup_logger(
        name=__package__, level=args.log_level.upper(), filename=args.log_filename
    )

    logger.info(
        f"Started {version.__progname__} {version.__version__} {version.__copyright__} on "
        f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
    )

    logger.info(f"Command line: {' '.join(sys.argv)}")

    dotenv_loaded = load_dotenv(args.dotenv_filename)  # set up the environment for scoring

    ext = None

    if args.config_filename:
        ext = args.config_filename.suffix

    if ext in (f".{e}" for e in config_parse.INPUT_FORMAT_CHOICES):
        fmt = ext[1:]
    else:
        fmt = args.config_format

    config_source = args.config_filename or "stdin"
    logger.info(f"Reading run configuration from {config_source} using format {fmt}")
    input_config = config_parse.read_config(args.config_filename, fmt)
    val_config = ReinventConfig(**input_config)

    if args.enable_rdkit_log_levels:
        enable_rdkit_log(args.enable_rdkit_log_levels)

    run_type = val_config.run_type
    runner = getattr(runmodes, f"run_{run_type}", None)

    if runner is None:
        from reinvent.validation import ReinventConfig as _RC

        valid = list(_RC.model_fields["run_type"].annotation.__args__)
        raise RuntimeError(f"Unknown run_type '{run_type}'. Valid choices: {', '.join(valid)}")

    if val_config.version < version.__config_version__:
        raise RuntimeError(
            f"Need at least version 4.  Input file is for version {val_config.version}."
        )

    if dotenv_loaded:
        if args.dotenv_filename:
            filename = args.dotenv_filename
        else:
            filename = find_dotenv()

        logger.info(f"Environment loaded from dotenv file {filename}")

    logger.info(f"User {getpass.getuser()} on host {platform.node()}")
    logger.info(f"Python version {platform.python_version()}")
    logger.info(f"PyTorch version {torch.__version__} git {torch.version.git_version}")

    accelerator, accelerator_runtime = get_accelerator()
    logger.info(
        f"PyTorch compiled with {accelerator}"
        + (f" version {accelerator_runtime}" if accelerator_runtime else "")
    )

    logger.info(f"RDKit version {rdkit.__version__}")
    logger.info(f"Platform {platform.platform()}")

    if cuda_driver_version := get_cuda_driver_version():
        logger.info(f"CUDA driver version {cuda_driver_version}")

    if torch.cuda.is_available():
        logger.info(f"Number of PyTorch CUDA devices {torch.cuda.device_count()}")

    device = getattr(val_config, "device", None)

    if not device:
        device = "cpu"

    actual_device = set_torch_device(args.device, device)

    report_hardware(actual_device)

    seed = args.seed or getattr(val_config, "seed", None)

    if seed is not None:
        set_seed(seed)
        logger.info(f"Set seed for all random generators to {seed}")

    tb_logdir = getattr(val_config, "tb_logdir", None)

    if tb_logdir:
        tb_logdir = os.path.abspath(tb_logdir)
        logger.info(f"Writing TensorBoard summary to {tb_logdir}")

    write_config = None

    if hasattr(val_config, "json_out_config") and val_config.json_out_config:
        json_out_config = os.path.abspath(val_config.json_out_config)
        logger.info(f"Writing JSON config file to {json_out_config}")
        write_config = write_json_config(val_config.model_dump(), json_out_config)

    responder_config = getattr(val_config, "responder", None)

    if responder_config:
        setup_responder(responder_config)
        logger.info(
            f"Using remote monitor endpoint {val_config.responder['endpoint']} "
            f"with frequency {val_config.responder['frequency']}"
        )

    try:
        runner(
            input_config=extract_sections(input_config),
            device=actual_device,
            tb_logdir=tb_logdir,
            responder_config=responder_config,
            write_config=write_config,
        )
    except StageInterruptedControlled as e:
        logger.info(f"Requested to terminate: {e.args[0]}")

    report_resource_usage(actual_device)

    logger.info(
        f"Finished {version.__progname__} on {datetime.datetime.now().strftime('%Y-%m-%d')}"
    )


def main_script():
    """Main entry point from the command line"""

    args = parse_command_line()

    # Pre-logger checks: report immediately, before the log file is opened.
    if args.config_filename and not args.config_filename.exists():
        fatal(f"Config file not found: {args.config_filename}\n" "Check the path and try again.")

    if args.log_filename:
        from pathlib import Path

        log_parent = Path(args.log_filename).parent
        if not log_parent.exists():
            fatal(
                f"Log file directory does not exist: {log_parent}\n"
                "Create the directory or choose a different path with --log-filename."
            )

    try:
        main(args)
    except pydantic.ValidationError as exc:
        # main() calls setup_logger() as its first action, so the logger is
        # configured by the time any exception escapes — look it up so the
        # same message that appears on stderr is also written to the log file.
        fatal(format_validation_error(exc), log=logging.getLogger(__package__), exc_info=exc)
    except FileNotFoundError as exc:
        fatal(
            f"File not found: {exc.filename}\n"
            "Check that the file path in your configuration is correct.",
            log=logging.getLogger(__package__),
            exc_info=exc,
        )
    except (ValueError, RuntimeError) as exc:
        fatal(str(exc), log=logging.getLogger(__package__), exc_info=exc)
    except Exception as exc:
        fatal(
            f"Unexpected error: {type(exc).__name__}: {exc}\n"
            "See the log file for the full traceback.",
            log=logging.getLogger(__package__),
            exc_info=exc,
        )


if __name__ == "__main__":
    main_script()
