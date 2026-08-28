"""Launch Isaac Sim Kit before executing a target Isaac Lab script.

Kit must start before task/config modules import pxr, ONNX, protobuf, or gRPC.
This wrapper is used by shell entrypoints that select PhysX at runtime, after
their normal Python module has already been designed around kitless Newton.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: {sys.argv[0]} TARGET_SCRIPT [TARGET_ARGS ...]")

    target_script = sys.argv[1]
    target_args = sys.argv[2:]

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, _ = parser.parse_known_args(target_args)

    # Do not let Kit consume target-script flags such as --help, --task, or
    # --viz. AppLauncher receives its supported options through the namespace.
    sys.argv = [sys.argv[0]]
    app_launcher = AppLauncher(launcher_args)
    try:
        sys.argv = [target_script, *target_args]
        target_dir = os.path.dirname(os.path.abspath(target_script))
        sys.path.insert(0, target_dir)
        # play.py has a sibling module named cli_args. Kit also imports a
        # top-level module with that generic name, so discard Kit's cached one.
        sys.modules.pop("cli_args", None)
        print(f"[INFO]: Kit started; running {target_script}", flush=True)
        runpy.run_path(target_script, run_name="__main__")
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    main()
