# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

import builtins

from isaaclab_tasks.utils import import_packages

# Isaac Lab's app launcher may temporarily evict packages containing "lab" from
# sys.modules while Kit starts. Keep registration idempotent across that re-import.
if not getattr(builtins, "_engineai_rl_lab_tasks_registered", False):
    # The blacklist is used to prevent importing configs from sub-packages.
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    import_packages(__name__, _BLACKLIST_PKGS)
    builtins._engineai_rl_lab_tasks_registered = True
