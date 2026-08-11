Changelog
---------

Unreleased
~~~~~~~~~~

Changed
^^^^^^^

* Migrated the tracking environments and launch scripts from Isaac Lab 2.3 to the Isaac Lab 3.0
  multi-backend APIs.
* Added Newton MJWarp physics presets and kit-less training/playback support through
  ``physics=newton_mjwarp`` and the 3.0 visualizer CLI.
* Migrated quaternion data to XYZW and added explicit quaternion, joint-name, and body-name metadata
  for newly converted motion files while retaining legacy NPZ compatibility.
* Updated RSL-RL integration, Python requirement, extension metadata, and wheel package discovery for
  the Isaac Lab 3.0/Python 3.12 stack.

0.1.0 (2026-06-05)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
