from dataclasses import MISSING

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlModelGraphCfg:
    """Configuration for the restricted Node/Cell/Route model graph."""

    class_name: str = "ModelGraph"
    nodes: dict[str, dict] = MISSING
    routes: list[dict] = MISSING
    output: str = MISSING
    obs_normalization: bool = False
    distribution_cfg: RslRlMLPModelCfg.DistributionCfg | None = None
