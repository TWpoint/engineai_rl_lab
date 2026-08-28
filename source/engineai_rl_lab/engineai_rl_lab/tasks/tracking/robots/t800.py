from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

from engineai_rl_lab.tasks.tracking.robots.actuator import (
    DelayedImplicitActuatorCfg,
    FixedPositionImplicitActuatorCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import apply_articulation_ordering_preset
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass

_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets"


def _rigid_body_properties():
    """Build available backend fragments without requiring PhysX in kit-less Newton installs."""
    if find_spec("isaaclab_physx") is None:
        return None
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    return [
        PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        )
    ]


def _articulation_properties():
    """Author self-collision settings for every installed physics backend."""
    properties = []
    if find_spec("isaaclab_physx") is not None:
        from isaaclab_physx.sim.schemas import PhysxArticulationCfg

        properties.append(
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            )
        )
    if find_spec("isaaclab_newton") is not None:
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg

        properties.append(NewtonArticulationCfg(self_collision_enabled=False))
    return properties or None


EFFORT_LIMIT_Q300HL = 415  # hip pitch, knee pitch
EFFORT_LIMIT_Q300H = 370  # hip roll
EFFORT_LIMIT_Q200H = 222  # hip yaw, torso yaw
EFFORT_LIMIT_Q50H = 160  # ankle pitch, ankle roll, shoulder, elbow pitch
EFFORT_LIMIT_Q25H = 52  # elbow yaw, head pitch, head yaw

VELOCITY_LIMIT_Q300HL = 25.96  # hip pitch, knee pitch
VELOCITY_LIMIT_Q300H = 25.31  # hip roll
VELOCITY_LIMIT_Q200H = 23.19  # hip yaw
VELOCITY_LIMIT_Q50H = 33.51  # ankle pitch, ankle roll, shoulder, elbow pitch
VELOCITY_LIMIT_Q25H = 35.2  # elbow yaw

ARMATURE_Q300HL = 0.2427264  # hip pitch, knee pitch
ARMATURE_Q300H = 0.14110848  # hip roll
ARMATURE_Q200H = 0.0448737  # hip yaw, torso yaw
ARMATURE_Q50H = 0.0354625  # ankle pitch, ankle roll, shoulder, elbow pitch
ARMATURE_Q25H = 0.00671625  # elbow yaw, head pitch, head yaw

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_Q300HL = ARMATURE_Q300HL * (NATURAL_FREQ**2)
STIFFNESS_Q300H = ARMATURE_Q300H * (NATURAL_FREQ**2)
STIFFNESS_Q200H = ARMATURE_Q200H * (NATURAL_FREQ**2)
STIFFNESS_Q50H = ARMATURE_Q50H * (NATURAL_FREQ**2)
STIFFNESS_Q25H = ARMATURE_Q25H * (NATURAL_FREQ**2)

DAMPING_Q300HL = 2.0 * DAMPING_RATIO * ARMATURE_Q300HL * NATURAL_FREQ
DAMPING_Q300H = 2.0 * DAMPING_RATIO * ARMATURE_Q300H * NATURAL_FREQ
DAMPING_Q200H = 2.0 * DAMPING_RATIO * ARMATURE_Q200H * NATURAL_FREQ
DAMPING_Q50H = 2.0 * DAMPING_RATIO * ARMATURE_Q50H * NATURAL_FREQ
DAMPING_Q25H = 2.0 * DAMPING_RATIO * ARMATURE_Q25H * NATURAL_FREQ

DEFAULT_Q_HIP_PITCH = -0.06
DEFAULT_Q_HIP_ROLL = 0.0
DEFAULT_Q_HIP_YAW = 0.0
DEFAULT_Q_KNEE_PITCH = 0.12
DEFAULT_Q_ANKLE_PITCH = -0.06
DEFAULT_Q_ANKLE_ROLL = 0.0
DEFAULT_Q_TORSO_YAW = 0.0
DEFAULT_Q_SHOULDER_PITCH = 0.0
DEFAULT_Q_SHOULDER_ROLL_L = 0.15
DEFAULT_Q_SHOULDER_ROLL_R = -0.15
DEFAULT_Q_SHOULDER_YAW = 0.0
DEFAULT_Q_ELBOW_PITCH = -0.25
DEFAULT_Q_ELBOW_YAW = 0.0
DEFAULT_Q_HEAD_PITCH = 0.0
DEFAULT_Q_HEAD_YAW = 0.0


@configclass
class RobotArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None


T800_CYLINDER_CFG = RobotArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSET_DIR / "t800" / "serial_t800_stable.usda"),
        activate_contact_sensors=True,
        rigid_props=_rigid_body_properties(),
        articulation_props=_articulation_properties(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.06),
        joint_pos={
            "J00_HIP_PITCH_L": DEFAULT_Q_HIP_PITCH,
            "J01_HIP_ROLL_L": DEFAULT_Q_HIP_ROLL,
            "J02_HIP_YAW_L": DEFAULT_Q_HIP_YAW,
            "J03_KNEE_PITCH_L": DEFAULT_Q_KNEE_PITCH,
            "J04_ANKLE_PITCH_L": DEFAULT_Q_ANKLE_PITCH,
            "J05_ANKLE_ROLL_L": DEFAULT_Q_ANKLE_ROLL,
            "J06_HIP_PITCH_R": DEFAULT_Q_HIP_PITCH,
            "J07_HIP_ROLL_R": DEFAULT_Q_HIP_ROLL,
            "J08_HIP_YAW_R": DEFAULT_Q_HIP_YAW,
            "J09_KNEE_PITCH_R": DEFAULT_Q_KNEE_PITCH,
            "J10_ANKLE_PITCH_R": DEFAULT_Q_ANKLE_PITCH,
            "J11_ANKLE_ROLL_R": DEFAULT_Q_ANKLE_ROLL,
            "J12_TORSO_YAW": DEFAULT_Q_TORSO_YAW,
            "J13_SHOULDER_PITCH_L": DEFAULT_Q_SHOULDER_PITCH,
            "J14_SHOULDER_ROLL_L": DEFAULT_Q_SHOULDER_ROLL_L,
            "J15_SHOULDER_YAW_L": DEFAULT_Q_SHOULDER_YAW,
            "J16_ELBOW_PITCH_L": DEFAULT_Q_ELBOW_PITCH,
            "J17_ELBOW_YAW_L": DEFAULT_Q_ELBOW_YAW,
            "J18_SHOULDER_PITCH_R": DEFAULT_Q_SHOULDER_PITCH,
            "J19_SHOULDER_ROLL_R": DEFAULT_Q_SHOULDER_ROLL_R,
            "J20_SHOULDER_YAW_R": DEFAULT_Q_SHOULDER_YAW,
            "J21_ELBOW_PITCH_R": DEFAULT_Q_ELBOW_PITCH,
            "J22_ELBOW_YAW_R": DEFAULT_Q_ELBOW_YAW,
            "J23_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
            "J24_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch_and_knee_pitch": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J00_HIP_PITCH_L",
                "J06_HIP_PITCH_R",
                "J03_KNEE_PITCH_L",
                "J09_KNEE_PITCH_R",
            ],
            effort_limit_sim=EFFORT_LIMIT_Q300HL * 0.95,
            velocity_limit_sim=VELOCITY_LIMIT_Q300HL,
            stiffness=180.0,
            damping=5.0,
            armature=ARMATURE_Q300HL,
            min_delay=1,
            max_delay=3,
        ),
        "hip_roll": DelayedImplicitActuatorCfg(
            joint_names_expr=["J01_HIP_ROLL_L", "J07_HIP_ROLL_R"],
            effort_limit_sim=EFFORT_LIMIT_Q300H,
            velocity_limit_sim=VELOCITY_LIMIT_Q300H,
            stiffness=100.0,
            damping=3.0,
            armature=ARMATURE_Q300H,
            min_delay=1,
            max_delay=3,
        ),
        "hip_yaw": DelayedImplicitActuatorCfg(
            joint_names_expr=["J02_HIP_YAW_L", "J08_HIP_YAW_R"],
            effort_limit_sim=EFFORT_LIMIT_Q200H,
            velocity_limit_sim=VELOCITY_LIMIT_Q200H,
            stiffness=100.0,
            damping=3.0,
            armature=ARMATURE_Q200H,
            min_delay=1,
            max_delay=3,
        ),
        "ankles": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J04_ANKLE_PITCH_L",
                "J05_ANKLE_ROLL_L",
                "J10_ANKLE_PITCH_R",
                "J11_ANKLE_ROLL_R",
            ],
            effort_limit_sim=EFFORT_LIMIT_Q50H,
            velocity_limit_sim=VELOCITY_LIMIT_Q50H,
            stiffness=40.0,
            damping=2.0,
            armature=ARMATURE_Q50H,
            min_delay=1,
            max_delay=3,
        ),
        "torso_yaw": DelayedImplicitActuatorCfg(
            joint_names_expr=["J12_TORSO_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q200H,
            velocity_limit_sim=VELOCITY_LIMIT_Q200H,
            stiffness=100.0,
            damping=3.0,
            armature=ARMATURE_Q200H,
            min_delay=1,
            max_delay=3,
        ),
        "shoulder_and_elbow_pitch": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J13_SHOULDER_PITCH_L",
                "J14_SHOULDER_ROLL_L",
                "J15_SHOULDER_YAW_L",
                "J16_ELBOW_PITCH_L",
                "J18_SHOULDER_PITCH_R",
                "J19_SHOULDER_ROLL_R",
                "J20_SHOULDER_YAW_R",
                "J21_ELBOW_PITCH_R",
            ],
            effort_limit_sim=EFFORT_LIMIT_Q50H,
            velocity_limit_sim=VELOCITY_LIMIT_Q50H,
            stiffness=50.0,
            damping=0.3,
            armature=ARMATURE_Q50H,
            min_delay=1,
            max_delay=3,
        ),
        "heads_and_elbow_yaw": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J17_ELBOW_YAW_L",
                "J22_ELBOW_YAW_R",
                "J23_HEAD_PITCH",
                "J24_HEAD_YAW",
            ],
            effort_limit_sim=EFFORT_LIMIT_Q25H,
            velocity_limit_sim=VELOCITY_LIMIT_Q25H,
            stiffness=50.0,
            damping=0.3,
            armature=ARMATURE_Q25H,
            min_delay=1,
            max_delay=3,
        ),
    },
    joint_sdk_names=[
        "J00_HIP_PITCH_L",
        "J01_HIP_ROLL_L",
        "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L",
        "J04_ANKLE_PITCH_L",
        "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R",
        "J07_HIP_ROLL_R",
        "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R",
        "J10_ANKLE_PITCH_R",
        "J11_ANKLE_ROLL_R",
        "J12_TORSO_YAW",
        "J13_SHOULDER_PITCH_L",
        "J14_SHOULDER_ROLL_L",
        "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L",
        "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R",
        "J19_SHOULDER_ROLL_R",
        "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R",
        "J22_ELBOW_YAW_R",
        "J23_HEAD_PITCH",
        "J24_HEAD_YAW",
    ],
)
T800_CYLINDER_CFG = apply_articulation_ordering_preset(T800_CYLINDER_CFG, "physx")

# V21 robot variant: the policy does not command the head, and these actuators hold it at zero.
# Keep this separate from T800_CYLINDER_CFG so earlier task versions retain 25-DoF control.
T800_FIXED_HEAD_CFG = deepcopy(T800_CYLINDER_CFG)
T800_FIXED_HEAD_CFG.actuators["heads_and_elbow_yaw"].joint_names_expr = [
    "J17_ELBOW_YAW_L",
    "J22_ELBOW_YAW_R",
]
T800_FIXED_HEAD_CFG.actuators["fixed_head"] = FixedPositionImplicitActuatorCfg(
    joint_names_expr=["J23_HEAD_PITCH", "J24_HEAD_YAW"],
    effort_limit_sim=EFFORT_LIMIT_Q25H,
    velocity_limit_sim=VELOCITY_LIMIT_Q25H,
    stiffness=50.0,
    damping=0.3,
    armature=ARMATURE_Q25H,
    fixed_position=0.0,
)

T800_ACTION_SCALE: dict[str, float] = {
    "J00_HIP_PITCH_L": 0.5,
    "J01_HIP_ROLL_L": 0.2,
    "J02_HIP_YAW_L": 0.2,
    "J03_KNEE_PITCH_L": 0.5,
    "J04_ANKLE_PITCH_L": 0.5,
    "J05_ANKLE_ROLL_L": 0.2,
    "J06_HIP_PITCH_R": 0.5,
    "J07_HIP_ROLL_R": 0.2,
    "J08_HIP_YAW_R": 0.2,
    "J09_KNEE_PITCH_R": 0.5,
    "J10_ANKLE_PITCH_R": 0.5,
    "J11_ANKLE_ROLL_R": 0.2,
    "J12_TORSO_YAW": 0.2,
    "J13_SHOULDER_PITCH_L": 0.2,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.2,
    "J17_ELBOW_YAW_L": 0.05,
    "J18_SHOULDER_PITCH_R": 0.2,
    "J19_SHOULDER_ROLL_R": 0.2,
    "J20_SHOULDER_YAW_R": 0.05,
    "J21_ELBOW_PITCH_R": 0.2,
    "J22_ELBOW_YAW_R": 0.05,
    "J23_HEAD_PITCH": 0.2,
    "J24_HEAD_YAW": 0.2,
}

T800_HEAD_JOINT_POSITIONS: dict[str, float] = {
    "J23_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
    "J24_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
}
T800_POLICY_JOINT_NAMES: list[str] = [
    name for name in T800_ACTION_SCALE if name not in T800_HEAD_JOINT_POSITIONS
]
T800_POLICY_ACTION_SCALE: dict[str, float] = {
    name: T800_ACTION_SCALE[name] for name in T800_POLICY_JOINT_NAMES
}
