from importlib.util import find_spec
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            )
        )
    if find_spec("isaaclab_newton") is not None:
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg

        properties.append(NewtonArticulationCfg(self_collision_enabled=True))
    return properties or None


# Physical Parameters (based on pm.py motor specs)
# High-torque joints: Q90 motor (HIP_PITCH, HIP_ROLL, KNEE_PITCH)
ARMATURE_Q90 = 0.0453
EFFORT_LIMIT_Q90 = 164.0
VELOCITY_LIMIT_Q90 = 26.3

# Low-torque joints: Q25 motor (HIP_YAW, ANKLE, WAIST, SHOULDER, ELBOW, HEAD)
ARMATURE_Q25 = 0.0067
EFFORT_LIMIT_Q25 = 52.0
VELOCITY_LIMIT_Q25 = 35.2

# Control parameters: 10Hz natural frequency with critical damping
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# Calculate stiffness and damping based on natural frequency and damping ratio
STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2  # ≈ 178.5
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2  # ≈ 26.4
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ  # ≈ 11.4
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ  # ≈ 1.69


@configclass
class RobotArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None


PM01_CYLINDER_CFG = RobotArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSET_DIR / "pm01" / "serial_pm01_edu.usd"),
        activate_contact_sensors=True,
        rigid_props=_rigid_body_properties(),
        articulation_props=_articulation_properties(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*HIP_PITCH.*": -0.06,
            ".*KNEE_PITCH.*": 0.12,
            ".*ANKLE_PITCH.*": -0.06,
            ".*ELBOW_PITCH.*": -0.25,
            "J14_SHOULDER_ROLL_L": 0.15,
            "J19_SHOULDER_ROLL_R": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*HIP.*",
                ".*KNEE.*",
            ],
            effort_limit_sim={
                ".*HIP_PITCH.*": EFFORT_LIMIT_Q90,
                ".*HIP_ROLL.*": EFFORT_LIMIT_Q90,
                ".*HIP_YAW.*": EFFORT_LIMIT_Q25,
                ".*KNEE_PITCH.*": EFFORT_LIMIT_Q90,
            },
            velocity_limit_sim={
                ".*HIP_PITCH.*": VELOCITY_LIMIT_Q90,
                ".*HIP_ROLL.*": VELOCITY_LIMIT_Q90,
                ".*HIP_YAW.*": VELOCITY_LIMIT_Q25,
                ".*KNEE_PITCH.*": VELOCITY_LIMIT_Q90,
            },
            stiffness={
                ".*HIP_PITCH.*": STIFFNESS_Q90,
                ".*HIP_ROLL.*": STIFFNESS_Q90,
                ".*HIP_YAW.*": STIFFNESS_Q25,
                ".*KNEE_PITCH.*": STIFFNESS_Q90,
            },
            damping={
                ".*HIP_PITCH.*": DAMPING_Q90,
                ".*HIP_ROLL.*": DAMPING_Q90,
                ".*HIP_YAW.*": DAMPING_Q25,
                ".*KNEE_PITCH.*": DAMPING_Q90,
            },
            armature={
                ".*HIP_PITCH.*": ARMATURE_Q90,
                ".*HIP_ROLL.*": ARMATURE_Q90,
                ".*HIP_YAW.*": ARMATURE_Q25,
                ".*KNEE_PITCH.*": ARMATURE_Q90,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=0.5,
            armature=ARMATURE_Q25,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J12_WAIST_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=DAMPING_Q25,
            armature=ARMATURE_Q25,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*SHOULDER.*",
                ".*ELBOW.*",
            ],
            effort_limit_sim={
                ".*SHOULDER.*": EFFORT_LIMIT_Q25,
                ".*ELBOW.*": EFFORT_LIMIT_Q25,
            },
            velocity_limit_sim={
                ".*SHOULDER.*": VELOCITY_LIMIT_Q25,
                ".*ELBOW.*": VELOCITY_LIMIT_Q25,
            },
            stiffness={
                ".*SHOULDER.*": STIFFNESS_Q25,
                ".*ELBOW.*": STIFFNESS_Q25,
            },
            damping={
                ".*SHOULDER.*": DAMPING_Q25,
                ".*ELBOW.*": DAMPING_Q25,
            },
            armature={
                ".*SHOULDER.*": ARMATURE_Q25,
                ".*ELBOW.*": ARMATURE_Q25,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["J23_HEAD_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=DAMPING_Q25,
            armature=ARMATURE_Q25,
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
        "J12_WAIST_YAW",
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
        "J23_HEAD_YAW",
    ],
)
PM01_CYLINDER_CFG = apply_articulation_ordering_preset(PM01_CYLINDER_CFG, "physx")

PM01_ACTION_SCALE = {}
for a in PM01_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    if not isinstance(e, dict):
        e = {n: e for n in a.joint_names_expr}
    if not isinstance(s, dict):
        s = {n: s for n in a.joint_names_expr}
    for n in e.keys():
        # if "ANKLE" in n:
        #     continue
        if n in s and s[n]:
            PM01_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
