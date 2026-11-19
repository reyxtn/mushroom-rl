"""
Fetch the Unitree G1 model and turn it into the locomotion model, on demand.

The model is MuJoCo Menagerie's G1 for GPU simulation (mujoco_menagerie,
unitree_g1/g1_mjx.xml); nothing is committed to mushroom-rl. G1Base calls
ensure_g1_model() at construction, and the first call on a machine fetches
the pinned upstream commit and turns it into the model Unitree trains the G1
on (unitree_rl_mjlab, src/assets/robots/unitree_g1, with what mjlab applies
at run time there): the simulation options, Unitree's soles, the collision
setup (every collision capsule on, self collisions included, friction only
at the feet), the joint armature, the position actuators with Unitree's
gains and effort limits, and the angular momentum sensor. It adds a standing
keyframe and a floor, compiles the result and moves it into place.

Lookup order: $MUSHROOM_RL_G1_DIR if set, then the package data dir, then
$XDG_CACHE_HOME/mushroom_rl/g1/<build id>. Pre-fetch where there is network
with

    python -m mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.g1.fetch_g1_model

The build id covers the upstream commit and every parameter below, so
changing any of them invalidates old copies.

"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import fcntl
except ImportError:  # Windows: no inter-process lock, single process assumed
    fcntl = None


MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"

# Pinned upstream commit (2026-09-01), the one the Go2 is fetched at. Bump deliberately.
MENAGERIE_COMMIT = "e4049d0a3bfd58d2a3081614e6777d4007e3f86a"

MODEL_DIR = "unitree_g1"
MODEL_NAME = "g1_mjx.xml"
ENV_VAR = "MUSHROOM_RL_G1_DIR"

MODEL_FILES = ["g1.xml", "scene.xml"]
DOC_FILES = ["LICENSE"]
ASSET_DIR = "assets"

ROOT_BODY = "pelvis"
FREE_JOINT = "floating_base_joint"
FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FEET_SITES = ("left_foot", "right_foot")
ANGMOM_SENSOR = "root_angmom"  # angular momentum of the whole robot about its centre of mass

# Simulation options of Unitree's G1 velocity task (mjlab defaults where the
# task sets nothing).
OPTION = dict(
    timestep="0.005",
    integrator="implicitfast",
    iterations="10",
    ls_iterations="20",
    cone="pyramidal",
    impratio="1",
)

# Motors, (inertia reflected to the joint in kg m^2, effort limit in N m).
# The ankles and the waist roll and pitch are driven by two 5020s through a
# linkage, taken as one motor of twice the inertia and effort. Unitree's
# values (g1_constants.py).
MOTOR_5020 = (0.003609725, 25.0)
MOTOR_7520_14 = (0.01017752004, 88.0)
MOTOR_7520_22 = (0.025101925, 139.0)
MOTOR_4010 = (0.00425, 5.0)
MOTOR_2X5020 = (2 * 0.003609725, 2 * 25.0)

# Position actuators, (substring of joint name, motor). First match wins.
ACTUATORS = (
    ("hip_pitch", MOTOR_7520_14),
    ("hip_yaw", MOTOR_7520_14),
    ("hip_roll", MOTOR_7520_22),
    ("knee", MOTOR_7520_22),
    ("ankle", MOTOR_2X5020),
    ("waist_yaw", MOTOR_7520_14),
    ("waist", MOTOR_2X5020),
    ("shoulder", MOTOR_5020),
    ("elbow", MOTOR_5020),
    ("wrist_roll", MOTOR_5020),
    ("wrist", MOTOR_4010),
)
# The gains follow from the inertia: the joint behaves as a 10 Hz oscillator
# with a damping ratio of 2.
NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# Soles: Unitree's, seven capsules side by side, where Menagerie has a box
# and three capsules. These are the five between the two toe capsules, which
# are Menagerie's; fromto in the frame of the ankle roll link, either foot.
SOLE_CAPSULES = (
    "-0.044 -0.018 -0.025 0.123 -0.018 -0.025",
    "-0.052 -0.01 -0.025 0.13 -0.01 -0.025",
    "-0.054 0 -0.025 0.132 0 -0.025",
    "-0.052 0.01 -0.025 0.13 0.01 -0.025",
    "-0.044 0.018 -0.025 0.123 0.018 -0.025",
)
FOOT_SITE_POS = "0.04 0 -0.035"  # at the sole

# Collisions: every "*_collision" capsule collides, with the floor and with
# the other links. Only the feet have friction of their own.
FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
FOOT_GEOM = dict(contype="1", conaffinity="1", condim="3", priority="1", friction="0.6 0.005 0.0001")
BODY_GEOM = dict(contype="1", conaffinity="1", condim="1", priority="0")
# Pairs of bodies that never collide, Unitree's.
EXCLUDES = (
    ("left_elbow_link", "left_wrist_pitch_link"),
    ("right_elbow_link", "right_wrist_pitch_link"),
    ("pelvis", "right_hip_roll_link"),
    ("pelvis", "left_hip_roll_link"),
)

# Standing pose ("home" keyframe), Unitree's. Joints not listed are zero.
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
}
FOOT_CLEARANCE = 0.002  # gap between sole and floor in the keyframe


def actuator_params(joint_name):
    """kp, kd, effort limit and armature of the actuator of a joint."""
    for key, (armature, effort) in ACTUATORS:
        if key in joint_name:
            kp = armature * NATURAL_FREQUENCY**2
            kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQUENCY
            return kp, kd, effort, armature
    raise RuntimeError(f"no actuator parameters for joint {joint_name!r}; extend ACTUATORS")


def build_id():
    """Short hash of everything that affects the generated model."""
    spec = json.dumps(
        [
            MENAGERIE_COMMIT,
            OPTION,
            ACTUATORS,
            NATURAL_FREQUENCY,
            DAMPING_RATIO,
            SOLE_CAPSULES,
            FOOT_SITE_POS,
            FOOT_GEOM_PATTERN,
            FOOT_GEOM,
            BODY_GEOM,
            EXCLUDES,
            DEFAULT_POSE,
            FOOT_CLEARANCE,
        ],
        sort_keys=True,
    )
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


def _log(msg):
    print(f"[fetch_g1_model] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------


def package_dir():
    return Path(__file__).resolve().parent.parent / "data" / "g1"


def cache_dir():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "mushroom_rl" / "g1" / build_id()


def candidate_dirs():
    override = os.environ.get(ENV_VAR)
    if override:
        return [Path(override).expanduser().resolve()]
    return [package_dir(), cache_dir()]


def is_complete(d):
    d = Path(d)
    if not all((d / f).is_file() for f in MODEL_FILES):
        return False
    assets = d / ASSET_DIR
    if not assets.is_dir() or not any(assets.iterdir()):
        return False
    prov = d / "PROVENANCE.txt"
    return prov.is_file() and f"Build:  {build_id()}" in prov.read_text()


def _is_writable(d):
    p = Path(d)
    while not p.exists():
        p = p.parent
    return os.access(p, os.W_OK)


# ----------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------


@contextlib.contextmanager
def _lock(path):
    if fcntl is None:
        yield
        return
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _git(*args, cwd=None):
    try:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{e.stderr.strip()}\n"
            "Building the G1 model needs network access to github.com. On an "
            "offline machine, run `python -m "
            "mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.g1.fetch_g1_model` where "
            f"there is network access, or set {ENV_VAR} to an existing copy."
        ) from e


def _fetch(workdir):
    if shutil.which("git") is None:
        raise RuntimeError(
            f"git not found on PATH; needed to fetch the G1 model "
            f"(or set {ENV_VAR} to an existing copy)"
        )
    _log(f"fetching {MODEL_DIR} @ {MENAGERIE_COMMIT[:12]} from {MENAGERIE_URL}")
    repo = Path(workdir)
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("remote", "add", "origin", MENAGERIE_URL, cwd=repo)
    _git("config", "core.sparseCheckout", "true", cwd=repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text(f"/{MODEL_DIR}/\n")
    _git(
        "fetch",
        "-q",
        "--depth",
        "1",
        "--filter=blob:none",
        "origin",
        MENAGERIE_COMMIT,
        cwd=repo,
    )
    _git("checkout", "-q", "FETCH_HEAD", cwd=repo)
    src = repo / MODEL_DIR
    if not (src / MODEL_NAME).is_file():
        raise RuntimeError(f"{MODEL_DIR}/{MODEL_NAME} not found in mujoco_menagerie at {MENAGERIE_COMMIT}")
    return src


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------


def _fmt(values):
    return " ".join(f"{v:.10g}" for v in values)


def _postprocess(src_xml, out_xml):
    """Turn Menagerie's MJCF into the model Unitree's G1 task is built on."""
    import mujoco

    # The joint ranges sit in the default classes of the MJCF: read them compiled.
    m = mujoco.MjModel.from_xml_path(str(src_xml))
    joint_range = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_range[j] for j in range(m.njnt)}

    tree = ET.parse(src_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    compiler.set("autolimits", "true")
    root.remove(root.find("option"))
    option = ET.Element("option", OPTION)
    root.insert(list(root).index(compiler) + 1, option)

    # Soles: Unitree's five capsules replace Menagerie's box and thick centre
    # capsule; the two toe capsules stay, as 1 and 7 of the seven. The sole
    # geoms are named after the foot site.
    bodies = {body.get("name"): body for body in root.iter("body")}
    for body_name, site_name in zip(FEET_BODIES, FEET_SITES):
        body = bodies[body_name]
        geoms = {geom.get("name"): geom for geom in body.findall("geom")}
        box, centre, toe = (geoms.get(f"{site_name}{part}_collision") for part in ("_box", "2", "3"))
        if box is None or centre is None or toe is None:
            raise RuntimeError(f"sole geoms of {body_name} not found; upstream model changed")
        body.remove(box)
        toe.set("name", f"{site_name}7_collision")
        at = list(body).index(centre)
        body.remove(centre)
        for i, fromto in enumerate(SOLE_CAPSULES):
            capsule = {"name": f"{site_name}{i + 2}_collision", "class": "foot_capsule", "fromto": fromto}
            body.insert(at + i, ET.Element("geom", capsule))
        next(site for site in body.findall("site") if site.get("name") == site_name).set("pos", FOOT_SITE_POS)

    # Collisions.
    foot = re.compile(FOOT_GEOM_PATTERN)
    n_foot = 0
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if not name.endswith("_collision"):
            continue
        is_foot = bool(foot.match(name))
        n_foot += is_foot
        for k, v in (FOOT_GEOM if is_foot else BODY_GEOM).items():
            geom.set(k, v)
    if n_foot != 14:
        raise RuntimeError(f"expected 14 foot collision capsules, found {n_foot}; upstream model changed")
    contact = ET.SubElement(root, "contact")
    for body1, body2 in EXCLUDES:
        ET.SubElement(contact, "exclude", dict(body1=body1, body2=body2))

    # Joints: name check on the free joint, armature on the hinges. The
    # friction loss, limit softness and force limit Menagerie gives the joints
    # are dropped: Unitree's model has none, and the effort limit is the
    # actuator's.
    free = root.find(".//freejoint")
    if free is None or free.get("name") != FREE_JOINT:
        raise RuntimeError(f"free joint {FREE_JOINT!r} not found; upstream model changed")
    for joint in root.find("default").iter("joint"):
        for attribute in ("frictionloss", "solimplimit", "solreflimit", "actuatorfrcrange"):
            joint.attrib.pop(attribute, None)
    hinges = []
    for joint in root.find("worldbody").iter("joint"):
        joint.set("armature", _fmt([actuator_params(joint.get("name"))[3]]))
        hinges.append(joint)

    # Position actuators, one per hinge joint, in joint order, in place of
    # Menagerie's. The target is not confined to the joint range: at a joint
    # limit the policy can still ask for the full effort, as with mjlab's
    # actuators.
    root.remove(root.find("actuator"))
    actuator = ET.SubElement(root, "actuator")
    for joint in hinges:
        name = joint.get("name")
        lo, hi = (float(v) for v in joint_range[name])
        kp, kd, effort, _ = actuator_params(name)
        ET.SubElement(
            actuator,
            "general",
            dict(
                name=name,
                joint=name,
                biastype="affine",
                gainprm=_fmt([kp, 0, 0]),
                biasprm=_fmt([0, -kp, -kd]),
                ctrlrange=_fmt([lo - effort / kp, hi + effort / kp]),
                forcerange=_fmt([-effort, effort]),
            ),
        )
    hinge_names = [j.get("name") for j in hinges]

    # Sensors: the one the task reads, in place of Menagerie's.
    root.remove(root.find("sensor"))
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "subtreeangmom", dict(name=ANGMOM_SENSOR, body=ROOT_BODY))

    unknown = set(DEFAULT_POSE) - set(hinge_names)
    if unknown:
        raise RuntimeError(f"DEFAULT_POSE names unknown joints: {sorted(unknown)}")

    # Write once without the keyframe so the standing height can be solved
    # on the final geometry, then add the keyframe and write again.
    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="unicode", xml_declaration=False)

    m = mujoco.MjModel.from_xml_path(str(out_xml))
    d = mujoco.MjData(m)
    qpos = np.zeros(m.nq)
    qpos[3] = 1.0
    for name, val in DEFAULT_POSE.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[m.jnt_qposadr[jid]] = val
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    qpos[2] = FOOT_CLEARANCE - _lowest_foot_point(m, d)

    ctrl = np.array(
        [qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in hinge_names]
    )
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", dict(name="home", qpos=_fmt(qpos), ctrl=_fmt(ctrl)))

    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="unicode", xml_declaration=False)
    return hinge_names


def _lowest_foot_point(m, d):
    """Height of the lowest point of the foot capsules."""
    import mujoco

    lowest = np.inf
    for gid in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid])
        if body not in FEET_BODIES or m.geom_contype[gid] == 0:
            continue
        if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise RuntimeError("foot collision geoms are expected to be capsules; upstream model changed")
        radius, half = m.geom_size[gid][:2]
        axis = d.geom_xmat[gid].reshape(3, 3)[:, 2]
        ends = d.geom_xpos[gid][2] + np.array([-1.0, 1.0]) * half * axis[2]
        lowest = min(lowest, ends.min() - radius)
    return lowest


SCENE_XML = """\
<mujoco model="g1 scene">
  <include file="g1.xml"/>

  <statistic center="0 0 0.7" extent="1.4"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" condim="3" contype="1" conaffinity="1"/>
  </worldbody>
</mujoco>
"""


# ----------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------


def _verify(dest):
    import mujoco

    for scene in MODEL_FILES:
        m = mujoco.MjModel.from_xml_path(str(dest / scene))
        d = mujoco.MjData(m)

        joint0 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, 0)
        if joint0 != FREE_JOINT or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"{scene}: joint 0 is {joint0!r}, expected free joint {FREE_JOINT!r}")
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home") != 0:
            raise RuntimeError(f"{scene}: keyframe 'home' must be keyframe 0")
        if m.nu != m.njnt - 1:
            raise RuntimeError(f"{scene}: {m.nu} actuators for {m.njnt - 1} hinge joints")
        if not (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all():
            raise RuntimeError(f"{scene}: not all actuators are position-type")
        for site in FEET_SITES:
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
                raise RuntimeError(f"{scene}: foot site {site!r} not found")

        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        low = _lowest_foot_point(m, d)
        if abs(low - FOOT_CLEARANCE) > 1e-3:
            raise RuntimeError(f"{scene}: lowest foot point at {low:.4f}, expected {FOOT_CLEARANCE}")
        if d.ncon and scene == MODEL_FILES[0]:
            raise RuntimeError(f"{scene}: {d.ncon} self contacts in the home pose")

        _log(
            f"  {scene}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
            f"mass={m.body_subtreemass[1]:.1f}kg pelvis_z={m.key_qpos[0][2]:.3f}"
        )


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------


def _build(src, dest):
    dest.mkdir(parents=True)
    shutil.copytree(src / ASSET_DIR, dest / ASSET_DIR)
    for name in DOC_FILES:
        shutil.copy(src / name, dest / name)
    hinge_names = _postprocess(src / MODEL_NAME, dest / "g1.xml")
    (dest / "scene.xml").write_text(SCENE_XML)
    (dest / "PROVENANCE.txt").write_text(
        "Unitree G1 model, from MuJoCo Menagerie's MJCF\n"
        f"Source: {MENAGERIE_URL}\n"
        f"Path:   {MODEL_DIR}/{MODEL_NAME}\n"
        f"Commit: {MENAGERIE_COMMIT}\n"
        f"Build:  {build_id()}\n"
        "\n"
        "Built by mushroom_rl/environments/mujoco_warp_envs/locomotion/unitree/g1/fetch_g1_model.py.\n"
        "Licensed under the terms in LICENSE (BSD-3-Clause, Unitree Robotics).\n"
        "\n"
        "Modifications relative to the upstream MJCF, which turn it into the model of Unitree's G1\n"
        "velocity task (unitree_rl_mjlab, Apache-2.0, with what mjlab applies there at run time):\n"
        "  - simulation options (see OPTION in fetch_g1_model.py);\n"
        "  - soles of seven capsules (SOLE_CAPSULES) in place of a box and three, foot sites at the sole;\n"
        "  - collision capsules enabled, friction on the feet only (FOOT_GEOM, BODY_GEOM), no explicit\n"
        "    contact pairs, four excluded body pairs (EXCLUDES);\n"
        "  - joint armature per motor; no joint friction loss, limit softness or force limit;\n"
        "  - position actuators with PD gains and effort limits (ACTUATORS);\n"
        "  - the angular momentum sensor in place of the IMU sensors;\n"
        "  - 'home' keyframe (DEFAULT_POSE), height solved numerically;\n"
        "  - scene.xml with a floor.\n"
        f"Actuated joints, in actuator order:\n  " + "\n  ".join(hinge_names) + "\n"
    )


def fetch_g1_model(dest, force=False):
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _lock(dest.parent / f".{dest.name}.lock"):
        if not force and is_complete(dest):
            return dest

        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{dest.name}.tmp-") as tmp:
            tmp = Path(tmp)
            src = _fetch(tmp / "mujoco_menagerie")
            staged = tmp / "model"
            _build(src, staged)
            _verify(staged)
            if dest.exists():
                os.replace(dest, tmp / "stale")
            os.replace(staged, dest)

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    _log(f"G1 model ready in {dest} ({size / 1e6:.1f} MB)")
    return dest


def ensure_g1_model(scene="scene.xml"):
    """
    Return the path of the requested G1 scene file, building the model first
    if no complete copy exists.

    """
    if scene not in MODEL_FILES:
        raise ValueError(f"unknown G1 scene {scene!r}, expected one of {MODEL_FILES}")

    candidates = candidate_dirs()
    for d in candidates:
        if is_complete(d):
            return d / scene

    target = next((d for d in candidates if _is_writable(d)), None)
    if target is None:
        raise RuntimeError(
            "G1 model not found and no writable location to build it into. "
            f"Tried: {', '.join(map(str, candidates))}. "
            f"Set {ENV_VAR} to a writable directory."
        )
    fetch_g1_model(target)
    return target / scene


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default: first writable lookup location)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if a complete copy exists")
    args = parser.parse_args()

    if args.dest is not None:
        fetch_g1_model(args.dest, force=args.force)
    elif args.force:
        target = next((d for d in candidate_dirs() if _is_writable(d)), None)
        if target is None:
            sys.exit("no writable location; pass --dest")
        fetch_g1_model(target, force=True)
    else:
        print(ensure_g1_model().parent)


if __name__ == "__main__":
    main()
