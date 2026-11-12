"""
Fetch Unitree's H2 model and turn it into the locomotion model, on demand.

The model is the MJCF Unitree trains the H2 on (unitree_rl_mjlab,
src/assets/robots/unitree_h2); nothing is committed to mushroom-rl. H2Base
calls ensure_h2_model() at construction, and the first call on a machine
fetches the pinned upstream commit and applies what mjlab applies at run
time there: the simulation options, the collision setup (every collision
capsule on, self collisions included, friction only at the feet), the joint
armature and the position actuators with Unitree's gains and effort limits.
It adds a standing keyframe and a floor, compiles the result and moves it
into place.

Lookup order: $MUSHROOM_RL_H2_DIR if set, then the package data dir, then
$XDG_CACHE_HOME/mushroom_rl/h2/<build id>. Pre-fetch where there is network
with

    python -m mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.h2.fetch_h2_model

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


UNITREE_URL = "https://github.com/unitreerobotics/unitree_rl_mjlab.git"

# Pinned upstream commit (2026-04-13). Bump deliberately.
UNITREE_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"

MODEL_DIR = "src/assets/robots/unitree_h2/xmls"
MODEL_NAME = "h2.xml"
ENV_VAR = "MUSHROOM_RL_H2_DIR"

MODEL_FILES = ["h2.xml", "scene.xml"]
ASSET_DIR = "assets"

ROOT_BODY = "pelvis"
FREE_JOINT = "floating_base_joint"
FEET_BODIES = ("left_ankle_pitch_link", "right_ankle_pitch_link")
FEET_SITES = ("left_foot", "right_foot")

# Simulation options of Unitree's H2 velocity task (mjlab defaults where the
# task sets nothing).
OPTION = dict(
    timestep="0.005",
    integrator="implicitfast",
    iterations="10",
    ls_iterations="20",
    cone="pyramidal",
    impratio="1",
)

# Position actuators, (substring of joint name, kp, kd, effort limit). First
# match wins. Unitree's values (h2_constants.py).
ACTUATORS = (
    ("hip", 200.0, 4.0, 360.0),
    ("knee", 200.0, 4.0, 360.0),
    ("ankle_roll", 40.0, 2.0, 19.0),
    ("ankle_pitch", 40.0, 2.0, 66.0),
    ("waist", 150.0, 3.0, 120.0),
    ("shoulder", 40.0, 2.0, 54.0),
    ("elbow", 40.0, 2.0, 54.0),
    ("wrist_roll", 40.0, 2.0, 54.0),
    ("wrist", 20.0, 1.0, 25.0),
)
ARMATURE = "0.01"

# Collisions: every "*_collision" capsule collides, with the floor and with
# the other links. Only the feet have friction of their own.
FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
FOOT_GEOM = dict(contype="1", conaffinity="1", condim="3", priority="1", friction="0.6 0.005 0.0001")
BODY_GEOM = dict(contype="1", conaffinity="1", condim="1", priority="0")

# Standing pose ("home" keyframe), Unitree's. Joints not listed are zero.
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.25,
    "right_hip_pitch_joint": -0.25,
    "left_knee_joint": 0.5,
    "right_knee_joint": 0.5,
    "left_ankle_pitch_joint": -0.25,
    "right_ankle_pitch_joint": -0.25,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
}
FOOT_CLEARANCE = 0.002  # gap between sole and floor in the keyframe


def actuator_params(joint_name):
    """kp, kd and effort limit of the actuator of a joint."""
    for key, kp, kd, effort in ACTUATORS:
        if key in joint_name:
            return kp, kd, effort
    raise RuntimeError(f"no actuator parameters for joint {joint_name!r}; extend ACTUATORS")


def build_id():
    """Short hash of everything that affects the generated model."""
    spec = json.dumps(
        [
            UNITREE_COMMIT,
            OPTION,
            ACTUATORS,
            ARMATURE,
            FOOT_GEOM_PATTERN,
            FOOT_GEOM,
            BODY_GEOM,
            DEFAULT_POSE,
            FOOT_CLEARANCE,
        ],
        sort_keys=True,
    )
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


def _log(msg):
    print(f"[fetch_h2_model] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------


def package_dir():
    return Path(__file__).resolve().parent.parent / "data" / "h2"


def cache_dir():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "mushroom_rl" / "h2" / build_id()


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
            "Building the H2 model needs network access to github.com. On an "
            "offline machine, run `python -m "
            "mushroom_rl.environments.mujoco_warp_envs.locomotion.unitree.h2.fetch_h2_model` where "
            f"there is network access, or set {ENV_VAR} to an existing copy."
        ) from e


def _fetch(workdir):
    if shutil.which("git") is None:
        raise RuntimeError(
            f"git not found on PATH; needed to fetch the H2 model "
            f"(or set {ENV_VAR} to an existing copy)"
        )
    _log(f"fetching {MODEL_DIR} @ {UNITREE_COMMIT[:12]} from {UNITREE_URL}")
    repo = Path(workdir)
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("remote", "add", "origin", UNITREE_URL, cwd=repo)
    _git("config", "core.sparseCheckout", "true", cwd=repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text(f"/{MODEL_DIR}/\n")
    _git(
        "fetch",
        "-q",
        "--depth",
        "1",
        "--filter=blob:none",
        "origin",
        UNITREE_COMMIT,
        cwd=repo,
    )
    _git("checkout", "-q", "FETCH_HEAD", cwd=repo)
    src = repo / MODEL_DIR
    if not (src / MODEL_NAME).is_file():
        raise RuntimeError(f"{MODEL_DIR}/{MODEL_NAME} not found in unitree_rl_mjlab at {UNITREE_COMMIT}")
    return src


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------


def _fmt(values):
    return " ".join(f"{v:.6g}" for v in values)


def _postprocess(src_xml, out_xml):
    """Apply to Unitree's MJCF what mjlab applies to it when the task is built."""
    import mujoco

    tree = ET.parse(src_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    compiler.set("autolimits", "true")
    option = ET.Element("option", OPTION)
    root.insert(list(root).index(compiler) + 1, option)

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

    # Joints: name check on the free joint, armature on the hinges.
    hinges = []
    for joint in root.iter("joint"):
        if joint.get("type") == "free":
            if joint.get("name") != FREE_JOINT:
                raise RuntimeError(f"free joint is {joint.get('name')!r}, expected {FREE_JOINT!r}")
            continue
        if joint.get("name") is None:  # the <default> entries
            continue
        joint.set("armature", ARMATURE)
        hinges.append(joint)

    # Position actuators, one per hinge joint, in joint order. The target is
    # not confined to the joint range: at a joint limit the policy can still
    # ask for the full effort, as with mjlab's actuators.
    actuator = ET.SubElement(root, "actuator")
    for joint in hinges:
        name = joint.get("name")
        lo, hi = (float(v) for v in joint.get("range").split())
        kp, kd, effort = actuator_params(name)
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
<mujoco model="h2 scene">
  <include file="h2.xml"/>

  <statistic center="0 0 0.9" extent="1.8"/>

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
    hinge_names = _postprocess(src / MODEL_NAME, dest / "h2.xml")
    (dest / "scene.xml").write_text(SCENE_XML)
    (dest / "PROVENANCE.txt").write_text(
        "Unitree H2 model, from Unitree's MJCF\n"
        f"Source: {UNITREE_URL}\n"
        f"Path:   {MODEL_DIR}/{MODEL_NAME}\n"
        f"Commit: {UNITREE_COMMIT}\n"
        f"Build:  {build_id()}\n"
        "\n"
        "Built by mushroom_rl/environments/mujoco_warp_envs/locomotion/unitree/h2/fetch_h2_model.py.\n"
        "Licensed under the terms of the upstream repository (Apache-2.0,\n"
        "Unitree Robotics).\n"
        "\n"
        "Modifications relative to the upstream MJCF (what mjlab applies there at run time):\n"
        "  - simulation options (see OPTION in fetch_h2_model.py);\n"
        "  - collision capsules enabled, friction on the feet only (FOOT_GEOM, BODY_GEOM);\n"
        "  - joint armature;\n"
        "  - position actuators with PD gains and effort limits (ACTUATORS);\n"
        "  - 'home' keyframe (DEFAULT_POSE), height solved numerically;\n"
        "  - scene.xml with a floor.\n"
        f"Actuated joints, in actuator order:\n  " + "\n  ".join(hinge_names) + "\n"
    )


def fetch_h2_model(dest, force=False):
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _lock(dest.parent / f".{dest.name}.lock"):
        if not force and is_complete(dest):
            return dest

        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{dest.name}.tmp-") as tmp:
            tmp = Path(tmp)
            src = _fetch(tmp / "unitree_rl_mjlab")
            staged = tmp / "model"
            _build(src, staged)
            _verify(staged)
            if dest.exists():
                os.replace(dest, tmp / "stale")
            os.replace(staged, dest)

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    _log(f"H2 model ready in {dest} ({size / 1e6:.1f} MB)")
    return dest


def ensure_h2_model(scene="scene.xml"):
    """
    Return the path of the requested H2 scene file, building the model first
    if no complete copy exists.

    """
    if scene not in MODEL_FILES:
        raise ValueError(f"unknown H2 scene {scene!r}, expected one of {MODEL_FILES}")

    candidates = candidate_dirs()
    for d in candidates:
        if is_complete(d):
            return d / scene

    target = next((d for d in candidates if _is_writable(d)), None)
    if target is None:
        raise RuntimeError(
            "H2 model not found and no writable location to build it into. "
            f"Tried: {', '.join(map(str, candidates))}. "
            f"Set {ENV_VAR} to a writable directory."
        )
    fetch_h2_model(target)
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
        fetch_h2_model(args.dest, force=args.force)
    elif args.force:
        target = next((d for d in candidate_dirs() if _is_writable(d)), None)
        if target is None:
            sys.exit("no writable location; pass --dest")
        fetch_h2_model(target, force=True)
    else:
        print(ensure_h2_model().parent)


if __name__ == "__main__":
    main()
