#!/usr/bin/env python3
"""
Fetch the Unitree H2 description and build an MJCF for it, on demand.

Unitree ships the H2 as a URDF (unitree_ros/robots/h2_description). Nothing
is committed to this repository: H2Base calls ensure_h2_model() at
construction, and the first call on a machine fetches the pinned upstream
commit, converts the URDF with MuJoCo, post-processes the result into a
locomotion model, compiles it, and moves it into place.

The build does, in order:
    1. sparse-fetch robots/h2_description at UNITREE_ROS_COMMIT;
    2. copy H2.urdf and only the .stl meshes it references (the .dae copies
       and the closed-loop variant are not needed);
    3. patch the URDF: enable the commented-out floating joint so the pelvis
       becomes a floating body, and fix the <mujoco> compiler block so mesh
       paths resolve;
    4. load the URDF with MuJoCo and save it back out as MJCF;
    5. post-process the MJCF into h2.xml: solver options, joint dynamics,
       collision geometry reduced to two foot boxes, position actuators with
       per-joint PD gains, an IMU site with sensors, and a standing keyframe
       whose height is solved so the feet just touch the ground;
    6. write scene.xml (floor, light, skybox) that includes h2.xml;
    7. compile both and check the invariants the environment relies on.

Lookup order and offline use are the same as for the Go2:
    1. $MUSHROOM_RL_H2_DIR, if set (used exclusively);
    2. mushroom_rl/environments/mujoco_envs/data/h2;
    3. $XDG_CACHE_HOME/mushroom_rl/h2/<build id>.
Pre-fetch on a machine with network access with

    python -m mushroom_rl.environments.mujoco_warp_envs.h2.vendor_h2

The build id in the cache path and in PROVENANCE.txt covers the upstream
commit and the parameters below (gains, default pose, options), so changing
any of them invalidates old copies.

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


UNITREE_ROS_URL = "https://github.com/unitreerobotics/unitree_ros.git"

# Pinned upstream commit ("update G1+", 2026-09-29). Bump deliberately.
UNITREE_ROS_COMMIT = "da52948f035165aae2709d30255f5cd3e62875a0"

MODEL_DIR = "robots/h2_description"
URDF_NAME = "H2.urdf"
ENV_VAR = "MUSHROOM_RL_H2_DIR"

MODEL_FILES = ["h2.xml", "scene.xml"]
SOURCE_FILES = ["H2.urdf"]
ASSET_DIR = "meshes"

ROOT_BODY = "pelvis"
FREE_JOINT = "floating_base_joint"
FEET_BODIES = {
    "left_foot": "left_ankle_pitch_link",
    "right_foot": "right_ankle_pitch_link",
}

# Simulation options. Same values as the vendored Go2 mjx scene.
OPTION = dict(
    timestep="0.002",
    iterations="8",
    ls_iterations="20",
    cone="pyramidal",
    impratio="100",
)

# Joint dynamics, from Unitree's own MJCF (H2_loop.xml).
JOINT_DYNAMICS = dict(damping="0.05", armature="0.01", frictionloss="0.2")

# Foot contact parameters.
# Group 3 keeps the boxes out of the default render (the viewer shows groups
# 0-2), so they no longer cover the foot meshes; press 3 in the viewer to
# show them. Translucent so the shoe stays visible when they are shown.
# Group and rgba are visual only: contacts depend on contype/conaffinity.
FOOT_GEOM = dict(
    condim="3",
    friction="0.8 0.02 0.01",
    priority="1",
    contype="1",
    conaffinity="1",
    group="3",
    rgba="0.1 0.6 1.0 0.35",
)

# PD gains of the position actuators, (substring of joint name, kp, kd).
# First match wins. Starting values in the style of unitree_rl_gym's H1
# config; they have not been tuned for the H2 and are the first thing to
# revisit if the stand test oscillates or sags.
GAINS = [
    ("hip_yaw", 200.0, 5.0),
    ("hip_roll", 200.0, 5.0),
    ("hip_pitch", 200.0, 5.0),
    ("knee", 300.0, 6.0),
    ("ankle_pitch", 60.0, 2.0),
    ("ankle_roll", 40.0, 2.0),
    ("waist_yaw", 200.0, 5.0),
    ("waist_roll", 300.0, 6.0),
    ("waist_pitch", 300.0, 6.0),
    ("head", 50.0, 1.0),
    ("shoulder", 100.0, 2.0),
    ("elbow", 100.0, 2.0),
    ("wrist", 20.0, 1.0),
]

# Standing pose ("home" keyframe). Joints not listed are zero. Legs slightly
# crouched with hip + knee + ankle summing to zero so the feet stay flat;
# arms slightly out and bent so they clear the hips.
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.3,
    "right_hip_pitch_joint": -0.3,
    "left_knee_joint": 0.6,
    "right_knee_joint": 0.6,
    "left_ankle_pitch_joint": -0.3,
    "right_ankle_pitch_joint": -0.3,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 0.8,
    "right_elbow_joint": 0.8,
}
FOOT_CLEARANCE = 0.002  # gap between sole and floor in the keyframe


def build_id():
    """Short hash of everything that affects the generated model."""
    spec = json.dumps(
        [
            UNITREE_ROS_COMMIT,
            OPTION,
            JOINT_DYNAMICS,
            FOOT_GEOM,
            GAINS,
            DEFAULT_POSE,
            FOOT_CLEARANCE,
        ],
        sort_keys=True,
    )
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


def _log(msg):
    print(f"[vendor_h2] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------


def package_dir():
    from mushroom_rl.environments import mujoco_envs

    return Path(mujoco_envs.__file__).resolve().parent / "data" / "h2"


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
    vendor(target)
    return target / scene


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
            "mushroom_rl.environments.mujoco_warp_envs.h2.vendor_h2` where "
            f"there is network access, or set {ENV_VAR} to an existing copy."
        ) from e


def _fetch(workdir):
    if shutil.which("git") is None:
        raise RuntimeError(
            f"git not found on PATH; needed to fetch the H2 model "
            f"(or set {ENV_VAR} to an existing copy)"
        )
    _log(f"fetching {MODEL_DIR} @ {UNITREE_ROS_COMMIT[:12]} from {UNITREE_ROS_URL}")
    repo = Path(workdir)
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("remote", "add", "origin", UNITREE_ROS_URL, cwd=repo)
    _git("config", "core.sparseCheckout", "true", cwd=repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text(f"/{MODEL_DIR}/\n")
    _git(
        "fetch",
        "-q",
        "--depth",
        "1",
        "--filter=blob:none",
        "origin",
        UNITREE_ROS_COMMIT,
        cwd=repo,
    )
    _git("checkout", "-q", "FETCH_HEAD", cwd=repo)
    src = repo / MODEL_DIR
    if not src.is_dir():
        raise RuntimeError(
            f"{MODEL_DIR} not found in unitree_ros at {UNITREE_ROS_COMMIT}"
        )
    return src


# ----------------------------------------------------------------------
# URDF -> raw MJCF
# ----------------------------------------------------------------------


def _copy_and_patch_urdf(src, dest):
    """Copy the URDF and its meshes; return the patched URDF path."""
    dest.mkdir(parents=True)
    text = (src / URDF_NAME).read_text().replace("\r", "")

    # Only the meshes the URDF references; the folder also carries .dae
    # duplicates and the closed-loop linkage meshes.
    refs = sorted(set(re.findall(r'filename="meshes/([^"]+)"', text)))
    if not refs:
        raise RuntimeError(
            "no mesh references found in H2.urdf; upstream layout changed"
        )
    (dest / ASSET_DIR).mkdir()
    for f in refs:
        shutil.copy2(src / ASSET_DIR / f, dest / ASSET_DIR / f)

    # 1. Floating base. Upstream ships it commented out, which makes MuJoCo
    #    weld the pelvis to the world.
    text = text.replace(
        '<!-- <link name="world"></link>', '<link name="world"></link>', 1
    )
    text = text.replace("</joint> -->", "</joint>", 1)
    fb = text.find('<joint name="floating_base_joint" type="floating">')
    pelvis = text.find('<link name="pelvis">')
    if fb < 0 or pelvis < fb or "<!--" in text[fb:pelvis] or "-->" in text[fb:pelvis]:
        raise RuntimeError(
            "could not enable the floating base joint; upstream URDF changed"
        )

    # 2. Compiler block. Upstream sets meshdir="meshes" while the file names
    #    already carry meshes/, which MuJoCo resolves to meshes/meshes/.
    text, n = re.subn(
        r"<mujoco>.*?</mujoco>",
        '<mujoco><compiler meshdir="meshes" strippath="true" '
        'discardvisual="false" balanceinertia="true"/></mujoco>',
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise RuntimeError("no <mujoco> block in H2.urdf; upstream URDF changed")

    urdf = dest / URDF_NAME
    urdf.write_text(text)
    return urdf


def _urdf_to_mjcf(urdf, out):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(urdf))
    mujoco.mj_saveLastXML(str(out), model)
    return model


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------


def _gains(joint_name):
    for key, kp, kd in GAINS:
        if key in joint_name:
            return kp, kd
    raise RuntimeError(f"no PD gains for joint {joint_name!r}; extend GAINS")


def _fmt(values):
    return " ".join(f"{v:.6g}" for v in values)


def _mesh_aabb(model, mesh_id):
    """Axis-aligned bounding box of a mesh in the frame of the geom using it."""
    import mujoco

    adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    v = model.mesh_vert[adr : adr + num]
    # The compiler re-centres and re-orients every mesh onto its principal
    # axes and stores the offset in mesh_pos / mesh_quat; mesh_vert is in
    # that canonical frame, not in the body frame the geom was declared in.
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.mesh_quat[mesh_id])
    v = v @ rot.reshape(3, 3).T + model.mesh_pos[mesh_id]
    return v.min(axis=0), v.max(axis=0)


def _postprocess(raw_xml, model, out_xml):
    """Turn MuJoCo's URDF export into the locomotion model."""
    import mujoco

    tree = ET.parse(raw_xml)
    root = tree.getroot()
    root.set("model", "h2")

    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    option = ET.Element("option", OPTION)
    root.insert(list(root).index(compiler) + 1, option)

    worldbody = root.find("worldbody")
    pelvis = worldbody.find(f"body[@name='{ROOT_BODY}']")
    if pelvis is None:
        raise RuntimeError(f"root body {ROOT_BODY!r} not found after URDF import")

    # Joints: name the free joint, give the hinges their dynamics.
    hinge_names = []
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("type") == "free":
                joint.set("name", FREE_JOINT)
                continue
            for k, v in JOINT_DYNAMICS.items():
                joint.set(k, v)
            hinge_names.append(joint.get("name"))

    # Collision geometry. Drop every mesh collision geom (the URDF reuses the
    # visual meshes for collision, which is far too much for a batched GPU
    # sim) and add one box per foot from the foot mesh's bounding box. The
    # visual geoms already carry contype="0" conaffinity="0" group="1".
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("contype") != "0":
                body.remove(geom)

    mesh_ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i): i
        for i in range(model.nmesh)
    }
    for geom_name, body_name in FEET_BODIES.items():
        body = worldbody.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"foot body {body_name!r} not found")
        lo, hi = _mesh_aabb(model, mesh_ids[body_name])
        center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        ET.SubElement(
            body,
            "geom",
            dict(
                name=geom_name,
                type="box",
                pos=_fmt(center),
                size=_fmt(half),
                **FOOT_GEOM,
            ),
        )

    # IMU site and sensors on the pelvis.
    ET.SubElement(pelvis, "site", dict(name="imu", pos="0 0 0", size="0.01"))
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", dict(name="imu_gyro", site="imu"))
    ET.SubElement(sensor, "accelerometer", dict(name="imu_acc", site="imu"))
    ET.SubElement(
        sensor, "framequat", dict(name="imu_quat", objtype="site", objname="imu")
    )

    # Position actuators, one per hinge joint, in joint order.
    actuator = ET.SubElement(root, "actuator")
    for name in hinge_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        fmin, fmax = model.jnt_actfrcrange[jid]
        if not model.jnt_actfrclimited[jid]:
            raise RuntimeError(f"joint {name!r} has no effort limit in the URDF")
        kp, kd = _gains(name)
        ET.SubElement(
            actuator,
            "general",
            dict(
                name=name,
                joint=name,
                biastype="affine",
                gainprm=_fmt([kp, 0, 0]),
                biasprm=_fmt([0, -kp, -kd]),
                ctrlrange=_fmt([lo, hi]),
                forcerange=_fmt([fmin, fmax]),
            ),
        )

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
        [
            qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]
            for n in hinge_names
        ]
    )
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", dict(name="home", qpos=_fmt(qpos), ctrl=_fmt(ctrl)))

    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="unicode", xml_declaration=False)
    return hinge_names


def _lowest_foot_point(m, d):
    import mujoco

    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    lowest = np.inf
    for geom_name in FEET_BODIES:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos, mat, half = (
            d.geom_xpos[gid],
            d.geom_xmat[gid].reshape(3, 3),
            m.geom_size[gid],
        )
        pts = pos + (corners * half) @ mat.T
        lowest = min(lowest, pts[:, 2].min())
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
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" condim="3"
      friction="0.8 0.02 0.01" contype="1" conaffinity="1"/>
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
            raise RuntimeError(
                f"{scene}: joint 0 is {joint0!r}, expected free joint {FREE_JOINT!r}"
            )
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home") != 0:
            raise RuntimeError(f"{scene}: keyframe 'home' must be keyframe 0")
        if m.nu != m.njnt - 1:
            raise RuntimeError(
                f"{scene}: {m.nu} actuators for {m.njnt - 1} hinge joints"
            )
        if not (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all():
            raise RuntimeError(f"{scene}: not all actuators are position-type")

        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        for geom_name in FEET_BODIES:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            up = d.geom_xmat[gid].reshape(3, 3)[2, 2]
            dx = d.geom_xpos[gid][0] - d.xpos[1][0]
            if up < 0.98:
                raise RuntimeError(
                    f"{scene}: {geom_name} is not flat in the home pose "
                    f"(z-axis up component {up:.3f}); check DEFAULT_POSE signs"
                )
            if abs(dx) > 0.15:
                raise RuntimeError(
                    f"{scene}: {geom_name} is {dx:+.2f} m from the pelvis "
                    "in x in the home pose; check DEFAULT_POSE signs"
                )
        low = _lowest_foot_point(m, d)
        if abs(low - FOOT_CLEARANCE) > 1e-3:
            raise RuntimeError(
                f"{scene}: lowest foot point at {low:.4f}, expected {FOOT_CLEARANCE}"
            )

        _log(
            f"  {scene}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
            f"mass={m.body_subtreemass[1]:.1f}kg pelvis_z={m.key_qpos[0][2]:.3f}"
        )


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------


def _build(src, dest):
    urdf = _copy_and_patch_urdf(src, dest)
    raw = dest / "h2_raw.xml"
    model = _urdf_to_mjcf(urdf, raw)
    hinge_names = _postprocess(raw, model, dest / "h2.xml")
    raw.unlink()
    (dest / "scene.xml").write_text(SCENE_XML)
    (dest / "PROVENANCE.txt").write_text(
        "Unitree H2 model, converted from the Unitree URDF\n"
        f"Source: {UNITREE_ROS_URL}\n"
        f"Path:   {MODEL_DIR}/{URDF_NAME}\n"
        f"Commit: {UNITREE_ROS_COMMIT}\n"
        f"Build:  {build_id()}\n"
        "\n"
        "Built by mushroom_rl/environments/mujoco_warp_envs/h2/vendor_h2.py.\n"
        "Licensed under the terms of the upstream repository (BSD-3-Clause,\n"
        "Unitree Robotics).\n"
        "\n"
        "Modifications relative to the upstream URDF:\n"
        "  - floating base joint enabled;\n"
        "  - mesh collision geoms removed, one box per foot added;\n"
        "  - hinge joints given damping/armature/frictionloss from H2_loop.xml;\n"
        "  - position actuators with PD gains added (see GAINS in vendor_h2.py);\n"
        "  - IMU site with gyro, accelerometer and framequat sensors;\n"
        "  - 'home' keyframe (see DEFAULT_POSE), height solved numerically.\n"
        f"Actuated joints, in actuator order:\n  " + "\n  ".join(hinge_names) + "\n"
    )


def vendor(dest, force=False):
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _lock(dest.parent / f".{dest.name}.lock"):
        if not force and is_complete(dest):
            return dest

        with tempfile.TemporaryDirectory(
            dir=dest.parent, prefix=f".{dest.name}.tmp-"
        ) as tmp:
            tmp = Path(tmp)
            src = _fetch(tmp / "unitree_ros")
            staged = tmp / "model"
            _build(src, staged)
            _verify(staged)
            if dest.exists():
                os.replace(dest, tmp / "stale")
            os.replace(staged, dest)

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    _log(f"H2 model ready in {dest} ({size / 1e6:.1f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default: first writable lookup location)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if a complete copy exists"
    )
    args = parser.parse_args()

    if args.dest is not None:
        vendor(args.dest, force=args.force)
    elif args.force:
        target = next((d for d in candidate_dirs() if _is_writable(d)), None)
        if target is None:
            sys.exit("no writable location; pass --dest")
        vendor(target, force=True)
    else:
        print(ensure_h2_model().parent)


if __name__ == "__main__":
    main()
