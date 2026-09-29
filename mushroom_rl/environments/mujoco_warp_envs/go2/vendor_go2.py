#!/usr/bin/env python3
"""
Fetch the Unitree Go2 model from mujoco_menagerie on demand.

The model is not committed to the repository. Go2Base calls
ensure_go2_model() at construction; the first call on a machine fetches the
pinned menagerie commit, applies the local patches, compiles the result and
moves it into place. Later calls find the model and return immediately.

Lookup order:
    1. $MUSHROOM_RL_GO2_DIR, if set (used exclusively);
    2. mushroom_rl/environments/mujoco_envs/data/go2 (the package data dir);
    3. $XDG_CACHE_HOME/mushroom_rl/go2/<commit> (default ~/.cache/...).
If none holds a complete copy, the model is fetched into the first writable
candidate. A copy counts as complete only if every model file is present and
its PROVENANCE.txt names the pinned commit, so bumping MENAGERIE_COMMIT
invalidates old copies automatically.

Compute nodes without internet: pre-fetch once on a machine that has it,

    python -m mushroom_rl.environments.mujoco_warp_envs.go2.vendor_go2

or point $MUSHROOM_RL_GO2_DIR at a copy on shared storage.

"""

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no inter-process lock, single process assumed
    fcntl = None


MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"

# Pinned upstream commit. The patches below are exact string replacements
# against this revision; bump deliberately and re-run the verification.
MENAGERIE_COMMIT = "e4049d0a3bfd58d2a3081614e6777d4007e3f86a"

MODEL_DIR = "unitree_go2"
ENV_VAR = "MUSHROOM_RL_GO2_DIR"

MODEL_FILES = ["go2.xml", "go2_mjx.xml", "scene.xml", "scene_mjx.xml"]
DOC_FILES = ["LICENSE", "README.md"]
ASSET_DIR = "assets"

# (file, old, new, expected number of occurrences). A count mismatch means
# upstream changed under us, and the fetch fails instead of silently
# producing an unpatched model.
PATCHES = [
    ("go2.xml", "<freejoint/>", '<freejoint name="base"/>', 1),
    ("go2_mjx.xml", "<freejoint/>", '<freejoint name="base"/>', 1),
    ("go2_mjx.xml", ' iterations="1"', ' iterations="8"', 1),
    ("go2_mjx.xml", 'ls_iterations="5"', 'ls_iterations="20"', 1),
]

PROVENANCE_TEMPLATE = """\
Unitree Go2 model
Source: {url}
Path:   {path}
Commit: {sha}

Fetched by mushroom_rl/environments/mujoco_warp_envs/go2/vendor_go2.py.
Licensed under the terms in LICENSE (BSD-3-Clause, Unitree Robotics).

Local modifications:
  - <freejoint/> renamed to <freejoint name="base"/> in go2.xml and
    go2_mjx.xml, so the floating base can be referenced by name.
  - go2_mjx.xml solver iterations 1 -> 8 and ls_iterations 5 -> 20, to
    silence per-step convergence warnings.
"""


def _log(msg):
    print(f"[vendor_go2] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------


def package_dir():
    # Resolved through the package rather than relative to this file, so it
    # survives moving this module into the locomotion folder.
    from mushroom_rl.environments import mujoco_envs

    return Path(mujoco_envs.__file__).resolve().parent / "data" / "go2"


def cache_dir():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "mushroom_rl" / "go2" / MENAGERIE_COMMIT[:12]


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
    return prov.is_file() and MENAGERIE_COMMIT in prov.read_text()


def _is_writable(d):
    # The directory may not exist yet: check the closest existing ancestor.
    p = Path(d)
    while not p.exists():
        p = p.parent
    return os.access(p, os.W_OK)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def ensure_go2_model(scene="scene_mjx.xml"):
    """
    Return the path of the requested Go2 scene file, fetching the model first
    if no complete copy exists.

    Args:
        scene (str): one of the scene files in MODEL_FILES.

    Returns:
        pathlib.Path of the scene file.

    """
    if scene not in MODEL_FILES:
        raise ValueError(f"unknown Go2 scene {scene!r}, expected one of {MODEL_FILES}")

    candidates = candidate_dirs()
    for d in candidates:
        if is_complete(d):
            return d / scene

    target = next((d for d in candidates if _is_writable(d)), None)
    if target is None:
        raise RuntimeError(
            "Go2 model not found and no writable location to fetch it into. "
            f"Tried: {', '.join(map(str, candidates))}. "
            f"Set {ENV_VAR} to a writable directory."
        )

    vendor(target)
    return target / scene


# ----------------------------------------------------------------------
# Fetch, patch, verify
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
            "Fetching the Go2 model needs network access to github.com. On an "
            "offline machine, run `python -m "
            "mushroom_rl.environments.mujoco_warp_envs.go2.vendor_go2` where "
            f"there is network access, or set {ENV_VAR} to an existing copy."
        ) from e


def _fetch(workdir):
    """Fetch only MODEL_DIR at the pinned commit. Returns the source directory."""
    if shutil.which("git") is None:
        raise RuntimeError(
            f"git not found on PATH; needed to fetch the Go2 model "
            f"(or set {ENV_VAR} to an existing copy)"
        )

    _log(f"fetching {MODEL_DIR} @ {MENAGERIE_COMMIT[:12]} from {MENAGERIE_URL}")
    repo = Path(workdir)
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("remote", "add", "origin", MENAGERIE_URL, cwd=repo)
    # Plain sparse-checkout pattern file instead of `git sparse-checkout`,
    # which behaves differently across git versions on an empty repo.
    _git("config", "core.sparseCheckout", "true", cwd=repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text(f"/{MODEL_DIR}/\n")
    # Fetching a commit by sha works on GitHub and, unlike clone --depth 1 of
    # a branch, lets the pin be an arbitrary historical commit.
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
    if not src.is_dir():
        raise RuntimeError(f"{MODEL_DIR} not found in menagerie at {MENAGERIE_COMMIT}")
    return src


def _copy(src, dest):
    dest.mkdir(parents=True)
    for fname in MODEL_FILES + DOC_FILES:
        f = src / fname
        if not f.is_file():
            raise RuntimeError(f"upstream file missing: {MODEL_DIR}/{fname}")
        shutil.copy2(f, dest / fname)
    shutil.copytree(src / ASSET_DIR, dest / ASSET_DIR)


def _patch(dest):
    for fname, old, new, expected in PATCHES:
        path = dest / fname
        text = path.read_text()
        found = text.count(old)
        if found != expected:
            raise RuntimeError(
                f"patch {old!r} -> {new!r} on {fname}: expected {expected} "
                f"occurrence(s), found {found}. Upstream changed; update PATCHES."
            )
        path.write_text(text.replace(old, new))


def _verify(dest):
    """Compile both scenes, so a broken copy fails here and not at training time."""
    import mujoco

    for scene in ("scene.xml", "scene_mjx.xml"):
        model = mujoco.MjModel.from_xml_path(str(dest / scene))
        joint0 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0)
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if joint0 != "base":
            raise RuntimeError(f"{scene}: free joint is {joint0!r}, expected 'base'")
        if home != 0:
            raise RuntimeError(
                f"{scene}: keyframe 'home' must be keyframe 0, got id {home}"
            )
        if model.nu != 12:
            raise RuntimeError(f"{scene}: expected 12 actuators, got {model.nu}")
        _log(
            f"  {scene}: nq={model.nq} nv={model.nv} nu={model.nu} "
            f"iterations={model.opt.iterations} ls={model.opt.ls_iterations}"
        )


def vendor(dest, force=False):
    """
    Fetch, patch and verify the model into dest.

    The model is built in a temporary directory next to dest and moved into
    place with a rename, under a file lock, so parallel processes (a seed
    sweep, several training jobs on one node) neither race nor see a
    half-written copy.

    """
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _lock(dest.parent / f".{dest.name}.lock"):
        if not force and is_complete(dest):
            return dest  # another process finished while we waited

        with tempfile.TemporaryDirectory(
            dir=dest.parent, prefix=f".{dest.name}.tmp-"
        ) as tmp:
            tmp = Path(tmp)
            src = _fetch(tmp / "menagerie")
            staged = tmp / "model"
            _copy(src, staged)
            _patch(staged)
            (staged / "PROVENANCE.txt").write_text(
                PROVENANCE_TEMPLATE.format(
                    url=MENAGERIE_URL, path=MODEL_DIR, sha=MENAGERIE_COMMIT
                )
            )
            _verify(staged)

            if dest.exists():
                os.replace(dest, tmp / "stale")  # removed with tmp
            os.replace(staged, dest)

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    _log(f"Go2 model ready in {dest} ({size / 1e6:.1f} MB)")
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
        "--force", action="store_true", help="re-fetch even if a complete copy exists"
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
        print(ensure_go2_model().parent)


if __name__ == "__main__":
    main()
