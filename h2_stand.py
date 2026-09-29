"""
Passive stand test for the generated H2 model, in plain MuJoCo (no warp,
no policy). Holds the keyframe pose with the position actuators and reports
what happens. Use it after changing GAINS or DEFAULT_POSE in vendor_h2.py.

What to expect from a biped: the PD holds the joints, but nothing balances
the body, and an inverted pendulum on a PD ankle is unstable unless the
ankle stiffness exceeds m*g*h (about 700 Nm/rad here, far above what a
67 Nm ankle can deliver). So the robot stands for roughly a second and then
topples slowly. That is the healthy outcome. The failures this test is for
are: NaN, joints that jitter or explode, sinking into the floor, joints
sagging far from the keyframe, and toppling within a few hundred ms.

    python h2_stand_test.py                 # model resolved like the env does
    python h2_stand_test.py --dir /some/h2  # a specific build

"""

import argparse

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=None, help="directory holding scene.xml")
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args()

    if args.dir is None:
        from mushroom_rl.environments.mujoco_warp_envs.h2.vendor_h2 import (
            ensure_h2_model,
        )

        scene = ensure_h2_model("scene.xml")
    else:
        scene = f"{args.dir}/scene.xml"

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    mujoco.mj_forward(m, d)
    z0 = d.qpos[2]
    print(
        f"mass {m.body_subtreemass[1]:.1f} kg, keyframe pelvis z {z0:.3f}, "
        f"com {d.subtree_com[1].round(3)}"
    )

    n = int(args.seconds / m.opt.timestep)
    toppled_at = None
    max_dev = 0.0
    for t in range(n):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            print(f"NaN at t={t * m.opt.timestep:.3f} s")
            return
        up = d.xmat[1].reshape(3, 3)[2, 2]
        if toppled_at is None and up < 0.7:
            toppled_at = t * m.opt.timestep
        max_dev = max(max_dev, np.abs(d.qpos[7:] - m.key_qpos[0][7:]).max())
        if (t + 1) % int(0.25 / m.opt.timestep) == 0:
            f = np.zeros(6)
            normal = sum(
                (mujoco.mj_contactForce(m, d, i, f), f[0])[1] for i in range(d.ncon)
            )
            print(
                f"t={(t + 1) * m.opt.timestep:5.2f}s z={d.qpos[2]:.3f} up={up:+.3f} "
                f"ncon={d.ncon} normal_force={normal:6.0f} N max_joint_dev={max_dev:.3f} rad"
            )
        if toppled_at is not None and up < 0.0:
            break

    if toppled_at is None:
        print(f"stayed upright for {args.seconds} s")
    else:
        print(
            f"toppled at t={toppled_at:.2f} s (expected for a passive PD biped; "
            "worry if it is under ~0.3 s or if the joints jitter)"
        )


if __name__ == "__main__":
    main()
