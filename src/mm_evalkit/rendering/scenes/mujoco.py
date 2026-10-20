"""MuJoCo motion view with activation-driven tendon styling."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from musclemimic_models import get_xml_path

from ..models import RenderClip
from ..timeline import normalize_shape


class MujocoScene:
    """Render one model while highlighting dataset-selected tendons."""

    def __init__(self, clip: RenderClip, *, width: int = 640, height: int = 480) -> None:
        self.clip = clip
        self.width = width
        self.height = height
        model_path = get_xml_path(clip.model_name)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if self.model.nq != clip.qpos.shape[1] or self.model.nu != clip.activations.shape[1]:
            self.model = self._reduced_model(model_path)
        self.data = mujoco.MjData(self.model)
        # Keep untextured model backgrounds consistent with the light layout.
        self.model.vis.rgba.haze[:] = 1.0
        if self.model.nq != clip.qpos.shape[1]:
            raise ValueError(
                f"Model {clip.model_name!r} has nq={self.model.nq}, but rollout qpos has {clip.qpos.shape[1]} columns."
            )
        if clip.activations.shape[1] != self.model.nu:
            raise ValueError(
                f"Model {clip.model_name!r} has nu={self.model.nu}, but rollout activation has "
                f"{clip.activations.shape[1]} columns."
            )

        self.model.tendon_rgba[:, 3] = 0.12
        self._tendons: list[tuple[int, tuple[float, float, float], np.ndarray]] = []
        for signal in clip.tendon_signals:
            tendon_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_TENDON,
                signal.tendon_name,
            )
            if tendon_id < 0:
                raise ValueError(f"Model {clip.model_name!r} has no tendon {signal.tendon_name!r}.")
            if not 0 <= signal.activation_index < self.model.nu:
                raise ValueError(f"Invalid activation index {signal.activation_index} for {signal.tendon_name!r}.")
            self._tendons.append(
                (
                    tendon_id,
                    signal.color,
                    normalize_shape(clip.activations[:, signal.activation_index]),
                )
            )

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.data.qpos[:] = clip.qpos[len(clip.qpos) // 2]
        mujoco.mj_forward(self.model, self.data)
        body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in clip.camera.body_names]
        missing = [name for name, body_id in zip(clip.camera.body_names, body_ids, strict=True) if body_id < 0]
        if missing:
            raise ValueError(f"Model {clip.model_name!r} has no camera body/bodies: {', '.join(missing)}.")
        self.camera.lookat[:] = self.data.xpos[body_ids].mean(axis=0)
        self.camera.distance = clip.camera.distance
        self.camera.azimuth = clip.camera.azimuth
        self.camera.elevation = clip.camera.elevation

        self.options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.options)
        self.options.sitegroup[:] = 0
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

    def _reduced_model(self, model_path) -> mujoco.MjModel:
        """Weld unexported joints and remove unexported actuators via MjSpec."""
        if self.clip.joint_names is None or self.clip.activation_names is None:
            raise ValueError(
                f"Model {self.clip.model_name!r} dimensions do not match this rollout, "
                "and the clip does not declare the exported joint/activation names."
            )
        spec = mujoco.MjSpec.from_file(str(model_path))
        retained_joints = set(self.clip.joint_names)
        retained_activations = set(self.clip.activation_names)
        for actuator in list(spec.actuators):
            if actuator.name not in retained_activations:
                spec.delete(actuator)
        for joint in list(spec.joints):
            if joint.name not in retained_joints:
                spec.delete(joint)
        model = spec.compile()
        joint_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt))
        activation_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
        )
        if joint_names != self.clip.joint_names:
            raise ValueError("Reduced model joint order does not match the rollout declaration.")
        if activation_names != self.clip.activation_names:
            raise ValueError("Reduced model actuator order does not match the rollout declaration.")
        return model

    def render(self, qpos: np.ndarray, sample_position: float) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        low = min(int(np.floor(sample_position)), len(self.clip.qpos) - 1)
        high = min(low + 1, len(self.clip.qpos) - 1)
        blend = sample_position - low
        for tendon_id, color, values in self._tendons:
            level = float((1.0 - blend) * values[low] + blend * values[high])
            self.model.tendon_rgba[tendon_id, :3] = np.asarray(color) * (0.28 + 0.72 * level)
            self.model.tendon_rgba[tendon_id, 3] = 0.30 + 0.70 * level
            self.model.tendon_width[tendon_id] = 0.002 + 0.010 * level
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
