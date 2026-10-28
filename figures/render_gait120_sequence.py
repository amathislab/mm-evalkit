"""Blender lateral gait poses from an actual exported MyoFullBody rollout.

Three strips are rendered, one per column of ``plot_gait120_comparison.py``:

* ``joints`` -- bones with hip, knee, ankle, subtalar and MTP landmarks over
  one right stride;
* ``kinetics`` -- bones with the mean vertical GRF as an arrow, over the
  analysis cache's kinetics window (by default 1.00 s from left heel strike;
  the same right stride with ``right_stride``);
* ``muscles`` -- bones with VL, BF, TA and GM paths coloured by mean muscle
  signal over the right stride.

Poses are one policy rollout, because there is no human skeleton to render.
The overlays are averaged profiles read from the figure's analysis cache
(``--analysis_cache``): kinetics and muscle strips are rendered once per
overlay set, the Gait120 subject average and each policy (see ``overlays``).
Stance bars mark where mean Gait120 vertical GRF exceeds 0.05 BW: over the
kinetics window for the kinetics strip, and over the force-plate right stride
of the cohort's references for the stride strips. Root translation is
removed for an in-place sequence; articulated joint positions are preserved.
The manifest records each pose's hip projection, so the figure can place every
pose exactly on its phase tick.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import subprocess
import sys
from pathlib import Path

PHASES = [0, 20, 40, 60, 80, 100]
HIGHLIGHTS = ["vaslat_r", "bflh_r", "tibant_r", "gasmed_r"]
# ``--extended`` exports every Gait120 EMG muscle and renders one muscle strip
# per group, so each strip shows a readable handful of paths.
MUSCLE_GROUPS = {
    "thigh": ["vaslat_r", "vasmed_r", "recfem_r", "bflh_r", "semiten_r"],
    "shank": ["tibant_r", "gasmed_r", "gaslat_r", "soleus_r", "perlong_r", "perbrev_r"],
}
JOINTS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r"]
# Landmark radius [m]: the foot joints sit a few centimetres apart.
JOINT_RADIUS = {"hip_flexion_r": 0.018, "knee_angle_r": 0.018, "ankle_angle_r": 0.015,
                "subtalar_angle_r": 0.012, "mtp_angle_r": 0.012}
FOOT_BODIES = {"calcn_r", "toes_r"}
# Where each label points, as a fraction of the path from origin to insertion:
# the muscle belly rather than a tendon.
BELLY = {"vaslat_r": 0.45, "bflh_r": 0.5, "tibant_r": 0.3, "gasmed_r": 0.25, "vasmed_r": 0.62, "recfem_r": 0.4,
         "semiten_r": 0.45, "gaslat_r": 0.3, "soleus_r": 0.45, "perlong_r": 0.35, "perbrev_r": 0.5}
# Muscle colour ramp, endpoints in sRGB. The figure draws its colour bar from
# the same two values. The low end stays a clear salmon so a silent muscle
# still separates from the cool grey bone.
EXCITATION_LOW = (0.98, 0.80, 0.68)
EXCITATION_HIGH = (0.62, 0.06, 0.04)
GRF_METRES_PER_BW = 0.32
STANCE_THRESHOLD_BW = 0.05
RESOLUTION = (900, 1100)
ORTHO_SCALE = 1.25


def _grf_profile(grf, *, cyclic: bool):
    """Phase and finite GRF samples; a stride is closed with its 0 % value."""
    import numpy as np

    phase = np.linspace(0, 100, 100)
    finite = np.isfinite(grf)
    phase_grf, values = phase[finite], grf[finite]
    # A stride's mean GRF is undefined at its very end (references end early,
    # rollouts earlier still). 100 % is the next heel strike, so it takes the
    # 0 % value; a fixed time window has no such closure.
    if cyclic and phase_grf[-1] < 100.0:
        phase_grf, values = np.r_[phase_grf, 100.0], np.r_[values, grf[0]]
    return phase_grf, values, phase[finite][-1]


def _stance(grf, *, cyclic: bool) -> list:
    """Contact per 1 % of the window, None where GRF was never measured."""
    import numpy as np

    phase_grf, values, last = _grf_profile(grf, cyclic=cyclic)
    grid = np.linspace(0, 100, 101)
    contact = np.interp(grid, phase_grf, values) > STANCE_THRESHOLD_BW
    return [bool(c) if g <= last + 1e-9 else None for c, g in zip(contact, grid, strict=True)]


def _overlay(grf, muscle, muscle_names, *, label, colour_label, peak_scaled, cyclic, highlights=HIGHLIGHTS) -> dict:
    """GRF, muscle colour and stance of one averaged profile, at each pose phase."""
    import numpy as np

    phase = np.linspace(0, 100, 100)
    phase_grf, grf_values, _ = _grf_profile(grf, cyclic=cyclic)
    colour = {}
    for name in highlights:
        profile = muscle[muscle_names.index(name)]
        colour[name] = profile / np.nanmax(profile) if peak_scaled else profile
    return {
        "label": label, "colour_label": colour_label,
        "grf_bw": {str(p): float(np.interp(p, phase_grf, grf_values)) for p in PHASES},
        "muscle_colour": {str(p): {n: float(np.interp(p, phase, c)) for n, c in colour.items()} for p in PHASES},
        "stance": _stance(grf, cyclic=cyclic),
    }


def load_analysis(analysis_cache: Path) -> dict:
    # Written by plot_gait120_comparison.py; a local, trusted file.
    return pickle.loads(analysis_cache.read_bytes())


def cycle_stance(data: dict, reference_dir: Path) -> list:
    """Mean right-stride stance of the cohort, from its force-plate references."""
    import numpy as np

    from mm_evalkit.datasets.gait120_kinetics import load_gait120_kinetics_trial

    strides = []
    for subject in data["subjects"]:
        path = reference_dir / f"S{subject:03d}" / f"S{subject:03d}_Trial05_Step01_Step02.npz"
        trial = load_gait120_kinetics_trial(path, window="right_stride")
        strides.append(trial.grf_cycles[0, trial.grf_names.index("right")])
    return _stance(np.nanmean(strides, axis=0), cyclic=True)


def overlays(data: dict, highlights=HIGHLIGHTS) -> dict:
    """Every overlay set the figures draw on the poses, keyed by render name.

    ``gait120`` is the main figure's: subject-averaged GRF, and EMG scaled to
    each muscle's peak so colour shows timing. The ``compare_*`` sets feed the
    human-versus-policy figure and share one absolute 0-1 scale (MVC-normalized
    EMG and excitation, as on the waveform axes), because a near-silent policy
    muscle scaled to its own peak would turn noise into full colour.
    """
    import numpy as np

    cyclic = data.get("kinetics_window", "left_heel_strike") == "right_stride"
    right = data["grf_names"].index("right")
    names = data["muscle_names"]
    signal = data.get("muscle_signal", "activation")
    human_grf = np.nanmean(data["human_grf"][:, right], axis=0)
    human_emg = np.nanmean(data["human_muscle"], axis=0)
    sets = {
        "gait120": _overlay(human_grf, human_emg, names, label=f"Gait120 mean (n = {len(data['subjects'])})",
                            colour_label="Mean EMG (of peak)", peak_scaled=True, cyclic=cyclic,
                            highlights=highlights),
        "compare_gait120": _overlay(human_grf, human_emg, names,
                                    label=f"Gait120 mean (n = {len(data['subjects'])})",
                                    colour_label="MVC-norm. EMG", peak_scaled=False, cyclic=cyclic,
                                    highlights=highlights),
    }
    for run in data["policies"]:
        sets[f"compare_{run['id']}"] = _overlay(
            run["avg_grf"][right], run["avg_muscle"], names, label=run["label"],
            colour_label=signal.capitalize(), peak_scaled=False, cyclic=cyclic, highlights=highlights,
        )
    return sets


def export_sequence(rollout: Path, output: Path, episode: int, analysis_cache: Path, reference_dir: Path,
                    *, extended: bool = False) -> Path:
    import mujoco
    import numpy as np
    from musclemimic_models import get_xml_path

    from mm_evalkit.datasets.gait120_kinetics import load_gait120_kinetics_trial
    from mm_evalkit.workflows.gait.constants import MOTION_SETS
    from mm_evalkit.workflows.gait.cycles import extract_valid_strides_flexible, get_feet_contacts
    from mm_evalkit.workflows.gait.kinetics_data import _rollout_reference

    analysis = load_analysis(analysis_cache)
    highlights = [name for group in MUSCLE_GROUPS.values() for name in group] if extended else HIGHLIGHTS
    window_kind = analysis.get("kinetics_window", "left_heel_strike")

    model = mujoco.MjModel.from_xml_path(str(get_xml_path("myofullbody")))
    data = mujoco.MjData(model)
    with np.load(rollout, allow_pickle=True) as archive:
        prefix = f"episode_{episode}"
        qpos = archive[f"{prefix}_joint_positions"]
        names = archive["joint_names"].tolist()
        contacts = get_feet_contacts(archive[f"{prefix}_touch_observations"], threshold=0.05)
        strides = extract_valid_strides_flexible(contacts, side="right", expected_n=2)
        if not strides:
            raise ValueError("The selected episode has no retained right-leg stride.")
        start, end = int(strides[0].start), int(strides[0].end)
        cycle_frames = np.rint(start + (end - start) * np.asarray(PHASES) / 100).astype(int)
        motion = str(archive["motion_names"][int(archive[f"{prefix}_traj_id"])])
        subject = re.search(r"/(S\d{3})/", motion)
        subject = subject.group(1) if subject else motion
        if window_kind == "right_stride":
            kinetics_frames = cycle_frames
        else:
            # The kinetics strip follows the cache's fixed window on the
            # exemplar's own reference trial, mapped onto rollout frames.
            motion_list = MOTION_SETS[str(archive["motion_group"])]
            _, reference = _rollout_reference(int(archive[f"{prefix}_traj_id"]), motion_list, reference_dir)
            w0, w1 = load_gait120_kinetics_trial(reference).gait_windows[0]
            steps = archive[f"{prefix}_kinetics_reference_steps"]
            kinetics_frames = np.array([
                int(np.flatnonzero(steps == frame)[0])
                for frame in np.rint(w0 + (w1 - 1 - w0) * np.asarray(PHASES) / 100).astype(int)
            ])

        # Resolve the exported joint ordering against the complete installed model.
        addresses = []
        offset = 0
        for name in names:
            joint = model.joint(name)
            width = 7 if model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_FREE else 1
            addresses.append((int(joint.qposadr[0]), offset, width))
            offset += width
        if offset != qpos.shape[1]:
            raise ValueError("Unsupported rollout joint layout.")

        bones = []
        for i in range(model.ngeom):
            name = model.geom(i).name
            body = model.body(model.geom_bodyid[i]).name
            if (model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and name.startswith("r_")
                    and body in {"pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "patella_r"}
                    and model.geom_group[i] == 0):
                mesh = int(model.geom_dataid[i])
                va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                bones.append({"id": i, "name": name, "foot": body in FOOT_BODIES,
                              "vertices": model.mesh_vert[va:va + vn].tolist(),
                              "faces": model.mesh_face[fa:fa + fn].tolist()})
        frames = {}
        for frame in sorted(set(cycle_frames.tolist()) | set(kinetics_frames.tolist())):
            data.qpos[:] = model.qpos0
            for address, source, width in addresses:
                data.qpos[address:address + width] = qpos[frame, source:source + width]
            # Keep root height and orientation; remove forward/lateral drift.
            data.qpos[:2] = 0
            mujoco.mj_forward(model, data)
            geometry = []
            foot = []
            for bone in bones:
                index = bone["id"]
                vertices = np.asarray(bone["vertices"]) @ data.geom_xmat[index].reshape(3, 3).T
                vertices += data.geom_xpos[index]
                geometry.append({"name": bone["name"], "vertices": vertices.tolist(), "faces": bone["faces"]})
                if bone["foot"]:
                    foot.append(vertices)
            foot = np.concatenate(foot)
            # The export has no centre of pressure; the foot vertices within
            # 1.5 cm of the lowest one stand in for the contact patch.
            patch = foot[foot[:, 2] < foot[:, 2].min() + 0.015]
            paths = {}
            for depth, name in enumerate(highlights):
                tendon = model.tendon(f"{name}_tendon").id
                a, n = int(data.ten_wrapadr[tendon]), int(data.ten_wrapnum[tendon])
                points = data.wrap_xpos.reshape(-1, 3)[a:a + n].copy()
                # Bring medial paths to the lateral display plane so all four
                # are visible. Their sagittal (y,z) coordinates are unchanged.
                # A small per-muscle depth step keeps overlapping paths from z-fighting.
                points[:, 0] = -0.21 - 0.004 * depth
                paths[name] = points.tolist()
            joints = {}
            for name in JOINTS:
                point = data.xanchor[model.joint(name).id].copy()
                point[0] = -0.23
                joints[name] = point.tolist()
            origin = patch.mean(axis=0)
            origin[0] = -0.25
            frames[str(frame)] = {
                "bones": geometry, "paths": paths, "joints": joints,
                "grf_origin": origin.tolist(),
            }
    payload = {
        "rollout": str(rollout.resolve()), "episode": episode, "subject": subject,
        "cycle_frames": cycle_frames.tolist(), "cycle_window": [start, end],
        "kinetics_frames": kinetics_frames.tolist(), "kinetics_window": window_kind,
        "phases": PHASES, "frames": frames, "overlays": overlays(analysis, highlights),
        "muscle_groups": MUSCLE_GROUPS if extended else None,
        "cycle_stance": cycle_stance(analysis, reference_dir),
        "muscle_colour_ramp": [EXCITATION_LOW, EXCITATION_HIGH], "analysis_cache": str(analysis_cache.resolve()),
        "grf_metres_per_bw": GRF_METRES_PER_BW, "resolution": list(RESOLUTION), "ortho_scale": ORTHO_SCALE,
        "note": "Actual policy poses, root translation removed. Highlighted muscle paths are projected "
                "onto the lateral plane for visibility; path thickness is illustrative. Overlays are averaged "
                "profiles (see overlays): vertical GRF arrows from the lowest foot vertices, muscle colour, and "
                "stance where mean vertical GRF > 0.05 BW.",
    }
    path = output / "sequence_geometry.json"
    path.write_text(json.dumps(payload))
    return path


def render_sequence(path: Path) -> None:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    payload = json.loads(path.read_text())
    output = path.parent
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = payload["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGBA"
    scene.world.color = (0.8, 0.8, 0.8)
    scene.view_settings.view_transform = "Standard"

    def linear(rgb):
        return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)

    def material(name, rgb, roughness=0.65):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*linear(rgb), 1)
        shader.inputs["Roughness"].default_value = roughness
        return mat

    bone_mat = material("Grey bone", (0.85, 0.85, 0.84))
    joint_mat = material("Joint landmarks", (0.11, 0.11, 0.10))
    grf_mat = material("Ground reaction force", (0.11, 0.11, 0.10), roughness=0.4)
    low, high = payload["muscle_colour_ramp"]
    ramp = [material(f"Excitation {i}", [a + (b - a) * i / 20 for a, b in zip(low, high, strict=True)])
            for i in range(21)]
    for name, location, energy, size in [
        ("Key", (-3, -4, 4), 380, 4), ("Fill", (-2, 4, 3), 230, 3),
    ]:
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 0.5)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    # Freestyle outlines: dark on muscles, so overlapping paths in the shared
    # display plane read as separate strands, and faint on bones and landmarks.
    collections = {}
    for name in ("Bones", "Muscles"):
        collections[name] = bpy.data.collections.new(name)
        scene.collection.children.link(collections[name])
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = "ABSOLUTE"
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True
    settings = view_layer.freestyle_settings
    for lineset in settings.linesets:
        lineset.show_render = False
    for name, rgb, alpha, thickness in [("Bones", (0.45, 0.45, 0.44), 0.7, 1.1),
                                        ("Muscles", (0.18, 0.03, 0.02), 1.0, 2.2)]:
        lineset = settings.linesets.new(name)
        lineset.select_by_collection = True
        lineset.collection = collections[name]
        lineset.select_silhouette, lineset.select_border = True, True
        lineset.select_crease = False
        lineset.linestyle.color = linear(rgb)
        lineset.linestyle.alpha = alpha
        lineset.linestyle.thickness = thickness
    camera_data = bpy.data.cameras.new("True lateral · orthographic")
    camera = bpy.data.objects.new("True lateral · orthographic", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (-4, 0, 0.55)
    camera.rotation_euler = (Vector((0, 0, 0.55)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = payload["ortho_scale"]
    generated = []

    def link(obj, mat, collection="Bones"):
        collections[collection].objects.link(obj)
        obj.data.materials.append(mat)
        generated.append(obj)

    def tube(name, points, radius, mat):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        curve.bevel_resolution = 3
        spline = curve.splines.new("POLY")
        coords = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            for i in range(10):
                coords.append([x + (y - x) * i / 10 for x, y in zip(a, b, strict=True)])
        coords.append(points[-1])
        spline.points.add(len(coords) - 1)
        for i, (point, xyz) in enumerate(zip(spline.points, coords, strict=True)):
            point.co = (*xyz, 1)
            point.radius = 0.22 + 0.78 * math.sin(math.pi * i / (len(coords) - 1)) ** 0.7
        link(bpy.data.objects.new(name, curve), mat, "Muscles")

    def arrow(origin, force_bw):
        vector = Vector(force_bw) * payload["grf_metres_per_bw"]
        if vector.length < 0.02:
            return
        head = min(0.05, 0.45 * vector.length)
        direction = vector.normalized()
        rotation = direction.to_track_quat("Z", "Y").to_euler()
        shaft = vector.length - head
        bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=shaft, vertices=24,
                                            location=Vector(origin) + direction * shaft / 2, rotation=rotation)
        bpy.context.object.data.materials.append(grf_mat)
        generated.append(bpy.context.object)
        bpy.ops.mesh.primitive_cone_add(radius1=0.022, radius2=0, depth=head, vertices=24,
                                        location=Vector(origin) + direction * (shaft + head / 2), rotation=rotation)
        bpy.context.object.data.materials.append(grf_mat)
        generated.append(bpy.context.object)

    def along_path(points, fraction):
        """Point at ``fraction`` of a path's length from its origin."""
        lengths = [(Vector(b) - Vector(a)).length for a, b in zip(points[:-1], points[1:], strict=True)]
        remaining = fraction * sum(lengths)
        for a, b, length in zip(points[:-1], points[1:], lengths, strict=True):
            if remaining <= length:
                return Vector(a).lerp(Vector(b), remaining / length if length else 0.0)
            remaining -= length
        return Vector(points[-1])

    def project(xyz):
        point = world_to_camera_view(scene, camera, Vector(xyz))
        return [point.x, point.y]

    projections = {}
    strips = [("joints", "joints", None, None)]
    if payload.get("muscle_groups"):
        overlay = payload["overlays"]["gait120"]
        strips.append(("kinetics_gait120", "kinetics", overlay, None))
        strips += [(f"muscles-{group}_gait120", "muscles", overlay, names)
                   for group, names in payload["muscle_groups"].items()]
    else:
        for key, overlay in payload["overlays"].items():
            strips += [(f"kinetics_{key}", "kinetics", overlay, None), (f"muscles_{key}", "muscles", overlay, None)]
    for strip, category, overlay, shown in strips:
        projections[strip] = []
        frames = payload["kinetics_frames"] if category == "kinetics" else payload["cycle_frames"]
        for phase, frame in zip(payload["phases"], frames, strict=True):
            for obj in generated:
                bpy.data.objects.remove(obj, do_unlink=True)
            generated.clear()
            data = payload["frames"][str(frame)]
            for bone in data["bones"]:
                mesh = bpy.data.meshes.new(bone["name"])
                mesh.from_pydata(bone["vertices"], [], bone["faces"])
                mesh.update()
                for poly in mesh.polygons:
                    poly.use_smooth = True
                link(bpy.data.objects.new(bone["name"], mesh), bone_mat)
            if category == "muscles":
                for name, points in data["paths"].items():
                    if shown is not None and name not in shown:
                        continue
                    level = overlay["muscle_colour"][str(phase)][name]
                    tube(name, points, 0.010, ramp[round(20 * min(max(level, 0.0), 1.0))])
            else:
                for name, point in data["joints"].items():
                    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=JOINT_RADIUS[name],
                                                         location=point)
                    obj = bpy.context.object
                    obj.name = name
                    obj.data.materials.append(joint_mat)
                    generated.append(obj)
            if category == "kinetics":
                arrow(data["grf_origin"], (0.0, 0.0, overlay["grf_bw"][str(phase)]))
            bpy.context.view_layer.update()
            projected = {"hip": project(data["joints"]["hip_flexion_r"])}
            for name, point in data["joints"].items():
                projected[name] = project(point)
            for name, points in data["paths"].items():
                projected[name] = project(along_path(points, BELLY[name]))
            projections[strip].append(projected)
            scene.render.filepath = str(output / f"{strip}_{phase:03d}.png")
            bpy.ops.render.render(write_still=True)
    manifest = {k: v for k, v in payload.items() if k != "frames"}
    manifest["projections"] = projections
    (output / "sequence.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "gait120_sequence.blend"))


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--analysis_cache", type=Path,
                        help="gait120_comparison_cache.pkl holding the human averages for the overlays.")
    parser.add_argument("--kinetics_reference_dir", type=Path,
                        help="Gait120 kinetics references: the exemplar's window and the cohort's stance.")
    parser.add_argument("--output_dir", type=Path, default=Path("assets/gait120_sequence"))
    parser.add_argument("--blender-scene", type=Path)
    parser.add_argument("--extended", action="store_true",
                        help="Export all eleven EMG muscles and render one Gait120 muscle strip per MUSCLE_GROUPS entry.")
    args = parser.parse_args(argv)
    if args.blender_scene:
        render_sequence(args.blender_scene)
        return
    if args.rollout is None or args.analysis_cache is None or args.kinetics_reference_dir is None:
        parser.error("--rollout, --analysis_cache and --kinetics_reference_dir are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = export_sequence(args.rollout, output, args.episode, args.analysis_cache,
                           args.kinetics_reference_dir.resolve(), extended=args.extended)
    subprocess.run([args.blender, "--background", "--threads", "8", "--python", str(Path(__file__).resolve()),
                    "--", "--blender-scene", str(path)], check=True)


if __name__ == "__main__":
    main()
