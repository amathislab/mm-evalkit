"""Render the actual MyoFullBody bones and muscle paths in Blender.

Run with the evaluation Python environment (MuJoCo and musclemimic_models):
    python figures/render_gait120_anatomy.py --blender /path/to/blender \
        --output_dir assets/gait120_anatomy

The exported scene contains model mesh vertices and evaluated tendon paths,
not an artist's replacement anatomy. Path thickness is illustrative. Blender
writes three transparent views, projection coordinates for vector labels, and
an editable .blend file. No simulation or evaluation results are changed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HIGHLIGHTS = {
    "vaslat_r": {"short": "VL", "label": "Vastus lateralis", "view": "anterior"},
    "tibant_r": {"short": "TA", "label": "Tibialis anterior", "view": "anterior"},
    "bflh_r": {"short": "BF", "label": "Biceps femoris (long head)", "view": "posterior"},
    "gasmed_r": {"short": "GM", "label": "Gastrocnemius medialis", "view": "posterior"},
}


def export_model(output_dir: Path) -> Path:
    import mujoco
    import numpy as np
    from musclemimic_models import get_xml_path

    model_path = get_xml_path("myofullbody")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bones = []
    for index in range(model.ngeom):
        name = model.geom(index).name
        if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (name.startswith("r_") and model.geom_group[index] == 0):
            continue
        body = model.body(model.geom_bodyid[index]).name
        if body not in {"pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "patella_r"}:
            continue
        mesh = int(model.geom_dataid[index])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        vertices = vertices @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index]
        start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        faces = model.mesh_face[start : start + count]
        bones.append({"name": name, "vertices": vertices.tolist(), "faces": faces.tolist()})

    paths = []
    anchors = {}
    for index in range(model.ntendon):
        name = model.tendon(index).name
        if not name.endswith("_r_tendon"):
            continue
        start, count = int(data.ten_wrapadr[index]), int(data.ten_wrapnum[index])
        points = data.wrap_xpos.reshape(-1, 3)[start : start + count]
        if len(points) < 2:
            continue
        # The full-body model also has right-sided trunk tendons. Keep only
        # paths in the pelvis/leg region, with an attachment below the pelvis.
        if points[:, 2].max() > 1.12 or points[:, 2].min() > 0.90:
            continue
        key = name.removesuffix("_tendon")
        paths.append({"name": key, "points": points.tolist(), "highlight": key in HIGHLIGHTS})
        if key in HIGHLIGHTS:
            # Mid-belly anchors, away from insertion tendons.
            distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            fraction = {"vaslat_r": 0.30, "tibant_r": 0.30, "bflh_r": 0.45, "gasmed_r": 0.38}[key]
            anchors[key] = [float(np.interp(distance[-1] * fraction, distance, points[:, i])) for i in range(3)]

    joints = {}
    for name, label in [("hip_flexion_r", "Hip"), ("knee_angle_r", "Knee"), ("ankle_angle_r", "Ankle")]:
        joints[label] = data.xanchor[model.joint(name).id].tolist()
    payload = {
        "model_xml": str(model_path),
        "pose": "model default qpos0; schematic angle markers do not represent measured values",
        "bones": bones,
        "paths": paths,
        "anchors": anchors,
        "joints": joints,
        "highlight_color": "#bd573e",
    }
    path = output_dir / "model_geometry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def render_blender(scene_path: Path) -> None:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    payload = json.loads(scene_path.read_text())
    output_dir = scene_path.parent
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x = 700
    scene.render.resolution_y = 1700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.world.color = (0.75, 0.75, 0.75)
    scene.view_settings.view_transform = "Standard"

    def material(name, color, roughness=0.6):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = roughness
        return mat

    bone_mat = material("Bone · warm ivory", (0.73, 0.71, 0.65))
    muted_mat = material("Other muscle paths", (0.60, 0.59, 0.56))
    highlight_mat = material("Selected muscle paths", (0.50, 0.095, 0.047))
    angle_mat = material("Angle markers", (0.035, 0.12, 0.17))
    for bone in payload["bones"]:
        mesh = bpy.data.meshes.new(bone["name"])
        mesh.from_pydata(bone["vertices"], [], bone["faces"])
        mesh.update()
        obj = bpy.data.objects.new(bone["name"], mesh)
        scene.collection.objects.link(obj)
        obj.data.materials.append(bone_mat)
        for polygon in mesh.polygons:
            polygon.use_smooth = True

    def curve(name, points, radius, mat, *, belly=False):
        data = bpy.data.curves.new(name, "CURVE")
        data.dimensions = "3D"
        data.resolution_u = 16
        data.bevel_depth = radius
        data.bevel_resolution = 4
        spline = data.splines.new("POLY")
        # Subdivide without moving any evaluated path out of its geometry.
        coords = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            for step in range(12):
                t = step / 12
                coords.append(tuple((1 - t) * x + t * y for x, y in zip(a, b, strict=True)))
        coords.append(points[-1])
        spline.points.add(len(coords) - 1)
        for i, (point, xyz) in enumerate(zip(spline.points, coords, strict=True)):
            point.co = (*xyz, 1)
            if belly:
                point.radius = 0.24 + 0.76 * math.sin(math.pi * i / (len(coords) - 1)) ** 0.7
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.data.materials.append(mat)
        return obj

    muscle_objects = []
    for path in payload["paths"]:
        selected = path["highlight"]
        obj = curve(
            path["name"], path["points"], 0.012 if selected else 0.0009,
            highlight_mat if selected else muted_mat, belly=selected,
        )
        muscle_objects.append(obj)

    # Schematic arcs in the sagittal plane, offset laterally for visibility.
    angle_objects = []
    for label, point in payload["joints"].items():
        x, y, z = point
        x -= 0.09
        radius = {"Hip": 0.064, "Knee": 0.055, "Ankle": 0.046}[label]
        start, end = math.radians(-55), math.radians(62)
        points = [[x, y + radius * math.cos(start + (end - start) * i / 40),
                   z + radius * math.sin(start + (end - start) * i / 40)] for i in range(41)]
        angle_objects.append(curve(f"{label} rotation arc", points, 0.003, angle_mat))
        for angle in (start, end):
            endpoint = [x, y + radius * math.cos(angle), z + radius * math.sin(angle)]
            angle_objects.append(curve(f"{label} angle ray", [[x, y, z], endpoint], 0.0016, angle_mat))
        # Filled arrowhead at the end of the arc.
        tip = Vector(points[-1])
        tangent = Vector((0, -math.sin(end), math.cos(end)))
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.008, radius2=0, depth=0.022,
                                      location=tip - tangent * 0.004)
        obj = bpy.context.object
        obj.name = f"{label} rotation arrow"
        obj.rotation_euler = tangent.to_track_quat("Z", "Y").to_euler()
        obj.data.materials.append(angle_mat)
        angle_objects.append(obj)

    def area_light(name, location, energy, size):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((-0.1, 0.16, 0.5)) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area_light("Large softbox", (-3, -4, 4), 360, 4)
    area_light("Posterior fill", (2, 3, 3), 300, 3)
    area_light("Rim", (-3, 3, 2), 200, 2)
    camera_data = bpy.data.cameras.new("Anatomy orthographic")
    camera = bpy.data.objects.new("Anatomy orthographic", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.22
    target = Vector((-0.105, 0.14, 0.56))
    projections = {}
    for view, location in {
        "anterior": (-1.5, -4, 1.15),
        "posterior": (0.7, 4, 1.15),
        "angles": (-4, 0.14, 0.65),
    }.items():
        camera.location = location
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        for obj in muscle_objects:
            obj.hide_render = view == "angles"
        for obj in angle_objects:
            obj.hide_render = view != "angles"
        bpy.context.view_layer.update()
        points = payload["joints"] if view == "angles" else {
            key: xyz for key, xyz in payload["anchors"].items() if HIGHLIGHTS[key]["view"] == view
        }
        projections[view] = {}
        for key, xyz in points.items():
            if view == "angles":
                xyz = [xyz[0] - 0.09, xyz[1] + 0.04, xyz[2]]
            point = world_to_camera_view(scene, camera, Vector(xyz))
            projections[view][key] = [point.x, point.y]
        scene.render.filepath = str(output_dir / f"{view}.png")
        bpy.ops.render.render(write_still=True)
    (output_dir / "legend.json").write_text(json.dumps({
        "model": "MyoFullBody", "highlight_color": payload["highlight_color"],
        "muscles": HIGHLIGHTS, "projections": projections,
        "note": "Model muscle paths; illustrative thickness. Rotation arcs are schematic.",
    }, indent=2) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "gait120_anatomy.blend"))


def main() -> None:
    args_list = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--output_dir", type=Path, default=Path("assets/gait120_anatomy"))
    parser.add_argument("--blender-scene", type=Path)
    args = parser.parse_args(args_list)
    if args.blender_scene:
        render_blender(args.blender_scene)
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_path = export_model(output)
    subprocess.run([
        args.blender, "--background", "--threads", "8", "--python", str(Path(__file__).resolve()),
        "--", "--blender-scene", str(scene_path),
    ], check=True)


if __name__ == "__main__":
    main()
