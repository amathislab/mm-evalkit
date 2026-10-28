# MM-EvalKit (MuscleMimic Evaluation Kit)

<p align="center">
  <a href="https://pypi.org/project/mm-evalkit/"><img src="https://img.shields.io/pypi/v/mm-evalkit.svg" alt="PyPI version"></a>
  <a href="https://github.com/amathislab/mm-evalkit/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <a href="https://arxiv.org/abs/2603.25544"><img src="https://img.shields.io/badge/Preprint-arXiv-b31b1b" alt="Preprint"></a>
  <a href="https://huggingface.co/collections/amathislab/musclemimic"><img src="https://img.shields.io/badge/-HF%20Assets-FFD21E?logo=huggingface&amp;logoColor=000" alt="HF Assets"></a>
</p>

`mm-evalkit` is the evaluation benchmark for MuscleMimic policies. It compares
rollout NPZ files with recorded human data at three physiological levels:

- kinematics: joint-angle RMSE and waveform correlation;
- kinetics: joint-moment and vertical-GRF RMSE and waveform correlation; and
- neuromuscular behavior: muscle-activation correlation and muscle RSA against
  human EMG.

The benchmark produces population and subject-matched metrics, human reference
baselines, statistical reports, figures, and exclusion audits.

## Dataset list

MM-EvalKit does not redistribute human recordings. Download each dataset from
its original archive and cite its accompanying paper using the entries in
[Citation and dataset credit](#citation-and-dataset-credit).

| Dataset | Official resources | Evaluation | Analysis unit |
|---|---|---|---|
| Gait120 | [Data](https://doi.org/10.6084/m9.figshare.27677016) · [Paper](https://doi.org/10.1038/s41597-025-05391-0) | Joint angles, joint moments, vertical GRF, muscle activation, and EMG | Subject and gait cycle |
| Wang | [Data](https://doi.org/10.5281/zenodo.6457662) · [Paper](https://doi.org/10.1017/wtc.2023.7) | Joint angles, muscle activation, and EMG | Population gait profile |
| ULTRA-MoCap | [Data](https://doi.org/10.6084/m9.figshare.28751156.v1) · [Paper](https://doi.org/10.1038/s41597-026-06687-5) | Arm kinematics, muscle activation, and EMG | Subject, movement, and trial |

Gait120 provides all three evaluation levels over one retained subject cohort.

### Gait120 inputs

The Gait120 loader does **not** read the original figshare archive directly. It
expects the cleaned redistribution, which carries the per-subject layout the
readers resolve (`Gait120_###_to_###/S###/EMG/ConvertedData.mat` and
`JointAngle/LevelWalking/Trial##/step##.mot`):

```bash
hf download amathislab/Gait120-EMG --repo-type dataset --local-dir /path/to/Gait120
```

The kinetics layer needs a second archive. Human joint moments come from
OpenSim inverse dynamics and human GRF from corrected force-plate recordings;
neither is part of the Gait120 release, and both are published separately as
per-subject, per-trial NPZ files (`S###/S###_Trial##_Step01_Step02.npz`):

```bash
hf download amathislab/Boo_LevelWalking --repo-type dataset --local-dir /path/to/gait120-kinetics
```

Both are redistributions of the original datasets under their own licenses.
Cite the original authors listed in
[Citation and dataset credit](#citation-and-dataset-credit), not the mirrors.
The `hf` command ships with `huggingface_hub`; install it with
`python -m pip install huggingface_hub` if it is not already available.

## Install

The PyPI distribution is named `mm-evalkit`; it provides the `mm_evalkit` Python
package and the `mm-evalkit` command-line interface. Install it with Python 3.11 or
newer:

```bash
python -m pip install mm-evalkit
```

Synchronized visualization requires the `render` extra:

```bash
python -m pip install 'mm-evalkit[render]'
```

## Configure data paths

```bash
mm-evalkit paths set gait120 /absolute/path/to/Gait120
mm-evalkit paths set wang /absolute/path/to/Wang
mm-evalkit paths set ultra-mocap /absolute/path/to/ULTRA-MoCap
mm-evalkit paths show
```

An explicit `--dataset_dir` on an evaluation command overrides the saved path.

## Define policies

Evaluation commands accept a YAML file containing one or more rollout exports:

```yaml
policies:
  - id: policy-a
    label: "Policy A"
    trajectory_data_path: trajectory_data/policy-a/episodes.npz
    color: "#2a78d6"
    linestyle: "-"
```

The rollout `motion_group` and `traj_id` values determine subject and trial
identity. Compared policies must use the same ordered references.

## Evaluate

### Gait120

```bash
mm-evalkit evaluate gait120 \
  --motion_group GAIT120_TESTING \
  --policy_config_file policies.yaml \
  --plots_dir results/gait120 \
  --kinetics_reference_dir /path/to/gait120-kinetics
```

`--kinetics_reference_dir` enables the kinetics layer. The rollout NPZ provides
policy joint-force and foot-contact-force signals. Human joint moments come
from inverse dynamics; human GRF comes from corrected Gait120 force-plate
measurements. Both directories come from the archives in
[Gait120 inputs](#gait120-inputs); the dataset path is the cleaned
redistribution rather than the original figshare download.

Supplying `--kinetics_reference_dir` restricts every layer to the subjects that
have an inverse-dynamics trial, because one cohort is shared across layers. A
kinematics-and-EMG run without it therefore reports a larger subject count.

MuscleMimic exports these signals with `fullbody.eval --use_mujoco
--export_trajectory --export_kinetics`. Activation is sampled before each control
step. GRF is saved in world coordinates at both control and physics frequency.
Gait120 compares the control-rate vertical GRF, normalized by body weight.

### Wang

```bash
mm-evalkit evaluate wang \
  --policy_config_file policies.yaml \
  --plots_dir results/wang
```

### ULTRA-MoCap

```bash
mm-evalkit evaluate ultra-mocap \
  --motion_group ULTRA_MOCAP_BIMANUAL_TESTING_SUBJECTS \
  --policy_config_file policies.yaml \
  --plots_dir results/ultramocap
```

Each run requires an empty output directory. Dataset-specific options are
available through:

```bash
mm-evalkit evaluate --list-datasets
mm-evalkit evaluate gait120 --help
mm-evalkit evaluate wang --help
mm-evalkit evaluate ultra-mocap --help
```

## Outputs

Evaluation outputs include:

- population and subject-matched metric tables;
- waveform and summary figures;
- paired Wilcoxon reports with Holm correction;
- retained-subject and exclusion audits; and
- run manifests where the workflow provides them.

Gait120 kinetics writes `kinetics_profiles.svg`, `kinetics_metrics.svg`,
`kinetics_metrics.csv`, and `kinetics_input_omissions.csv`.

Each dataset command documents its complete configuration through `--help`.
Input archives are validated before analysis starts.

## Comparison figure

`figures/plot_gait120_comparison.py` draws one Gait120 figure covering all three
physiological levels for any number of policies: right-limb joint angles, the
matching joint moments and vertical GRF, muscle activation against normalized
EMG, and subject-matched Pearson *r* and RMSE for each layer.

The script imports `mm_evalkit` from an installed package, or from `src/` when
run inside a checkout. It needs the runtime dependencies either way, so install
the package first if the import fails:

```bash
python -m pip install -e .
```

Point `--dataset_dir` and `--kinetics_reference_dir` at the two archives from
[Gait120 inputs](#gait120-inputs). `figures/policies.example.yaml` is a
template: copy it and edit each `trajectory_data_path` to your own rollouts,
since the paths it ships with are placeholders.

```bash
python figures/plot_gait120_comparison.py \
  --dataset_dir /path/to/Gait120 \
  --kinetics_reference_dir /path/to/gait120-kinetics \
  --policy_config_file figures/policies.example.yaml \
  --output_dir results/figure
```

It writes `gait120_comparison.{png,svg,pdf}` and `gait120_comparison_metrics.csv`,
which carries every plotted mean, SD and per-subject value. The analysis stage
takes several minutes, so its result is cached beside the output; pass
`--refresh` after changing the policies or the data.

Human data is drawn as one grey trace per retained subject rather than a mean
with a spread band, and policy profiles are drawn without a band. Colours and
line styles come from the policy YAML, so restyling means editing that file
rather than the script. `figures/policies.example.yaml` documents the palette
choice, including why the green entry also carries a dashed line style.

### Anatomical legend

Render MyoFullBody's right-leg bone meshes and muscle paths with Blender:

```bash
python figures/render_gait120_anatomy.py \
  --blender /path/to/blender \
  --output_dir assets/gait120_anatomy
```

Add `--anatomy_dir assets/gait120_anatomy` to the comparison command above to
include anterior and posterior views highlighting VL, BF long head, TA and GM,
plus a sagittal view marking hip, knee and ankle rotation. Muscle-path thickness
and rotation arcs are illustrative. The renderer also saves an editable
`gait120_anatomy.blend` scene. An existing analysis cache can be reused with
`--cache /path/to/gait120_comparison_cache.pkl` when only the layout changes.

For a lateral gait sequence aligned to the waveform axes, render poses from
one stride of a rollout episode. Run the comparison command once first: its
analysis cache supplies the human averages drawn on the poses.

```bash
python figures/render_gait120_sequence.py \
  --blender /path/to/blender \
  --rollout /path/to/episodes.npz \
  --episode 1 \
  --analysis_cache results/figure/gait120_comparison_cache.pkl \
  --kinetics_reference_dir /path/to/gait120-kinetics \
  --output_dir assets/gait120_sequence
```

Use `--sequence_dir assets/gait120_sequence` on the comparison command for the
wide layout: three columns for angles, kinetics and muscle activity, plus a
fourth column of agreement metrics. Each waveform column starts with a pose
strip whose hips sit exactly on the phase ticks, and a stance/swing bar. The
angle and muscle columns show one right stride; the kinetics column shows its
own window (below). The poses are one policy rollout, but the overlays are
Gait120 subject averages: kinetics poses carry the mean vertical GRF as an
arrow, muscle paths are coloured by mean EMG scaled to each muscle's peak, and
stance bars mark where mean vertical GRF exceeds 0.05 BW. The same render also
feeds `gait120_overlays.*`, which sets each policy's mean GRF and muscle signal
against the Gait120 mean on the same poses. Poses illustrate one recorded
policy rollout, rather than a population mean. Root translation is removed, and
highlighted muscle paths are projected onto the lateral plane so that medial
paths remain visible.

The comparison figure draws policy muscle **excitation** by default:
`clip(muscle_commands, 0, 1)`, the control MuJoCo integrates into activation.
Pass `--muscle_signal activation` for the exported `muscle_activations`. The
muscle Pearson r follows the same choice, and a cached analysis is refused if
it was built with the other signal. The analysis CLI takes the same
`--muscle_signal` flag and defaults to activation.

Moments and vertical GRF use the fixed 1.00 s window from left heel strike by
default, which every rollout covers completely; it sits about half a cycle out
of phase with angles and EMG, so column B is labelled in seconds.
`--kinetics_window right_stride` uses the force-plate right stride instead
(each Gait120 `Step01_Step02` trial spans two right strides, and only the
second right stance lands on a plate). That shares the other columns' phase,
but rollouts stop a few frames before their trial ends, so a stride covered for
at least 90 % from heel strike is kept with its last ~7 % undefined.

### Aligning a rollout first

Compared rollouts must share one ordered motion group. Some exports carry an
unaligned `episode_N_traj_qpos`, the reference trajectory the policy tracks, in
which the coupled MyoFullBody knee's dependent degrees of freedom have not been
recomputed. Copy an aligned reference into such an archive with:

```bash
python figures/align_rollout.py \
  --source rollouts/new_policy.npz \
  --reference rollouts/aligned_policy.npz \
  --output rollouts/new_policy_aligned.npz
```

Only `traj_qpos` is rewritten, and only after both archives are checked to agree
on episode count, joint names, motion names, and the trajectory each episode
refers to. Pass `--verify-against` a second aligned archive to confirm the
reference is policy-independent before it is copied.

Alignment changes only analyses that read the retargeted reference, namely
`--kinematics_source mfb-retargeted` and the synchronized renderers. The default
`opensim-ik` comparison takes its ground truth from the human dataset and is
unaffected.

### Reading the figure

Two properties of the Gait120 inputs belong in any caption:

- **Cohort.** Supplying `--kinetics_reference_dir` restricts every layer to the
  subjects that have an inverse-dynamics trial, because one cohort is shared
  across layers. Kinematics and EMG alone are available for more subjects, so a
  kinematics-only run reports a larger *n* than this figure.
- **Phase reference.** Joint-angle and EMG cycles run right-heel-strike to
  right-heel-strike, while the moment and GRF references are cut as a fixed
  1.00 s window from *left* heel strike. The kinetics row therefore sits roughly
  half a cycle out of phase with the other two and is not a stride-normalized
  cycle. Metrics are unaffected, since human and policy data share one phase
  convention within each layer.

Muscle activity is scored by correlation only: normalized EMG and simulated
activation share no common scale.

## Synchronized visualization

Renderers synchronize policy motion with human and artificial muscle signals:

```bash
mm-evalkit render gait120 --help
mm-evalkit render ultra-mocap --help
```

## Demos

### Overhead reach (`P11_OverheadReach_180`) · [ULTRA-MoCap dataset](https://doi.org/10.6084/m9.figshare.28751156.v1)

![ULTRA-MoCap overhead-reach demo](https://raw.githubusercontent.com/amathislab/mm-evalkit/main/assets/demos/P11_OverheadReach_180_demo.gif)

### Shoulder rotation (`P12_ShoulderRotation_Normal`) · [ULTRA-MoCap dataset](https://doi.org/10.6084/m9.figshare.28751156.v1)

![ULTRA-MoCap shoulder-rotation demo](https://raw.githubusercontent.com/amathislab/mm-evalkit/main/assets/demos/P12_ShoulderRotation_Normal_demo.gif)

### Gait step (`S052_step01`) · [Gait120 dataset](https://doi.org/10.6084/m9.figshare.27677016)

![Gait120 step demo](https://raw.githubusercontent.com/amathislab/mm-evalkit/main/assets/demos/S052_step01_demo.gif)

## Add a dataset

Dataset adapters declare their source contract, workflow, CLI entrypoint, and
tests. Keep dataset files outside the repository and document their license and
access requirements.

External packages register adapters in `pyproject.toml`:

```toml
[project.entry-points."mm_evalkit.datasets"]
example = "example_package.cli:main"

[project.entry-points."mm_evalkit.renderers"]
example = "example_package.render:prepare_clip"
```

Validation should cover a minimal valid dataset, malformed inputs, alignment,
normalization, metrics, registry discovery, and an end-to-end run. The
[architecture guide](https://github.com/amathislab/mm-evalkit/blob/main/ARCHITECTURE.md)
defines the extension boundaries.

## Development

```bash
git clone https://github.com/amathislab/mm-evalkit.git
cd mm-evalkit
make install-dev
make format
make lint
make test
make precommit
```

Ruff and pre-commit cover every Python file in the repository.

See [Architecture](https://github.com/amathislab/mm-evalkit/blob/main/ARCHITECTURE.md)
for data flow, module ownership, input contracts, and public extension points.

## Citation and dataset credit

If you use MM-EvalKit, cite the MuscleMimic paper:

```bibtex
@article{Li2026MuscleMimic,
  title={Towards Embodied AI with MuscleMimic:
    Unlocking full-body musculoskeletal motor learning at scale},
  author={Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and
    Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and
    Mathis, Alexander},
  journal={arXiv preprint arXiv:2603.25544},
  year={2026}
}
```

Please also cite every human dataset used in your evaluation:

```bibtex
@article{boo2025comprehensive,
  title={Comprehensive human locomotion and electromyography dataset: Gait120},
  author={Boo, Junyo and Seo, Dongwook and Kim, Minseung and Koo, Seungbum},
  journal={Scientific data},
  volume={12},
  number={1},
  pages={1023},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

@article{wang2023wearable,
  title={A wearable real-time kinetic measurement sensor setup for human locomotion},
  author={Wang, Huawei and Basu, Akash and Durandau, Guillaume and Sartori, Massimo},
  journal={Wearable technologies},
  volume={4},
  pages={e11},
  year={2023},
  publisher={Cambridge University Press}
}

@article{fritsche2026ultra,
  title={ULTRA-MoCap: A multimodal IMU and sEMG dataset for upper body joint kinematics analysis},
  author={Fritsche, Oliver and Camacho, Steven and Hossain, Md Sanzid Bin and Halfpenny, Tyler and Arciniegas, Carlos and Dranetz, Joseph and Hadley, Dexter and Guo, Zhishan and Choi, Hwan},
  journal={Scientific Data},
  volume={13},
  number={1},
  pages={622},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```
