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
measurements.

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
