# `mm-evalkit` architecture

`mm-evalkit` implements the evaluation benchmark for MuscleMimic policies. It
measures kinematic, kinetic, and neuromuscular agreement between rollout NPZ
files and recorded human data.

## Benchmark domains

| Level | Policy signal | Human reference | Metrics |
|---|---|---|---|
| Kinematics | Joint angles | OpenSim IK or a shared retargeted trajectory | RMSE and Pearson waveform correlation |
| Kinetics | Joint moments and foot-contact forces | OpenSim inverse-dynamics moments and corrected Gait120 force-plate GRF | RMSE and Pearson waveform correlation |
| Neuromuscular | Muscle activation | EMG | Muscle correlation and representational similarity analysis |

Each metric is reported at its valid population or subject-matched scope. Human
reference baselines quantify between-subject variability, within-subject
reliability, and RSA noise ceilings where the dataset supports them.

## Dataset workflows

| Dataset | Workflow | Analysis unit |
|---|---|---|
| Gait120 | `gait` | Subject and gait cycle |
| Wang | `gait` | Population gait profile |
| ULTra-MoCap | `bimanual` | Subject, movement, and trial |

The Gait120 workflow evaluates all three benchmark levels over one retained
subject cohort.

Kinetics uses raw actuator forces, foot-contact forces, channel units, model
mass, and reference-frame indices embedded in the rollout NPZ. The configured
Gait120 kinetic reference directory supplies the human moment and GRF records.

## Dependency direction

```text
CLI / Python API
       |
       v
registry and validated configuration
       |
       v
workflows/        gait and bimanual orchestration
       |
       +-------------------+
       v                   v
datasets/               stats/
human-data readers      RSA, Wilcoxon, report figures
```

- `datasets/` owns released-data schemas, channel maps, units, and motion
  identities.
- `workflows/` owns rollout decoding, cohort selection, metrics, exclusions,
  reporting, and pipeline order.
- `stats/` owns reusable statistical calculations and their figures.
- `rendering/` owns optional synchronized visualization: scenes, signal panels,
  layouts, frame composition, and video encoding.
- `registry.py` connects public dataset names to benchmark adapters and optional
  visualization adapters.

## Data flow

### Evaluation

```text
CLI arguments
  -> PolicySpec and dataset configuration
  -> rollout schema validation
  -> policy and human-data loading
  -> shared cohort selection
  -> profile construction
  -> metrics and references
  -> Wilcoxon tests
  -> tables, figures, manifests, and exclusion audits
```

### Optional visualization

```text
CLI / Python API
  -> render adapter
  -> RenderClip
  -> scene + signal panel + layout
  -> frame loop
  -> FFmpeg output
```

## Source layout

```text
mm_evalkit/
├── cli.py                      # evaluate / render / paths dispatch
├── cli_options.py              # shared command-line fields and aliases
├── registry.py                 # dataset and renderer discovery
├── policies.py                 # PolicySpec and YAML validation
├── paths.py / paths_cli.py     # dataset path resolution
├── activation.py               # rollout activation-phase alignment
├── arrays.py                   # NumPy array aliases
├── datasets/
│   ├── gait120.py              # gait kinematics and EMG
│   ├── gait120_kinetics.py     # force-plate and inverse-dynamics records
│   ├── wang.py                 # population gait profiles
│   ├── ultra_mocap.py          # preprocessed sEMG records
│   └── motion_sets.py          # ordered trajectory identities
├── stats/
│   ├── rsa.py                  # representational similarity
│   ├── wilcoxon.py             # paired tests and Holm correction
│   └── report.py               # statistical figures
├── rendering/
│   ├── models.py               # RenderClip and shared records
│   ├── renderer.py             # reusable frame loop
│   └── scenes/ panels/ layouts/
└── workflows/
    ├── gait/
    │   ├── state.py            # GaitAnalysis and stage records
    │   ├── loading.py          # validated state assembly
    │   ├── channels.py         # model channel order and subsets
    │   ├── cycles.py           # gait-event and cycle extraction
    │   ├── policy_data.py      # rollout kinematics and activation
    │   ├── kinetics_data.py    # rollout/reference kinetic profiles
    │   ├── human_data.py       # human cohort and gait selection
    │   ├── metrics.py / kinetics_metrics.py
    │   ├── references.py / statistics.py
    │   ├── diagnostics.py / reporting.py / plotting/
    │   └── pipeline.py
    └── bimanual/
        ├── models.py           # typed stage records
        ├── segmentation.py     # repetition boundaries
        ├── rollout.py / trials.py / muscles.py
        ├── metrics.py / references.py / statistics.py
        ├── reporting.py / tables.py / *_plots.py
        └── pipeline.py
```

## Workflow state

Gait stages share [`GaitAnalysis`](src/mm_evalkit/workflows/gait/state.py). It owns:

- validated configuration and channel definitions;
- the retained human cohort;
- one `PolicyRun` per compared policy;
- kinematic, kinetic, and neuromuscular profiles;
- population, subject-matched, and reference metrics; and
- output paths and plot style.

[`load_gait_analysis`](src/mm_evalkit/workflows/gait/loading.py) performs input I/O
and constructs this state. Pipeline stages fill declared `slots=True` records.

The bimanual workflow uses typed records in
[`workflows/bimanual/models.py`](src/mm_evalkit/workflows/bimanual/models.py) for
rollouts, trials, repetitions, metrics, and reports.

## Input contracts

- `PolicySpec` validates policy IDs, labels, rollout paths, colors, and optional
  dataset-specific inputs.
- `motion_group` defines the ordered identity behind every rollout `traj_id`.
- `activation_sample_phase` places muscle activation on the declared state
  timeline.
- `kinetics_sample_phase`, force-channel metadata, model mass, and exact
  reference steps define embedded policy kinetics.
- Multi-policy analyses require matching coordinate schemas, trajectory IDs,
  reference waveforms, subjects, trials, and repetition boundaries.
- Missing human records and incomplete kinetic windows produce explicit audit
  rows. Invalid shapes, units, identities, or required channels stop loading.
- Undefined correlations are stored as `NaN` and excluded through finite-pair
  aggregation.

## Stage ownership

- Loaders validate external files and return typed records.
- Selection stages define the shared scientific cohort.
- Metric modules compute values from selected records.
- Reference modules compute human-human and within-subject benchmarks.
- Statistical modules operate on subject-indexed paired samples.
- Plot and table modules render computed results.
- Pipelines call stages in dependency order.

## Public interfaces

```bash
mm-evalkit evaluate --list-datasets
mm-evalkit evaluate gait120 --help
mm-evalkit evaluate wang --help
mm-evalkit evaluate ultra-mocap --help
mm-evalkit render gait120 --help
mm-evalkit render ultra-mocap --help
mm-evalkit paths show
```

Python callers use the registry directly:

```python
from mm_evalkit import run_dataset

run_dataset("gait120", ["--help"])
```

External packages register benchmark adapters through `mm_evalkit.datasets` and
optional visualization adapters through `mm_evalkit.renderers`. Dataset listing
reads entry-point metadata; adapter modules load when selected. Built-in
adapter names are reserved.

## Quality gates

Tests mirror `src/mm_evalkit/` and use pytest `importlib` mode. The package-wide
commands are:

```bash
make format
make lint
make test
make precommit
```

The Ruff configuration and pre-commit hooks cover every Python file in the
repository.
