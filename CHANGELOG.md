# Changelog for REINVENT

This follows the guideline on [keep a changelog](https://keepachangelog.com/)


## [4.9.34] 2026-11-06

Original implementation by Marten ([@focccus](https://github.com/focccus)),
[PR #353](https://github.com/MolecularAI/REINVENT4/pull/353).

### Added

- **Modular diversity filter architecture**: the monolithic `DiversityFilter` base class has been
  replaced by a pluggable `memories/filters/` package.  Available filter types (set via
  `[diversity_filter] type`): `IdenticalMurckoScaffold`, `IdenticalTopologicalScaffold`,
  `ScaffoldSimilarity`, `PenalizeSameSmiles`, and `BitBirchDiversityFilter` (requires the optional
  `bblean` dependency: `pip install reinvent[bitbirch]`).
- **Configurable penalty functions**: `Step`, `Linear`, `Tanh`, `Sigmoid`, and `Erf` penalty shapes
  are now selectable via `[diversity_filter] penalty_function`.
- **Intrinsic reward system**: `RND` (Random Network Distillation) and `Information`-based intrinsic
  rewards can be attached to the diversity filter via `[intrinsic_reward]`.
- **IS-weighted inception**: importance-sampling weighted sampling (`is_weighted_sampling`) and score
  reweighting (`is_weighted_scores`) in `[inception]`; diversity-penalty coupling uses the new
  read-only `peek_penalty` so the inception buffer no longer pollutes the diversity filter state.
- **Extended TensorBoard reporting**: `Diversity/` and `Inception/` scalar namespaces expose
  per-step filter and inception metrics; the remote reporter gains structured diversity and inception
  sub-records.
- **Config documentation**: `configs/diversity_filter.md` and `configs/inception.md` document all
  new options.

### Fixed

- Inception diversity-penalty coupling: `Inception.update` was calling
  `DiversityFilter.calculate_penalty` on the full stored buffer every step, which is not read-only —
  it inserts fingerprints / scaffolds into the shared filter bypassing the `minscore` gate and
  exact-SMILES dedup.  Buffer molecules were re-inserted every step, saturating clusters early,
  penalising freshly sampled molecules in those clusters, and growing the BitBirch tree by up to
  `memory_size` fingerprints per step.  Fixed by adding a read-only `peek_penalty` method to
  `DiversityFilter` (default: all-ones) overridden in `ScaffoldDiversityFilter`,
  `BitBirchDiversityFilter`, and `PenalizeSameSmiles`; `Inception.update` now calls `peek_penalty`.
- `InformationPenalty.update_score` returned `None` when no molecule reached `minscore` and a
  `scaffolds` list shorter than the batch when some did, causing `RuntimeError` in the CSV logger
  header-drift guard and silent row truncation in `write_summary`.  Fixed by always returning a
  full-batch-length `DiversityResults`.
- `BucketCounter` backward-compatibility shim: the `f72305d4` refactoring removed
  `memories/bucket_counter.py` without a shim, so checkpoints pickled before that refactoring
  raised `ModuleNotFoundError` on load.  Added a proper re-export shim at the old path.
- `BitBirchDiversityFilter`: passing `recluster_tolerance` to the `BitBirch()` constructor
  alongside a custom `merge_criterion` raised an immediate `ValueError` (mutually exclusive
  arguments).  Fixed by forwarding `recluster_tolerance` only at the point `set_merge` is called.
- `BitBirchDiversityFilter`: after the first recluster, `set_merge("diameter")` silently replaced
  the custom `BitBirchTrackingDiameterMerge` with `_FastDiameterMerge`, disabling bucket tracking
  for all subsequent steps (all penalties became 1.0).  Fixed by restoring `self.merge_criterion`
  after reclustering.
- `[diversity_filter] debug` was registered in `SectionDiversityFilter` as `bit_birch_debug`,
  diverging from the documented name and from `[inception].debug`.  Any config following the docs
  failed `extra="forbid"` validation.  Renamed to `debug` (no deprecation alias needed; the field
  was introduced in this PR).
- Checkpoint backward-compatibility regression introduced by the `f72305d4` diversity-filter
  refactoring: the four old stub files (`identical_murcko_scaffold.py`, etc.) became effectively
  abstract after new abstract methods were added to `DiversityFilter`, causing `torch.load` to raise
  `TypeError` when loading checkpoints saved before the refactoring.  Fixed by converting the stubs
  to re-export shims for the corresponding concrete classes in `memories/filters/`.


## [4.9.33] 2026-11-04

### Changed

- Loss modulation: configuration validation now rejects `loss_modulation = true` endpoints when
  `learning_strategy.type` is not `dap` (the scalar-loss strategies `mascof`/`mauli`/`sdap` and the
  reinforce variants cannot be modulated per molecule); the error is raised at config read instead
  of mid-training.


## [4.9.32] 2026-11-02

### Added

- `bblean` (BitBirch diversity filter) added as an optional dependency installable via `pip install reinvent[bitbirch]` or `pip install reinvent[all]`.  Added `bitbirch` to the `OPTIONAL_DEPENDENCIES` tuple in `install.py`.


## [4.9.31] 2026-11-02

### Fixed

- Added missing `psutil` dependency


## [4.9.30] 2026-10-26

### Fixed

- `MolVolume`: removed artificial `randomSeed=0xF00D` from `EmbedMolecule`; the seed
  had no scientific justification and was only present to satisfy a now-removed exact-value
  assertion in the unit test.


## [4.9.29] 2026-10-22

### Fixed

- `test_comp_mol_volume`: replaced brittle exact-value assertions with
  property-based checks (finite, positive, < 500 Å³, larger molecule scores
  higher).  RDKit's ETKDG embedder can produce slightly different conformers
  across versions even with a fixed seed, making exact volume comparisons
  inherently fragile.


## [4.9.28] 2026-10-20

### Added

- New scoring component `PathFPSimilarity`: Tanimoto similarity using RDKit
  path-based (topological) fingerprints (`RDKFingerprint`), as an alternative to
  the Morgan-fingerprint-based `TanimotoSimilarity`.  Supports `min_path`,
  `max_path`, `n_bits`, and `aggregation` (`"max"` / `"mean"` / `"min"`, default
  `"max"`) parameters.  Tagged with `@molcache` for RDKit `Mol` caching.
- New transform `symmetric_exponential_decay`: computes `exp(-k * |x - target|)`,
  a symmetric bell centred on `target` that peaks at 1.0 when `x == target`.
  Distinct from the existing one-sided `exponential_decay` transform.
- `configs/SCORING.md` updated with entries for `PathFPSimilarity`,
  `SymmetricExponentialDecay`, and the previously undocumented `aggregation`
  parameter of `TanimotoSimilarity`.


## [4.9.27] 2026-10-20

### Fixed

- `SmilesAssociatedComponentResults.fetch_scores` now returns `(NaN,) * n_endpoints` for SMILES absent from the results dict instead of raising `KeyError`; `fetch_metadata` returns `None` instead of raising `AttributeError`.  Prevents reporters from crashing on incomplete component results while preserving score ordering.  Ported from public PR [#357](https://github.com/MolecularAI/REINVENT4/pull/357) by BangBang-afk.


## [4.9.26] 2026-10-20

### Fixed

- `MatchingSubstructure` now raises `ValueError` with the offending pattern string if a SMARTS cannot be parsed by RDKit, rather than storing a `None` pattern and crashing with an opaque Boost `ArgumentError` at scoring time.  Silently dropping the pattern is unsafe because `patterns_per_endpoint` is index-aligned with `use_chirality` and endpoint names.  Ported from public PR [#367](https://github.com/MolecularAI/REINVENT4/pull/367) by Ishaan Samantray.


## [4.9.25] 2026-10-20

### Fixed

- All `@molcache`-decorated scoring components that passed `None` molecules to RDKit now return
  `NaN` instead of raising `AttributeError` (or other heterogeneous exceptions depending on
  RDKit version): `LinKInvent` fragment components, `PhysChem`, `RDKitDescriptors`, `MolVolume`,
  `TPSA`, `PMI`, `GroupCount`, `MatchingSubstructure`, `SAScore`, and `RingPrecedence`.
- `MolVolume`, `TPSA`, and `PhysChem`: broadened `except ValueError` to `except Exception` so
  all RDKit exception types are caught consistently across versions.
- `PMI`: replaced the `embed_result == -1` sentinel check with `try/except Exception` around
  the full `AddHs`/`EmbedMolecule`/`CalcNPR1`/`CalcNPR2` sequence.
- `LinKInvent`: added a systemic-failure `ERROR` log when every molecule in a batch fails to
  score (indicates a component or configuration bug rather than per-SMILES noise).
- One `_none_molecule_scores_nan` unit test added per affected component.


## [4.9.24] 2026-10-13

### Fixed

- Scoring components whose optional dependencies are not installed are now silently skipped at startup rather than logging a noisy `ERROR`.  If the user's config requests such a component, a clear `RuntimeError` is raised explaining which dependency is missing and how to install it.
- `get_components` now catches `KeyError` instead of `AttributeError` when looking up an unknown component type, fixing a latent bug where an unregistered component name produced an unhelpful `AttributeError` instead of the intended `"Unknown scoring component"` message.


## [4.9.23] 2026-10-13

### Changed

- `pumas` is now an optional dependency; install with `pip install reinvent[pumas]` or `pip install reinvent[all]`.  When `pumas` is absent, importing the scoring module no longer raises `ImportError`; setting `use_pumas = true` raises a clear `ImportError` with an install hint at scorer construction time.


## [4.9.22] 2026-10-09

### Added

- New test file `tests/utils/test_errors.py` (11 tests): covers `format_validation_error` formatting, truncation of long input values, and `fatal()` exit code, stderr output, and logger integration
- New test file `tests/utils/test_logmon.py` (10 tests): covers `setup_logger` single- vs dual-handler behaviour, handler levels, `SuppressTracebackFormatter` traceback suppression and record non-mutation, and the full split-stream integration scenario
- New test file `tests/test_reinvent_config.py` (22 tests): covers `ReinventConfig.run_type` Literal (all valid and multiple invalid values), `device` regex validator (all valid patterns including `xpu` for Intel ARC, multiple invalid strings), `version` field bounds, `extra="forbid"` enforcement, and `ExistingFilePath` / `WritableFilePath` annotated types
- New test file `tests/runmodes/unit_tests/test_rl_validation.py` (30 tests): covers `SectionStage.termination` Literal, `min_steps < max_steps` cross-field validator, `SectionInception` cross-field validator and bounds, `StepLRConfiguration` and `LambdaLRConfiguration` `__post_init__` bounds (including the `warmup=0` ZeroDivisionError case and `gamma=0` silent-kill case)
- New test file `tests/runmodes/unit_tests/test_smiles_gate.py` (10 tests): covers the `run_scoring` RDKit validity mask for valid, invalid, mixed, and empty SMILES; verifies `filter_by_pattern` no longer uses `np.ones` and correctly derives masks from `sampled.states` (including mock-based verification that `INVALID`/`DUPLICATE` states map to `False` masks)


## [4.9.21] 2026-10-09

### Fixed

- **Scoring run mode** (`run_scoring`): the validity mask was built as `True` for any non-empty string, allowing chemically invalid SMILES to reach scoring plugins. The mask is now derived from `Chem.MolFromSmiles` — any SMILES that RDKit cannot parse gets `mask=False` and is short-circuited to score `0.0` without touching any plugin. A warning is logged listing the count of skipped SMILES.
- **Sampling SMARTS filter** (`filter_by_pattern` in `run_sampling`): the scorer was called with `np.ones` masks (unconditionally `True` for all SMILES), bypassing the validity gate when `unique_molecules=False`. The masks are now derived from `sampled.states` (the same `SmilesState.INVALID` / `DUPLICATE` flags set by `validate_smiles()`), consistent with how the RL path handles validity.
- File extension check in `run_scoring.read_data` is now case-insensitive (`.CSV`/`.SMI` accepted alongside `.csv`/`.smi`).


## [4.9.20] 2026-10-09

### Fixed

- Eliminated duplicate `CRITICAL` log lines that appeared when a terminating error was raised: the pattern `logger.critical(msg); raise RuntimeError(msg)` has been replaced by a bare `raise` throughout `create_adapter`, `run_staged_learning`, `run_sampling`, `run_transfer_learning`, and `Reinvent.py`. The top-level `fatal()` handler is now the single point responsible for logging terminating errors
- `format_validation_error` truncates large `input` values (e.g. the full stage dict from a model-level cross-field validator) to 80 characters so the terminal output stays readable


## [4.9.19] 2026-10-09

### Fixed

- `torch.load` in `create_adapter` is now wrapped in a `try/except`; corrupt or non-Reinvent model files now produce a clear `"Failed to load model file: …"` message instead of a raw `UnpicklingError` or `EOFError`
- `inception.smiles_file` being set for a non-Reinvent model type was silently ignored; it now logs a `WARNING` explaining that seeding is only supported for the Reinvent model
- `use_checkpoint=True` with no `staged_learning` key in the agent checkpoint now logs a `WARNING` and starts with a fresh diversity filter instead of silently falling through to the `elif` branch
- File extension check in `aa_enumerator.read_smiles_csv_file` is now case-insensitive (`.CSV` and `.SMI` are accepted alongside `.csv` and `.smi`); error message for unknown extensions now states the expected formats


## [4.9.18] 2026-10-09

### Changed

- `SectionStage` gains a cross-field validator: `min_steps` must be less than `max_steps`; previously a stage with `min_steps=50, max_steps=10` ran all steps without the terminator ever evaluating the score
- `SectionInception.memory_size` and `sample_size` are now bounded `ge=1`; a new cross-field validator ensures `sample_size ≤ memory_size`; `inception.smiles_file` now uses `ExistingFilePath`
- All `getattr(module, dynamic_name)` calls that previously raised a bare `AttributeError` on unknown model types now use a guarded pattern and raise a `RuntimeError` with a clear message: `run_staged_learning` (RL learning class), `run_transfer_learning` (TL runner class, now also reports when a model type has no TL implementation), `setup_sampler` (sampler class), `get_pair_generator` (Mol2Mol pair generator)
- Empty SMILES result from `read_smiles_csv_file` is now treated as a fatal error in both RL (`run_staged_learning`) and sampling (`run_sampling`); previously an empty file silently disabled SMILES conditioning (RL) or generated zero molecules (sampling) without any warning


## [4.9.17] 2026-10-09

### Changed

- `prior_file`, `agent_file` (RL), `input_model_file`, `smiles_file`, `validation_smiles_file` (TL), `model_file`, `smiles_file` (sampling), `smiles_file`, `amino_acid_library_file` (enumeration), and `transform_file` (data pipeline) are now validated to exist on disk at config-parse time via the new `ExistingFilePath` annotated type; missing files are caught before the run starts
- `output_model_file` (TL) and `chkpt_file` (RL stage) use a new `WritableFilePath` annotated type that checks the parent directory exists before the run begins, preventing a late `FileNotFoundError` during checkpoint save
- `SectionStage.termination` is now `Literal["simple", "null", "plateau"]`; invalid termination strings are rejected at validation time
- `clip_gradient_norm` (TL) is now bounded `ge=0.0`; `batch_size` (enumeration) is now bounded `ge=1`
- `StepLRConfiguration` and `LambdaLRConfiguration` in TL gain `__post_init__` validation: `lr`, `min`, `gamma`, `step`, `beta1`, `beta2`, `eps`, `factor`, and `warmup` are all checked for valid ranges — notably `warmup=0` previously caused a silent `ZeroDivisionError` in the learning-rate lambda
- Fixed dead error handler in `create_packages.py`: the `try/except KeyError` around the terminator `getattr` was unreachable (`getattr` raises `AttributeError`, not `KeyError`); replaced with a guarded `getattr(..., None)` check that reports valid choices


## [4.9.16] 2026-10-09

### Fixed

- Corrected a `NameError` in the main entry point caused by a missing `import logging` statement
- Eliminated duplicate error output when `--log-filename` is used: the terminal now shows a single `CRITICAL: <message>` line (no traceback); the log file receives the full timestamped entry with traceback. When no log file is configured both message and traceback appear on the terminal as before
- Added `SuppressTracebackFormatter` to `reinvent.utils.logmon` — used on the stderr handler whenever a log file is also active — so tracebacks are never forwarded to the terminal by the logging system itself


## [4.9.15] 2026-10-09

### Added

- New `reinvent.utils.errors` module with two shared helpers — `fatal()` and `format_validation_error()` — used by both entry points to separate user-facing messages from developer tracebacks

### Changed

- Both `reinvent` and `reinvent_datapre` entry points now validate the config file path and the log-file parent directory before attempting to open them, producing a clear message instead of a raw `FileNotFoundError`
- Unhandled exceptions in both entry points are now caught and routed through `fatal()`: pydantic `ValidationError` is reformatted into readable `[field] message (got: value)` lines; `FileNotFoundError`, `ValueError`, and `RuntimeError` show a one-line message; any other exception shows a minimal message and instructs the user to check the log file. The full traceback is written only to the log file, not to the terminal.


## [4.9.14] 2026-10-09

### Changed

- When a log file is specified with `--log-filename`, a second stderr handler is now always attached at `WARNING` level with a plain `LEVEL: message` format. Fatal errors and warnings are therefore always visible in the terminal even when all detailed output is redirected to a log file. This applies to both the main `reinvent` entry point and `reinvent_datapre`.


## [4.9.13] 2026-10-09

### Changed

- `run_type` in the top-level config is now validated against the set of known run modes (`sampling`, `staged_learning`, `transfer_learning`, `scoring`, `enumeration`); an unknown value now produces a clear error message listing valid choices instead of a bare `AttributeError`
- `device` is now validated at config-parse time against recognised PyTorch device strings (`cpu`, `cuda`, `mps`, `xpu` for Intel ARC, each with optional index); an invalid value is rejected immediately with a descriptive message
- `version` is now a declared config field (default: current config version) so that the version check always fires and a typo'd key is rejected by the `extra="forbid"` guard
- RL `batch_size` is now bounded (`ge=1`); `temperature` is bounded (`gt=0.0`) in both RL and sampling configs
- `sample_strategy` in RL and sampling configs is now a `Literal["multinomial", "beamsearch"]`; invalid strategies are rejected at validation time
- RL learning-strategy `type` is now a `Literal` of known strategy names (`dap`, `sdap`, `mauli`, `mascof`, and `_reinforce` variants)
- RL diversity-filter `type` is now a `Literal` of known filter class names; intrinsic-penalty `type` and `penalty_function` are similarly constrained
- Sampling `num_smiles` is now bounded (`ge=1`)
- Data-pipeline `FilterSection` now validates that `min_heavy_atoms` is strictly less than `max_heavy_atoms`


## [4.9.12] 2026-10-05

### Fixed

- Fixed order of components in PMI component


## [4.9.11] 2026-09-21

### Added

- Dynamic filter components: any scoring component can be promoted to a filter at runtime by setting `filter = true` in the component config block together with a step-type transform (`left_step`, `right_step`, `step`). The component runs in the filter pass (before scorers), gates downstream computation via the validity mask, and is reported in the output CSV but excluded from aggregation.


## [4.9.10] 2026-09-20

### Fixed

- Fixed potential issue with nan and valid masks being applied sequentially in reward.py


## [4.9.9] 2026-09-02

### Fixed

- Chained actions and specific token check in read\_smiles\_csv\_file()
- Fixed import unavailable on Windows


## [4.9.8] 2026-09-02

### Fixed

- Stereochemistry safety to diversity filter


## [4.9.7] 2026-08-21

### Fixed

- Improved CSV header writeout robustness for metadata columns which lead to malformed csvs in certain scenarios


## [4.9.6] 2026-08-18

### Added

- Fingerprint precedence based scoring component along with support script for creation of databases, in the same vein as ring precedence 


## [4.9.5] 2026-08-05

### Changed

- Reworked run\_staged\_learning.py to allow dynamic building of configuratioms


## [4.9.4] 2026-07-01

### Fixed

- Loading a model checkpoint that was saved with a V1-to-V2 promoted metadata dict (e.g. an agent trained from a prior without metadata) no longer raises `TypeError: ModelMetaDataV2.__init__() got an unexpected keyword argument 'origina_data_source'`. The V1 field `origina_data_source` is now renamed to `original_data_source` and the obsolete `updates`/`comments` lists are merged into `update_log` both when saving (`update_model_data_v2`) and when loading (`check_metadata`).


## [4.9.3] 2026-06-27

### Added

- Read stereochemisry support flag from model file metadata in sampling runmode


## [4.9.2] 2026-06-24

### Removed

- Removed SSH mount in the dockerfile for reinvent-scoring-plugins build per new security policy


## [4.9.1] 2026-06-24

### Added

- Aggregation parameter (mean, min, max) for TanimotoSimilarity scoring component to control how scores from multiple reference fingerprints are combined (default: max)
- Validation check in compute_scores to ensure components return exactly the declared number of score arrays (number_of_endpoints), preventing malformed CSV output


## [4.9.0] 2026-06-24

### Changed

- New metadata format including supporting tools


## [4.8.25] 2026-06-24

### Removed

- uv.lock in internal repository as not needed


## [4.8.24] 2026-06-10

### Added

- Better hardware reporting (accelarator)


## [4.8.23] 2026-06-10

### Changed

- Removed obsolete features: use\_cuda, unique\_sequence


## [4.8.22] 2026-06-03

### Changed

- Massively expanded unit and integration tests


## [4.8.21] 2026-06-02

### Changed

- Updated dependencies


## [4.8.20] 2026-06-02

### Changed

- Significant updates of notebooks and alignment with ChemProp2


## [4.8.19] 2026-06-02

### Changed

- Significant updates of documentation in the markdown files in configs/


## [4.8.18] 2026-06-01

### Changed

- Removed tomli dependency and replaced by built-in tomllib


## [4.8.17] 2026-05-29

### Fixed

- MacOS support for some Apple GPUs (see PyTorch documentation for which ones)


## [4.8.16] 2026-05-21

### Fixed

- advanced some package versions to resolve security breaches


## [4.8.15] 2026-05-21

### Changed

- refactored hardware reporting


## [4.8.14] 2026-05-21

### Fixed

- optional Maize config and log


## [4.8.13] 2026-05-21

### Changed

- stereochemistry support for classical (RNN) Reinvent prior


## [4.8.12] 2026-05-07

### Added

- log-level verbose for logging all sampled SMILES strings


## [4.8.11] 2026-05-05

### Fixed

- When unique\_sequences=true, .output replaces .item2 which is otherwise just a reference


## [4.8.10] 2026-05-04

### Added

- Support for uv


## [4.8.9] 2026-04-14

### Added

- Reagents metadata from the Synthsense SFScore endpoint 


## [4.8.8] 2026-04-10

### Fixed

- ROCs return length safety for RDKit-valid, OE-invalid SMILES


## [4.8.7] 2026-04-10

### Added

- Support for multi-task and multiple featurizers in the Chemprop2 component


## [4.8.6] 2026-04-02

### Added

- Simplify tensorboard logging to write all logs to one file


## [4.8.5] 2026-04-02

### Added

- Chemprop v2 scoring component


## [4.8.4] 2026-04-01

### Added

- CAZP outputs are saved locally for each step 


## [4.8.3] 2026-03-06

### Added

- Check if scoring component does not return scores for each SMILES


## [4.8.2] 2026-02-26

### Fixed

- Improved metadata safety for Maize when execeptions occur in worfklow


## [4.8.1] 2026-02-09

### Added

- Final RDKit conversion check in data pipeline


## [4.8.0] 2026-02-03

### Added

- Peptide enumeration run mode


## [4.7.22] 2026-01-21

### Fixed

- Expand bool variable in multi-process scoring


## [4.7.21] 2026-01-21

### Added

- Report peak GPU memory usage


## [4.7.20] 2026-01-12

### Fixed

- Documentation in ExternalProcess


## [4.7.19] 2026-01-12

### Fixed

- Empty metadata bug in SmilesAssociatedComponentResults


## [4.7.18] 2025-12-03

### Change

- Expanded matching substructure to take lists


## [4.7.17] 2025-11-25

### Added

- Log elapsed days if more than one day


## [4.7.16] 2025-11-24

### Fixed

- Cache which was always off


## [4.7.15] 2025-11-17

### Change

- More robust way to detect legacy JSON from Maize


## [4.7.14] 2025-11-12

### Added

- Variants of penalty functions and a RND-based filter type
￼

## [4.7.13] 2025-11-12

### Added

- SynthSense as CAZP replacement


## [4.7.12] 2025-11-06

### Added

- version information in --help


## [4.7.11] 2025-11-03

### Changed

- Maize step counter starts at 1

### Removed

- Maize does not pass on names anymore (are not used in ReinventEntry)


## [4.7.10] 2025-10-31

### Added

- iSIM support for TL


## [4.7.9] 2025-10-24

### Fixed

- Saver, conservative SMILES check to avoid later problems in scoring components


## [4.7.8] 2025-10-17

### Fixed

- Torch's CPU interface is not the same as for the GPU e.g. get\_device\_name() does not exist for the CPU


## [4.7.7] 2025-10-10

### Added

- Support reporting on XPU architetures


## [4.7.6] 2025-10-07

### Added

- No cache feature for scoring components e.g. CAZP in some settings


## [4.7.5] 2025-10-07

### Changed

- New install procedure using a Python script to construct pip command line

### Removed

- JSON configuration files

### Added

- Simple script to convert TOML to JSON
- CAZP support files


## [4.7.4] 2025-10-07

### Fixed

- Wrong str converion in inception


## [4.7.3] 2025-10-07

### Fixed

- Tests for RDKit 2025 


## [4.7.2] 2025-10-03

### Added

- Multi-score support for maize


## [4.7.1] 2025-10-02

### Added

- Metadata and QoL improvements to ring precedence scoring component


## [4.7.0] 2025-09-19

### Added

- Inception for all generators


## [4.6.27] 2025-08-28

### Fixed

- Fixed list comprehension bug in mask smiles indexing


## [4.6.26] 2025-08-07

### Fixed

- Fixed bug when multiple warheads generate the same fragment smiles in LinkInvent, tripping up score caching


## [4.6.25] 2025-08-06

### Fixed

- typo in variable name in logging for invalid tokens


## [4.6.24] 2025-07-19

### Added

- Moved in CAZP to match public release


## [4.6.23] 2025-07-15

### Fixed

- Small fix for peptide camsol scoring componenet


## [4.6.22] 2025-07-01

### Fixed

- Do not skip categorical values in raw scores for remote monitoring


## [4.6.21] 2025-06-12

### Fixed

- Metadata is now printed in evaluate scoring jobs as well 


## [4.6.20] 2025-06-11

### Added

- CamSol scoring component for peptide solubility prediction


## [4.6.19] 2025-06-03

### Fixed

- skip warning message if RL terminates normally


## [4.6.18] 2025-05-21

### Fixed

- beamsearch had an unsafe integer division

## [4.6.17] 2025-05-13

### Added

- metadata write out to output csv
- add custom alert matching pattern metadata


## [4.6.16] 2025-05-12

### Added

- optional InchiKey deduplication for data pipeline


## [4.6.15] 2025-05-06

### Added

- batched sampling for classical Reinvent


## [4.6.14] 2025-04-30

### Added

- output SMILES state in sampling runmode


## [4.6.13] 2025-04-28

### Added

- token check: leads to termination if invalid token is found, for all models


## [4.6.12] 2025-04-28

### Added

- GPU memory reporting for TL


## [4.6.11] 2025-04-12

### Fixed

- json\_config\_path is None if not explicitly set in the config file


## [4.6.10] 2025-04-12

### Added

- Feature parameter for ChemProp scoring component


## [4.6.9] 2025-04-03

### Changed

- Check against validated config rather than original input config dict


## [4.6.8] 2025-04-02

### Changed

- Multiple endpoints for ExternalProcess scoring component


## [4.6.7] 2025-04-01

### Fixed

- Enable ROCS in multiprocessing environment


## [4.6.6] 2025-04-01

### Added

- Ring system precedence based scoring component 


## [4.6.5] 2025-04-01

### Fixed

- Fragment & annotated smiles fixes reimplemented
- Libinvent tensorboard logging cleaned SMILES instead of labeled SMILES


## [4.6.4] 2025-04-01

### Fixed

- Fragment & annotated smiles fixes reimplemented


## [4.6.3] 2025-03-31

### Fixed

- Parallel scoring enabled


## [4.6.2] 2025-03-31

### Fixed

- Sampling SMARTS filters indexing 


## [4.6.1] 2025-03-28

### Fixed

- LibInvent handling of annotated SMILES in score caching


## [4.6.0] 2025-03-25

### Added

- iSIM similarity support for RL


## [4.5.27] 2025-03-25

### Fixed

- Data pipeline regex filter checks for carbons correctly


## [4.5.26] 2025-03-21

### Fixed

- Fragment scores are now correctly associated the full smiles in RL


## [4.5.25] 2025-03-20

### Added

- Split fillers for PepInvent in log and CSV files


## [4.5.24] 2025-03-20

### Added

- Optional -1 offset for training loss plot in TB


## [4.5.23] 2025-02-21

### Fixed

- Properly set seed: command line precedes config setting


## [4.5.22] 2025-02-14

### Fixed

- Ignore extra fillers when PepInvent got stuck in generating more fillers than needed


## [4.5.21] 2025-02-11

### Fixed

- Various typos and clarification in documentation


## [4.5.20] 2025-02-11

### Added

- Optional SMARTS filter for sampling


## [4.5.19] 2025-02-11

### Changed

- Clean-up in datapipeline regex


## [4.5.18] 2025-01-31

### Changed

- Added `weights_only=False` to all torch.load calls


## [4.5.17] 2025-01-23

### Changed

- Do no disconnect organic metal compounds in data pipeline


## [4.5.16] 2025-01-23

### Added

- Skip scoring if failure is encountered anywhere in Maize


## [4.5.15] 2024-12-18

### Added

- SMILES dictionary datastructure for component results


## [4.5.14] 2024-12-04

### Added

- Guard again atom kekulization exception in data pipeline


## [4.5.13] 2024-11-28

### Fixed

- Set Mol to be None when the number of generated fillers does not match the input masked slots in Pepinvent.


## [4.5.12] 2024-11-27

### Added

- Add a scoring function for calculating the number of atoms in the largest ring


## [4.5.11] 2024-11-18

### Changed

- Convert float nan and infs to valid json format before remote reporting


## [4.5.10] 2024-11-16

### Added

- optional tautomer canonicalisation in data pipeline


## [4.5.9] 2024-11-07

### Fixed

- read configuration file from stdin


## [4.5.8] 2024-11-07

### Changed

- refactor of top level code


## [4.5.7] 2024-11-07

### Fixed

- check if DF is set


## [4.5.6] 2024-11-05

### Added

- YAML configuration file reader


## [4.5.5] 2024-11-05

### Added

- Logging of configuration file absolute path

### Changed

- Automatic configuration file format detection


## [4.5.4] 2024-10-28

### Added

- Exponential decay transform

### Fixed

- Ambiguity in parsing optional parameters with multiple endpoints and multiple optional parameters


## [4.5.3] 2024-10-23

### Added

- component-level parameters for scoring components


## [4.5.2] 2024-10-23

### Added

- executable module: can run `python -m reinvent`


## [4.5.1] 2024-10-23

### Added

- SIGUSR1 for controlled termination


## [4.5.0] 2024-10-08

### Added

- PepInvent in Sampling and Staged learning mode with example toml config provided 
- PepInvent prior


## [4.4.37] 2024-10-07

### Fixed

- Atom map number removal for Libinvent sampling dropped SMILES


## [4.4.36] 2024-09-27

### Added

- Stage number for JSON to remote monitor

### Changed

- Relaxed dependencies


## [4.4.35] 2024-09-26

### Added

- Terminate staged learning on SIGTERM and check if running in multiprocessing environment

### Changed

- ValueError for all scoring components such that the staged learning handler can handle failing components


## [4.4.34] 2024-09-16

### Fixed

- SMILES in DF memory were wrongly computed


## [4.4.33] 2024-09-14

### Fixed

- run-qsartuna.py: convert ndarray to list to make it JSON serializble


## [4.4.32] 2024-09-13

### Fixed

- PMI component: check for embedding failure in RDKit's conformer generator


## [4.4.31] 2024-09-13

### Fixed

- Dockstream component wrongly quoted the SMILES string
- Diversity filter setup in config was ignored


## [4.4.30] 2024-09-12

### Fixed

- Fixed config reading bug for DF


## [4.4.29] 2024-09-05

### Changed

- Changed Molformer sampling valid and unique from percentage to fraction on tensorboard


## [4.4.28] 2024-08-29

### Fixed

- Fixed incorrect tanimoto similarity log in Mol2Mol sampling mode


## [4.4.27] 2024-07-23

### Fixed

- Corrected typo in Libinvent report


## [4.4.26] 2024-07-21

### Fixed

- Report for sampling returned np.array which is incompatibile with JSON serialization


## [4.4.25] 2024-07-19

### Fixed

- Allowed responder as an optional input in scoring input validation


## [4.4.24] 2024-07-19

### Fixed

- Fixed remote for Libinvent
- Batchsize defaults to 1 for TL


## [4.4.23] 2024-07-18

### Fixed

- Added temperature parameter in Sampling and RL config validation 


## [4.4.22] 2024-07-10

### Fixed

- Scalar return value from make\_grid\_image()


## [4.4.21] 2024-07-10

### Changed

- Removed labels from Libinvent generated molecules before passing to scoring components: maize, icolos, dockstream and ROCS
- Write out clean SMILES without labels to CSV and tensorboard in staged learning and sampling run modes
- Simplified update code for Libinvent RL


## [4.4.20] 2024-07-10

### Fixed

- Added missing layer\_normalization when getting Reinvent RNN model parameters


## [4.4.19] 2024-07-08

### Fixed

- Log likelihood calculation is handled more efficiently


## [4.4.18] 2024-07-05

### Fixed

- Added missing tag to legacy TanimotoDistance component


## [4.4.17] 2024-07-05

### Added

- Custom RDKit normalization transforms hard-coded or from file


## [4.4.16] 2024-07-01

### Fixed

- Safe-guard against invalid SMILES in SMILES file/CSV reading


## [4.4.15] 2024-06-27

### Added

- Allow dot SMILES fragment separator for Lib/Linkinvent input


## [4.4.14] 2024-06-27

### Changed

- Renamed pipeline options keep\_isotopes to keep\_isotope\_molecules

### Added

- New data pipeline options: uncharge, kekulize, randomize\_smiles


## [4.4.13] 2024-06-27

### Fixed

- Parallel logging in data pipeline preprocessor (partial only)


## [4.4.12] 2024-06-25

### Added

- Libinvent based on unified Transformer

### Fixed

- Error handling for unsupported tokens in RNN-based Libinvent


## [4.4.11] 2024-06-17

### Added

- Parallel processing of regex and rdkit filters

### Chamged

- RDKit filter uses simpler functions not compound functions


## [4.4.10] 2024-06-05

### Changed

- RL max\_score is optional and has a default of 1.0


## [4.4.9] 2024-06-05

### Added

- Parallel implementation of chemistry filter in data pipeline


## [4.4.8] 2024-06-05

### Fixed

- Added multinomial as default sampling strategy for Transformer. 


## [4.4.7] 2024-06-04

### Changed

- Chemistry filter: customizable normalization, do not use RDKit logger


## [4.4.6] 2024-06-04

### Changed

- Reworked remote responder


## [4.4.5] 2024-06-03

### Changed

- Reporting for sampling runmode


## [4.4.4] 2024-06-03

### Fixed

- Minor fix to DF validation


## [4.4.3] 2024-06-03

### Added

- Chemistry filter for data pipeline


## [4.4.2] 2024-05-30

### Fixed

- Check if agent is in RL state info to avoid unnecessary exception


## [4.4.1] 2024-05-30

### Changed

- Toplevel validation of config file to detect extra sections


## [4.4.0] 2024-05-30

### Added

- Initial support for data pipeline


## [4.3.26] 2024-05-30

### Added

- Example scoring component script for RAScore
- Reimplentation of Conversion class for backward compatibility


## [4.3.25] 2024-05-29

### Fixed

- Various fixes in TLRL notebook


## [4.3.24] 2024-05-27

### Changed

- Scoring runmode now also allows import of scoring components


## [4.3.23] 2024-05-25

### Changed

- More consistent JSON config writing (includes imported scoring functions)
- Better handling of value mapping


## [4.3.22] 2024-05-24

### Fixed

- All scoring components need to compute the number of endpoints, added where sensible


## [4.3.21] 2024-05-24

### Fixed

- Filter out invalid fragments for Lib/Linkinvent


## [4.3.20] 2024-05-22

### Changed

- Lifted static methods to module level functions

### Removed

- More chemistry code


## [4.3.19] 2024-05-21

### Removed

- Unused chemistry code and associated tests


## [4.3.18] 2024-05-17

### Added

- Chained reporters for RL

### Fixed

- Compatibility support for model file metadata: dict vs dataclass


## [4.3.17] 2024-05-15

### Changed

- Various cosmetic fixes to TB output

### Fixed

- TL responder: validation loss was reported as sampled loss
- Add metadata when "metadata" files is empty


## [4.3.16] 2024-05-14

### Changed

- Code clean-up in Reinvent model and RNN code
- Global pydantic configuration

### Fixed

- Affected test cases after code rearrangement


## [4.3.15] 2024-05-13

### Added

Various code improvements.
- Metadata writing for all created RL and TL models
- Chained reporters
- Prior registry
- Config validation


## [4.3.14] 2024-05-06

### Added

- Write additional information to RL CSV
  - For Mol2Mol, add _Input\_SMILEs_
  - For Linkinvent, add _Warheads_ and _Linker_
  - For Libinvnet, add _Input\_Scaffold_ and _R-groups_


## [4.3.13] 2024-05-06

### Added

- Notebook: plot of reverse sigmoid transform


## [4.3.12] 2024-05-06

### Added

- Stages can now define their own diversity filters.  Global filter always overwrites stage settings.  Currently no mechanism to carry over DF from previous stage, use single stage runs.


## [4.3.11] 2024-04-30

### Changed

- TanimotoSimilarity replaces TanimotoDistance

### Deprecated

- TanimotoDistance: computes actually a similarity and TanimotoSimilarity


## [4.3.10] 2024-04-29

### Changed

- ChemProp scoring component now supports multitask models


## [4.3.9] 2024-04-28

### Added

- Optional [scheduler] section for TL


## [4.3.8] 2024-04-24

### Fixed

- LibInvent: fixed issue with multiple R-groups on one atom
- ReactionFilter: selective filter will now function correctly as filter


## [4.3.7] 2024-04-22

### Added

- Notebook: a more complete RL/TL demo


## [4.3.6] 2024-04-22

### Fixed

- Fixed crash when all molecules in batch are filtered out


## [4.3.5] 2024-04-18

### Changed

- Code clean-up in create\_adapter()

### Fixed

- Import mol2mol vocabulary rather than copying the file


## [4.3.4] 2024-04-16

### Added

- Write invalid SMILES unchanged to RL CSV


## [4.3.3] 2024-04-16

### Added

- Notebook: demo on how to analyse RL CSV


## [4.3.2] 2024-04-15

### Added

- Dataclass validation for scoring component parameters

### Fixed

- Datatype in MatchingSubstructure's Parameters: only single SMARTS is allowed


## [4.3.1] 2024-04-15

### Added

- Notebook to demo simple RL run, TensorBoard visualisation and TensorBoard data extraction.


## [4.3.0] 2024-04-15

### Added

- Linkinvent based on unified Transformer model supported by RL and sampling.  Both beam search and multinomial sampling are implemented.


## [4.2.13] 2024-04-12

### Fixed

- downgraded Chemprop to 1.5.2 and sklearn to 1.2.2 to retain backward compatibility


## [4.2.12] 2024-04-10

### Changed

- New default torch device setup from PyTorch 2.x

### Added

- Config parameter "device" to explicitly set torch device e.g. "cuda:0"


## [4.2.11] 2024-04-08

### Fixed

- Fixed unknown token handling for Mol2mol TL


## [4.2.10] 2024-04-05

### Fixed

- Fixed dataloader for TL to use incomplete batch 


## [4.2.9] 2024-04-04

### Fixed

- Skip hash check in metadata if no metadata in model file


## [4.2.8] 2024-04-03

### Added

- Mol2Mol supports unknown tokens for all the priors



## [4.2.7] 2024-03-27

### Added

- Optional randomization in all TL epochs for Reinvent


## [4.2.6] 2024-03-21

### Fixed

- Return from make\_grid\_image()


## [4.2.5] 2024-03-20

### Added

- Log network parameters


## [4.2.4] 2024-03-20

### Changed

- Reworked TL code: clean-up, image layout, graph for Reinvent

### Added

- Added options and better statistics for TL: valid SMILES, duplicates
- Disableable standardization

### Removed

- KL divergence in TB output

### Fixed

- Batch size calculation


## [4.2.3] 2024-03-14

### Fixed

- Vocabulary for mol2mol and reinvent is saved as dictionary


## [4.2.2] 2024-03-13

### Fixed

- Sigmoid functions in scoring have now a stable implementation


## [4.2.1] 2024-03-12

### Removed

- Long chain (SMARTS for 5 aliphatic carbons) check


## [4.2.0] 2024-03-08

### Added

- Unified transformer code to faciliate new model designs


## [4.1.16] 2024-03-07

### Fixed

- filters now apply to transformed scores 


## [4.1.15] 2024-03-07

### Fixed

- Minor change inception filter: cleaner way of handling internal SMILES store


## [4.1.14] 2024-03-06

### Added

- Updated script for creating an empty classical Reinvent model

### Fixed

- Memory bug in TL related to similarity calculation: made this optional


## [4.1.13] 2024-03-05

### Added

- Allowed runs with only filter/penalty components 


## [4.1.12] 2024-03-05

### Added

- Better logging for Reinvent standardizer


## [4.1.11] 2024-02-27

### Fixed

- Inception filters out SMILES containing tokens that are not compatible with the prior


## [4.1.10] 2024-02-26

### Fixed

- Numerically stable double sigmoid implementation
- Number of CPUs for TL (Mol2Mol) is now 1


## [4.1.9] 2024-02-22

### Fixed

- Save models with no metadata


## [4.1.8] 2024-02-20

### Fixed

- Tensorboard histogram bug fixed again


## [4.1.7] 2024-02-20

### Fixed

- TL is now running for the expected `num_epochs`


## [4.1.6] 2024-02-19

### Fixed

- Get model\_type from save\_dict prior model correctly


## [4.1.5] 2024-02-13

### Fixed

- Staged learning does not allocate GPU memory if device is set to CPU


## [4.1.4] 2024-02-06

### Added

- Prior model files have been tagged with meta data
- Model files read in are checked for integrity


## [4.1.3] 2024-02-06

### Fixed

- Tab reader unit tests now uses mocks for open
- Wite correctly CSV scoring file when from one columns SMILES file


## [4.1.2] 2024-02-04

### Fixed

- Scoring filter components work as filters again


## [4.1.1] 2024-02-02

### Added

- CSV and SMILES file reader for the scoring run mode, will retain all columns form the input and write to output CSV


## [4.1.0] 2024-01-26

### Added

- Tobias Ploetz' (Merck) REINFORCE implementations of the DAP, MAULI and MASCOF RL reward strategies


## [4.0.36] 2024-01-22

### Added

- Check if RDKit descriptor names are valid


## [4.0.35] 2024-01-19

### Fixed

- Filename issue on Windows which lead to termination


## [4.0.34] 2024-01-17

### Added

- General scoring component for all 210 RDKit descriptors


## [4.0.33] 2023-12-14

### Added

- Optional cwd for run\_command()


## [4.0.32] 2023-12-12

### Fixed

- Collect _all_ names for remote monitoring\*


## [4.0.31] 2023-12-08

### Fixed

- Pass data to request as dict rather than a JSON string


## [4.0.30] 2023-12-07

### Fixed

- Pair generator multiprocessing in TL is supported on Linux, Windows, and MacOS

### Changed

- The number of cpus is optional and could be specified in the toml/json configuration file through the parameter `number_of_cpus`


## [4.0.29] 2023-12-07

### Fixed

- Added missing second mask in inception call to scoring function
- Fixed cuda out of memory for Reinvent batch sampling


## [4.0.28] 2023-12-05

### Fixed

- Handle the case when there are no non-cached SMILES and thus the scpring function does not need to run.
- Improved type safety in `value_mapping`


## [4.0.27] 2023-12-01

### Added

- Number of cpus can be specified in toml/json config files for TL jobs


## [4.0.26] 2023-12-01

### Fixed

- Check for CUDA before checking GPU memory otherwise will fail on CPU
- Removed obsolete code which broke TL with Reinvent
- Windows support: correct signal handling


## [4.0.25] 2023-11-29

### Added

- Scoring component MolVolume to compure molecular volume via RDKit


## [4.0.24] 2023-11-28

### Changed

- Minimal SMILES pre-processing for scoring to allow keeping of stereochemistry and only choose largest fragment, and use the general RDKit cleanup/sanitation/hydrogen.  Skip heavy filtering on molecules size, allowed atoms, tokens, vocabulary, etc.  This faciliates situation where only scoring is desired.


## [4.0.23] 2023-11-28

### Changed

- Allow zero weights to only display a component score.  This will have no effect on aggregation but the component score is still computed. So, be careful with computationally expensive components.


## [4.0.22] 2023-11-28

### Changed

- Flag to purge diversity filter memories after each staged learning stage.  This is useful in multiple stage runs and is equivalent to `use_checkpoint` for single stage reruns.


## [4.0.21] 2023-11-23

### Changed

- The CSV file from RL has controlled output precision: 7 for total score and transformed scores, 4 for all other floating point values

### Fixed

- Critical: all scores of duplicate SMILES including the first occurence where set to zero rather than the computed value
- Scores of duplicates are now correctly copied over from first occurence


## [4.0.20] 2023-11-20

### Fixed

- All tests support both cpu and gpu


## [4.0.19] 2023-11-20

### Changed

- Contidional import of `resource` to allow running on Windows


## [4.0.18] 2023-11-17

### Added

- Some rudimentary information on GPU memory usage in staged learning


## [4.0.17] 2023-11-17

### Fixed

- Bug in sampling related to the way the sampled.nlls object was treated. Now it is always pytorch tensor object without gradient on cpu


## [4.0.16] 2023-11-15

### Fixed

- Moved TPSA to separate component to enable TPSA calculation for polar S and P, the original RDKit implementation does not consider those atoms and the default is still to leave those out from TPSA calculation


## [4.0.15] 2023-11-15

### Fixed

- Issue with NaNs being in the raw and transformed code that would not allow to compute the mean


## [4.0.14] 2023-11-13

### Fixed

- Explicit serialization of JSON string because the internal one from requests may fail


## [4.0.13] 2023-11-09

### Fixed

- Added a patch to fix a bug on the native implementation of Pytorch related to the histogram functionality of the Tensorboard report


## [4.0.12] 2023-11-09

### Changed

- Added a check which raises an exception if the user enters scaffolds with multiple attachement points connected to the same atom (Libinvent, Linkinvent). This
will be lifted in a future update

### Fixed

- Fixed report format (TL)


## [4.0.11] 2023-11-02

### Fixed

- Normalize SMILES before passing to string-based scoring component because the SMILES may still contain lables (Libinvent)


## [4.0.10] 2023-10-27

### Fixed

- Fixed fragment effective length ratio error when fragment has single atom (e.g. "[*]N[*]") with max graph length=0 


## [4.0.9] 2023-10-25

### Changed

- For Compound Sampling, removed explicit dependency on matplotlib: report histgram and scatter plot to tensorboard only if matplotlib is available. 


## [4.0.8] 2023-10-23

### Changed

- For Compound Sampling, introduced a new parameter unique\_molecules to do canonicalize smiles deduplication instead of using sequemce deduplication which can be confusing


## [4.0.7] 2023-10-20

### Deprecated

- warn if sequence deduplication is requested


## [4.0.6] 2023-10-18

### Fixed

- TL integration test fixed (no impact on GUI or core)


## [4.0.5] 2023-10-17

### Added

- TL reporting of epochs

### Changed

- multiple scoring components are reported as means again rather than as lists


## [4.0.3] 2023-10-16

### Fixed

- fixed graph length fragment component


## [4.0.2] 2023-10-06

### Fixed

- fixes for fragment scoring components


## [4.0.1] 2023-10-03

### Changed

- scores reported for filters and penalties as in REINVENT3


## [4.0.0] 2023-09-12

### Added

- initial release of REINVENT4

### Deprecated
- RL rewards MASCOF, MAULI, SDAP (inefficient in practice)
- sequence deduplication (interferes with diversity filter and SMILES deduplication)


## Older Releases

### [3.2], [3.1], [3.0], [2.1], [2.0]

[Reinvent on GitHub](https://github.com/MolecularAI/Reinvent)
