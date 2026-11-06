# Scoring Components in REINVENT4

List of supported scoring components with their parameters, plus the
transformation and aggregation functions available in the scoring framework.

Component names are case-insensitive and hyphen/underscore-insensitive when
used as TOML keys (e.g. `TanimotoSimilarity`, `tanimoto_similarity` and
`tanimotosimilarity` all resolve to the same component).  The canonical names
listed here match the class names in the source.

All component parameters are implemented as lists because a single component block can define
multiple endpoints.  Even when only one endpoint is configured, each parameter
value must be a single-element list (the framework collects them automatically
from the per-endpoint `params.*` keys).


---

## Basic molecular physical properties (RDKit, no parameters)

These components accept no parameters.

| Component name           | Description                                           |
|--------------------------|-------------------------------------------------------|
| `Qed`                    | Quantitative Estimate of Drug-likeness (RDKit).       |
| `MolecularWeight`        | Molecular weight (RDKit `MolWt`).                     |
| `TPSA`                   | Topological polar surface area — see parameters below.|
| `GraphLength`            | Maximum topological distance in the molecular graph.  |
| `NumAtomStereoCenters`   | Number of stereocenters.                              |
| `HBondAcceptors`         | Number of H-bond acceptors (Lipinski).                |
| `HBondDonors`            | Number of H-bond donors (Lipinski).                   |
| `NumRotBond`             | Number of rotatable bonds.                            |
| `Csp3`                   | Fraction of sp³ carbons.                              |
| `numsp`                  | Number of sp-hybridized atoms.                        |
| `numsp2`                 | Number of sp²-hybridized atoms.                       |
| `numsp3`                 | Number of sp³-hybridized atoms.                       |
| `NumHeavyAtoms`          | Number of heavy atoms.                                |
| `NumHeteroAtoms`         | Number of heteroatoms.                                |
| `NumRings`               | Total ring count.                                     |
| `NumAromaticRings`       | Number of aromatic rings.                             |
| `NumAliphaticRings`      | Number of aliphatic rings.                            |
| `LargestRingSize`        | Number of atoms in the largest ring.                  |
| `SlogP`                  | Crippen octanol-water partition coefficient.          |


### TPSA — parameters

| Parameter      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `includeSandP` | `false` (default) — if `true`, include sulfur and phosphorus in PSA calc.  |


### PMI — Principal Moments of Inertia

| Parameter  | Description                                              |
|------------|----------------------------------------------------------|
| `property` | `"npr1"` or `"npr2"` to select the normalized PMI index. |

Use two endpoints (one for `npr1`, one for `npr2`) to assess 3D shape diversity.


### MolVolume — Molecular volume

| Parameter     | Default | Description                                   |
|---------------|---------|-----------------------------------------------|
| `grid_spacing`| `0.2`   | Grid spacing in Ångstroms for volume grid.    |
| `box_margin`  | `2.0`   | Margin around the molecule bounding box (Å).  |

Requires a 3D conformer; uses RDKit embedding.


### RDKitDescriptors — arbitrary RDKit descriptors

| Parameter    | Description                                                                    |
|--------------|--------------------------------------------------------------------------------|
| `descriptor` | Name of an RDKit descriptor from `rdkit.Chem.Descriptors` (case-insensitive). |

One endpoint per descriptor.


---

## Similarity and cheminformatics components

### TanimotoSimilarity

Tanimoto similarity using Morgan fingerprints.  `TanimotoDistance` is a
deprecated alias.

| Parameter     | Description                                                         |
|---------------|---------------------------------------------------------------------|
| `smiles`      | List of reference SMILES to compare against.                        |
| `radius`      | Morgan fingerprint radius.                                          |
| `use_counts`  | If `true`, use count fingerprints.                                  |
| `use_features`| If `true`, use pharmacophore-like features.                         |
| `aggregation` | `"max"` (default), `"mean"`, or `"min"` over the reference set.    |

Multiple endpoints are supported; each endpoint defines its own reference set
and fingerprint settings.


### PathFPSimilarity

Tanimoto similarity using RDKit path-based (topological) fingerprints
(`RDKFingerprint`).  Path fingerprints enumerate all linear bond paths up to
`max_path` bonds, capturing chain and path-based structural features differently
from circular Morgan fingerprints.

| Parameter     | Default | Description                                                      |
|---------------|---------|------------------------------------------------------------------|
| `smiles`      | required| List of reference SMILES to compare against.                     |
| `min_path`    | required| Minimum path length in bonds.                                    |
| `max_path`    | required| Maximum path length in bonds.                                    |
| `n_bits`      | required| Fingerprint bit-vector size.                                     |
| `aggregation` | `"max"` | `"max"`, `"mean"`, or `"min"` over the reference set.           |

Multiple endpoints are supported; each endpoint defines its own reference set
and fingerprint settings.


### CustomAlerts  *(filter)*

Filters out molecules that match any of the provided SMARTS patterns.  Returns
0.0 for any matching molecule; 1.0 otherwise.  Applied globally before
aggregation — no weight.

| Parameter | Description                               |
|-----------|-------------------------------------------|
| `smarts`  | List of unwanted SMARTS patterns.         |


### GroupCount

Counts how many times a SMARTS pattern is found in the molecule.

| Parameter | Description                               |
|-----------|-------------------------------------------|
| `smarts`  | SMARTS pattern to count.                  |


### MatchingSubstructure  *(penalty)*

Returns 1.0 if any pattern matches, 0.5 if no pattern matches.  Used to
preserve a score when a desired substructure is present, or penalize its absence.

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `smarts`        | SMARTS pattern(s) to match.                                      |
| `use_chirality` | If `true`, consider chirality in substructure matching.          |


### MMP — Matched Molecular Pairs

Detects matched molecular pairs between generated and reference compounds.
Returns the string `"MMP"` or `"No MMP"`; use with the `value_mapping` transform
to convert to a numeric score.

| Parameter              | Default | Description                                              |
|------------------------|---------|----------------------------------------------------------|
| `reference_smiles`     | required| List of reference SMILES.                                |
| `num_of_cuts`          | `1`     | Number of bonds cut during fragmentation.                |
| `max_variable_heavies` | `40`    | Maximum heavy-atom count change in an MMP.               |
| `max_variable_ratio`   | `0.33`  | Maximum ratio of variable to total heavy atoms in an MMP.|


### RingPrecedence

Scores molecules based on the empirical likelihood of their ring systems,
using a pre-computed ring NLL database.

| Parameter       | Description                                                                          |
|-----------------|--------------------------------------------------------------------------------------|
| `database_file` | Path to the JSON ring database (prepare with `support/compute_database_precedence.py`).|
| `nll_method`    | `"total"` (sum of all ring NLLs) or `"max"` (highest single ring NLL).              |
| `make_generic`  | If `true`, use generic ring SMILES; default `false`.                                 |


---

## Physics / structure / ligand-based components

### ROCSSimilarity  *(requires OpenEye toolkit)*

3D shape and chemical similarity via OpenEye ROCS.

| Parameter           | Default    | Description                                                               |
|---------------------|------------|---------------------------------------------------------------------------|
| `rocs_input`        | required   | Reference molecule file (SDF or SQ format).                               |
| `color_weight`      | `0.5`      | Weight for the color (pharmacophore) overlay score.                       |
| `shape_weight`      | `0.5`      | Weight for the shape overlay score.                                       |
| `max_stereocenters` | —          | Maximum number of stereocenters to enumerate.                             |
| `ewindow`           | —          | Energy window for conformer generation (kJ/mol).                          |
| `maxconfs`          | —          | Maximum number of conformers per compound.                                |
| `similarity_measure`| required   | `"Tanimoto"`, `"RefTversky"` or `"FitTversky"`.                          |
| `custom_cff`        | —          | Path to a custom ROCS color force field (optional).                       |


### DockStream  *(superseded by MAIZE)*

Generic docking interface supporting AutoDock Vina, rDock, OpenEye Hybrid,
Schrödinger Glide and CCDC GOLD via
[DockStream](https://github.com/MolecularAI/DockStream).

| Parameter              | Description                                                       |
|------------------------|-------------------------------------------------------------------|
| `configuration_path`   | Path to the DockStream JSON configuration file.                   |
| `docker_script_path`   | Path to the DockStream `AZdock/docker.py` script.                 |
| `docker_python_path`   | Python interpreter with DockStream installed.                     |


### Icolos  *(superseded by MAIZE)*

Generic workflow interface to [Icolos](https://github.com/MolecularAI/Icolos).

| Parameter     | Description                                       |
|---------------|---------------------------------------------------|
| `name`        | Name of the property to extract from Icolos output.|
| `executable`  | Path to the Icolos executable.                    |
| `config_file` | Path to the Icolos JSON configuration file.       |


### Maize

Generic interface to [MAIZE](https://github.com/MolecularAI/maize) workflows.

| Parameter       | Default | Description                                                              |
|-----------------|---------|--------------------------------------------------------------------------|
| `executable`    | required| Path to the Maize executable.                                            |
| `workflow`      | required| Path to the Maize workflow file (YAML/TOML/JSON).                        |
| `property`      | required| Name of the property to extract from Maize output.                       |
| `debug`         | `false` | Run Maize with verbose debug logging.                                    |
| `keep`          | `false` | Keep intermediate Maize files.                                           |
| `log`           | —       | Path for the Maize log file (optional).                                  |
| `config`        | —       | Path to a custom Maize system config file (optional).                    |
| `parameters`    | `{}`    | Dictionary of workflow parameter overrides.                              |
| `skip_normalize`| `false` | Skip SMILES normalization before passing to Maize.                       |
| `skip_on_failure`| `true` | Return zero scores instead of raising an error on Maize failure.         |
| `pass_fragments`| `false` | Pass fragmented SMILES (for LibInvent / LinkInvent) to the workflow.     |


---

## QSAR/QSPR model components

### ChemProp  *(ChemProp v1.x)*

Predictions from ChemProp D-MPNN models (v1.x only). Consider using v2.x (see below).

| Parameter              | Description                                                               |
|------------------------|---------------------------------------------------------------------------|
| `checkpoint_dir`       | Directory containing the ChemProp v1 model checkpoint.                   |
| `rdkit_2d_normalized`  | *Deprecated* — use `features` instead.                                   |
| `features`             | Feature type string, e.g. `""`, `"rdkit_2d_normalized"` or `"morgan"`.  |
| `target_column`        | Column name to extract for multi-task models.                             |


### ChemProp2  *(ChemProp v2.x)*

Predictions from ChemProp D-MPNN models (v2.x only).  Preferred ChemProp version.

| Parameter             | Description                                                                          |
|-----------------------|--------------------------------------------------------------------------------------|
| `model_path`          | Path to a `.pt` model file or a directory of `.pt` files for ensemble averaging.    |
| `featurizers`         | List of featurizer type names (e.g. `["V1RDKit2D"]`; omit for graph-only models).   |
| `featurizers_options` | List of option dicts for each featurizer (e.g. `[{}]`).                              |
| `target_column`       | Column name for multi-task models; required when the model has more than one task.   |


### Qptuna

Predictions from Qptuna pickle-serialized QSAR models.

| Parameter    | Description                              |
|--------------|------------------------------------------|
| `model_file` | Path to the Qptuna `.pkl` model file.   |


---

## Drug-likeness, synthesizability and reaction components

### SAScore

Ertl–Schuffenhauer synthetic accessibility score.  Returns a value in [1, 10]
(lower = easier to synthesize).  Based on https://doi.org/10.1186/1758-2946-1-8.
No parameters.


### SynthSense / CAZP

Interface to [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) via
an external command wrapper.  `CAZP` is a backward-compatible alias.

| Parameter                   | Description                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| `number_of_steps`           | Maximum number of retrosynthesis expansion steps.                                             |
| `time_limit_seconds`        | Time limit per molecule.                                                                      |
| `stock`                     | Stock configuration dict (overrides config file stock).                                       |
| `scorer`                    | AiZynthFinder scorer configuration dict.                                                     |
| `stock_profile`             | Named stock profile from the config file.                                                     |
| `reactions_profile`         | Named reactions profile from the config file.                                                 |
| `score_to_extract`          | Name of the AiZynthFinder score to return as the component score.                             |
| `reference_route_file`      | Reference route file for route-similarity scoring.                                            |
| `popularity_threshold`      | Threshold for the route-popularity endpoint.                                                  |
| `penalty_multiplier`        | Penalty multiplier for the route-popularity endpoint.                                         |
| `consider_subroutes`        | If `true`, include sub-routes in the popularity calculation.                                  |
| `min_subroute_length`       | Minimum sub-route length to consider.                                                         |
| `penalize_subroutes`        | Sub-route penalization mode.                                                                  |
| `bucket_threshold`          | Bucket threshold for fill-a-plate scoring.                                                    |
| `min_steps_for_penalization`| Minimum RL steps before fill-a-plate penalization activates.                                  |
| `penalization_enabled`      | Enable/disable fill-a-plate penalization.                                                     |


### ReactionFilter  *(filter, LibInvent / LinkInvent only)*

Filters molecules based on reaction feasibility.  Only works with molecules
that carry attachment-point labels (LibInvent scaffolds or LinkInvent warheads).

| Parameter         | Description                                                                     |
|-------------------|---------------------------------------------------------------------------------|
| `type`            | Filter type: `"selective"`, `"nonselective"` or `"definedselective"`.          |
| `reaction_smarts` | List of RDKit reaction SMARTS patterns defining allowed reactions.              |


---

## Generic / external scoring components

### ExternalProcess

Calls an external executable for scoring.  SMILES are passed on stdin; the
executable must return JSON on stdout in the form
`{"payload": {"<property>": [score, ...]}}`.

| Parameter    | Description                                                            |
|--------------|------------------------------------------------------------------------|
| `executable` | Path to the scoring executable.                                        |
| `args`       | Command-line arguments string passed to the executable.                |
| `property`   | Key to extract from the JSON payload (one per endpoint).               |


### REST

Generic REST scoring interface (contributed by Syngenta).  Expects the server
to return JSON in the form `{"version": 1, "payload": {"<property>": [score, ...]}}`.

| Parameter           | Description                              |
|---------------------|------------------------------------------|
| `server_url`        | Base URL of the server.                  |
| `server_port`       | Port number.                             |
| `server_endpoint`   | API endpoint path.                       |
| `predictor_id`      | Predictor identifier sent in the request.|
| `predictor_version` | Predictor version sent in the request.   |
| `header`            | Optional custom HTTP header string.      |


---

## LinkInvent fragment descriptors

The following components compute physico-chemical descriptors on the linker
fragment generated by LinkInvent (attachment points are capped with H before
calculation).  No parameters.

| Component name                  | Description                                         |
|---------------------------------|-----------------------------------------------------|
| `FragmentQed`                   | QED of the fragment.                                |
| `FragmentMolecularWeight`       | Molecular weight of the fragment.                   |
| `FragmentTPSA`                  | TPSA of the fragment.                               |
| `FragmentNumAtomStereoCenters`  | Number of stereocenters.                            |
| `FragmentHBondAcceptors`        | Number of H-bond acceptors.                         |
| `FragmentHBondDonors`           | Number of H-bond donors.                            |
| `FragmentNumRotBond`            | Number of rotatable bonds.                          |
| `FragmentCsp3`                  | Fraction of sp³ carbons.                            |
| `Fragmentnumsp`                 | Number of sp-hybridized atoms.                      |
| `Fragmentnumsp2`                | Number of sp²-hybridized atoms.                     |
| `Fragmentnumsp3`                | Number of sp³-hybridized atoms.                     |
| `FragmentEffectiveLength`       | Shortest path between attachment points.            |
| `FragmentGraphLength`           | Maximum topological distance in the fragment graph. |
| `FragmentLengthRatio`           | Effective length / graph length × 100.              |
| `FragmentNumHeavyAtoms`         | Number of heavy atoms.                              |
| `FragmentNumHeteroAtoms`        | Number of heteroatoms.                              |
| `FragmentNumRings`              | Total ring count.                                   |
| `FragmentNumAromaticRings`      | Number of aromatic rings.                           |
| `FragmentNumAliphaticRings`     | Number of aliphatic rings.                          |
| `FragmentSlogP`                 | Crippen SlogP of the fragment.                      |


---

## Transformation functions

Transforms are applied per endpoint to map raw component output into the
[0, 1] score range expected by the aggregation step.

### `sigmoid`

Smooth S-curve centred between `low` and `high`.

| Parameter | Description                                           |
|-----------|-------------------------------------------------------|
| `low`     | Lower bound of the transition region.                 |
| `high`    | Upper bound of the transition region.                 |
| `k`       | Steepness coefficient (scaled internally by 10).      |


### `reverse_sigmoid`

Inverse of `sigmoid` — scores are high outside [`low`, `high`].

Same parameters as `sigmoid`.


### `double_sigmoid`

Two-sided sigmoid; scores peak at 1.0 within [`low`, `high`] and decay toward
0.0 outside that range.

| Parameter  | Default | Description                                        |
|------------|---------|----------------------------------------------------|
| `low`      | required| Lower bound of the desired range.                  |
| `high`     | required| Upper bound of the desired range.                  |
| `coef_div` | `100.0` | Denominator coefficient controlling the peak width.|
| `coef_si`  | `150.0` | Steepness coefficient at the lower inflection.     |
| `coef_se`  | `150.0` | Steepness coefficient at the upper inflection.     |


### `right_step`

Returns 1.0 if the value ≥ `high`, else 0.0.

| Parameter | Description  |
|-----------|--------------|
| `high`    | Step threshold.|


### `left_step`

Returns 1.0 if the value ≤ `low`, else 0.0.

| Parameter | Description  |
|-----------|--------------|
| `low`     | Step threshold.|


### `step`

Returns 1.0 if `low` ≤ value ≤ `high`, else 0.0.

| Parameter | Description              |
|-----------|--------------------------|
| `low`     | Lower step threshold.    |
| `high`    | Upper step threshold.    |


### `exponential_decay`

Computes `exp(-k * x)`.  Values < 0 are clamped to 1.0.

| Parameter | Description                     |
|-----------|---------------------------------|
| `k`       | Decay rate constant (must > 0). |


### `symmetric_exponential_decay`

Computes `exp(-k * |x - target|)`.  Peaks at exactly 1.0 when x equals
`target` and decays symmetrically in both directions.  Use this instead of
`exponential_decay` when you want to reward values *close to* a specific
target rather than penalise values above zero.

| Parameter | Default | Description                                          |
|-----------|---------|------------------------------------------------------|
| `target`  | required| Reference value at which the score equals 1.0.       |
| `k`       | `1.0`   | Decay rate constant (must > 0); larger k = narrower peak. |


### `value_mapping`

Maps categorical/string outputs (e.g. `"MMP"` / `"No MMP"`) to float scores.
Values not found in the mapping are returned as `NaN`.

| Parameter | Description                                        |
|-----------|----------------------------------------------------|
| `mapping` | Dict mapping string labels to float values [0, 1]. |


---

## Aggregation functions

| Name (TOML)                              | Description                              |
|------------------------------------------|------------------------------------------|
| `arithmetic_mean` / `custom_sum`         | Weighted arithmetic mean; weights are normalized. |
| `geometric_mean` / `custom_product`      | Weighted geometric mean; weights are normalized. |

Weights are specified per endpoint and normalized across all scorer endpoints
before aggregation.  Filter and penalty components do not participate in the
normalization.