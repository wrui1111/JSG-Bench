# JSG Run Provenance

Revision 2026-11-20 (reference geometry `image-aligned-v2`): 45 SID-Set
references (8 JSG-Core, 37 JSG-Scale) whose masks have a different resolution
from their images were previously measured on the mask grid. They are now
mapped to image coordinates; see `revision_reproducibility_20261120.md` and
`experiments/reports/reference_geometry_correction.json`. With model outputs
fixed, strict-primary Joint@0.1 passes become JSG-Core 35 (was 34) and
JSG-Scale 156 (was 157); JSG-Xfer is unaffected (21). Hashes of every artifact
that embeds a SID-Set spatial reference were recomputed on 2026-11-20; the
superseded files are archived under `experiments/runs/frozen_reference_v1/`.

Revision 2026-11-18: corrected Xfer Set matching and added a distinct P/S
diagnostic; exporter, Xfer trace, and criterion-analysis hashes below were
refreshed. Primary strict counts and J-AUC are unchanged. The gate-only replay
under `verification/` is included in the local release package; full native
traces remain separate. See `revision_reproducibility_20261118.md`.

This ledger separates the two 600-report JSG-Scale run families, records the
100-report JSG-Xfer family, and binds the same-runtime routing and view-control
suite used in the manuscript. The two JSG-Scale families use the same frozen
report IDs, but they do not use the same proposal package and are not replicate
estimates. Percentages and hashes from one family must not be substituted for
those from the other.

All SHA-256 values below were recomputed from the retained local artifacts and
last verified on 2026-08-01. Generated requests, model outputs, normalized predictions, traces,
and reports are intentionally excluded from this public repository; the paths
record the layout used by the released scripts.

## Manuscript terminology and artifact mapping

The manuscript uses concise, human-readable terminology and does not expose
internal field names, prompt identifiers, run identifiers, trace tags, or
artifact paths. The following table provides the unique mapping needed to
connect the manuscript descriptions to the retained experimental records.

| Manuscript term | Machine identifier or record tag | Scope |
|---|---|---|
| candidate-region status | `candidate_verdict` | Candidate-conditioned model-output field |
| lighting shadow | `lighting_shadow` | Canonical artifact-label token |
| geometry structure | `geometry_structure` | Canonical artifact-label token |
| boundary seam | `boundary_seam` | Canonical artifact-label token |
| frozen candidate-conditioned prompt | `candidate_evidence_v1` | JSG-Core and JSG-Scale candidate-conditioned requests |
| marked-image-and-crop candidate-conditioned prompt | `candidate_evidence_v1_marked_crop` | JSG-Xfer candidate-conditioned requests |
| primary five-feature configuration | `scale-pixel5d-v2-primary`; proposal package `pixel5d-v2` | Primary JSG-Scale tables and controls |
| earlier six-feature prompt-sensitivity configuration | `prompt-legacy6d-v1` | Post-hoc JSG-Scale prompt analysis |
| baseline prompt | `candidate_v1` | Frozen baseline wording in the prompt-sensitivity analysis |
| target-definition prompt | `target_v2` | Target-definition variant in the prompt-sensitivity analysis |
| artifact-definition prompt | `artifact_v2` | Artifact-definition variant in the prompt-sensitivity analysis |
| unified-runtime transfer configuration | `xfer-pixel5d-v2-unified-runtime`; trace tag `pixel5d-v2/unified-runtime` | Complete 100-report JSG-Xfer evaluation |

The proposal-feature configuration mapping is exact and order-sensitive:

| Configuration | Ordered feature vector | Manuscript role |
|---|---|---|
| `scale-pixel5d-v2-primary` / `pixel5d-v2` | `rgb_std`, `gray_mean`, `gray_std`, `edge_mean`, `lap_var` | Primary five-feature JSG-Scale results |
| `prompt-legacy6d-v1` | `rgb_mean`, `rgb_std`, `gray_mean`, `gray_std`, `edge_mean`, `lap_var` | Earlier six-feature package used only for the post-hoc prompt-specification diagnostic |

## Common audit contract

All three families are evaluated under contract `jsg-bench-audit-v1.0`.

| Artifact | Repository path | SHA-256 |
|---|---|---|
| Frozen audit contract | `configs/jsg_audit_contract_v1.json` | `1436658743359b74f75beba5e19a0fffc2baa65a1a2ffc2587359fae3222f443` |

`experiments/export_jsg_audit_traces.py` is the canonical trace entry point;
`experiments/build_pixel5d_audit_traces.py` is a compatibility alias to that
same exporter. The exporter validates every contract-required trace field and
asserts the manuscript's frozen pass counts before writing the manifest.

| Canonical artifact | SHA-256 | Frozen check |
|---|---|---|
| `experiments/export_jsg_audit_traces.py` | `34aea80bc74491eea0409508ce94e2cc6b76cce38f9a448d500547e3e67e03d2` | Contract validation and count assertions |
| `experiments/build_pixel5d_audit_traces.py` | `1bd890152b7615acfe71bba6348602f7f877723bd3dbf00046c4541ce5f896cc` | Compatibility alias only |
| `experiments/runs/jsg_core_td_egfa_audit_trace_pixel5d_v2.jsonl` | `01510af46f16527ed3fd4ed08f04ab6e045a7ba3eb665b4c9a1894bbc2ad6b7b` | 100 reports; strict/shared/set passes 35/61/35 |
| `experiments/runs/jsg_scale_td_egfa_audit_trace_pixel5d_v2.jsonl` | `ec62d758bc9fc488d99831f28f58c27c77c84b031427dae2ea14fe7f0ed876a6` | 600 reports; strict/shared/set passes 156/196/156 |
| `experiments/runs/jsg_xfer_td_egfa_audit_trace_pixel5d_v2.jsonl` | `42f09e4587bf9c8e296ee5bad06eb4ccb05cb165c9c82cbb63a6d080820602aa` | 100 reports; strict/shared/set passes 21/40/21; P/S passes 30 |

## `recommended-controls-v1`

The same-runtime mixed-class routing, view-decoupling, shuffled-proposal, and
bbox-adjustment controls use Qwen2.5-VL-3B-Instruct with Transformers 4.57.6,
PyTorch 2.8.0, Apple MPS float16, scaled dot-product attention, greedy decoding,
`max_new_tokens=512`, and `max_pixels=602112`. The run seed is `20260729`;
although decoding is greedy, the seed is retained in each inference manifest.
The main 300-report view family and the 100-report proposal and bbox controls
were executed under one manifest. The disjoint 2,000-report candidate arm uses
four shard manifests with the same checkpoint and generation configuration.

Two fixed donor assignments support the controls. The crop-correspondence
assignment is a deterministic minimum-cost derangement based on standardized
log proposal area and aspect ratio within class-by-mask-area strata; self-pairs
are prohibited. The proposal-package assignment uses a seeded derangement
within each of the four mask-area strata and jointly replaces the candidate
bbox, marked image, crop, and low-level hints. The manuscript reports one
frozen assignment of each type rather than averaging over permutations.

| Manifest or assignment | SHA-256 | Check supplied |
|---|---|---|
| `experiments/reports/recommended_suite/inference_manifest_legacy_controlled.json` | `2814e97f70cf9f1e1f765e416304992b3143c8c39aadde20faf9638c4755320d` | Checkpoint shards, runtime, generation configuration, request hashes, and output hashes |
| `experiments/reports/recommended_suite/inference_manifest_negative2000_part0500.json` | `a729d8fc2a50620fcf0163a492bc51b34a8144c5851cc64cfd79c777e8ff35cb` | Candidate-arm shard 1 runtime and artifact hashes |
| `experiments/reports/recommended_suite/inference_manifest_negative2000_part1000.json` | `11f6511015a373f4e904bb8f1b8f05193ff82baa287bb88653a918ac81319d08` | Candidate-arm shard 2 runtime and artifact hashes |
| `experiments/reports/recommended_suite/inference_manifest_negative2000_part1500.json` | `6afb61bbf8398e06a36481735ada6040e9ac3562853f78c300e4bcc6e9b0b672` | Candidate-arm shard 3 runtime and artifact hashes |
| `experiments/reports/recommended_suite/inference_manifest_negative2000_part2000.json` | `21197a32d6e1acfc5559c7fd78758f3782f8036c9ead125587b90af07278dd0e` | Candidate-arm shard 4 runtime and artifact hashes |
| `experiments/runs/recommended_suite/sid_explain300_top3_shuffled_crop_permutation.jsonl` | `9ff3d512de6e058f713b20491c1283d2c92d73b2e774f06be50d9ac786949729` | Geometry-matched no-self-pair crop assignment |
| `experiments/runs/recommended_suite/jsg_core_shuffled_proposal_permutation.jsonl` | `cc6d023b65d2ac1a4c15c16b1f5e0fed2384946dc8f28eaad75e98e3a3613b37` | Seeded, within-stratum no-self-pair proposal-package assignment |
| `experiments/reports/recommended_suite/same_runtime_routing_audit.json` | `2208af748abf5ed56666621f34f554a64d9fd8347cec69f8b5039617b1454fcc` | Mixed-class routing counts and rates |
| `experiments/reports/recommended_suite/view_decoupling.json` | `c350fb99ba4a134aab555b7cbf172500f485b09fd5e49bf412dc6421f90971ef` | Same-ID view and crop-correspondence contrasts |
| `experiments/reports/recommended_suite/proposal_package_specificity.json` | `fdaf440290cae81b83e66ab9aa654ed6492177c7bd0b0fbab5b3ff4d40834806` | Shuffled-proposal passing-set and paired contrasts |

The released preparation, inference, and analysis entry points bind the public
implementation of this suite:

| Public script | SHA-256 |
|---|---|
| `experiments/prepare_recommended_suite.py` | `3f6c514bbb33b4911cdf5e014c733d0c2069d6fd65a5b85d70f76baa242d391a` |
| `experiments/run_qwen25vl_recommended_suite.py` | `2d11e7d40fcde149dd3eb8a6ec008895f254771f3e9968ddeb58bcc5250cbc97` |
| `experiments/analyze_recommended_suite.py` | `6133a1c35c4ea61eb5262bc0ac5309c42e640d06b4c5f405a26c9f4b3cdd7c38` |

## `scale-pixel5d-v2-primary`

This is the primary JSG-Scale run used in the main benchmark tables and the
spatial, paired-contrast, and criterion-relaxation analyses. It uses the
five-distinct-feature `pixel5d-v2` proposal package while holding the legacy
static prompt wording fixed. Its reported Artifact, Target, BBox-pass at 0.1,
and Joint at 0.1 rates are 80.67%, 56.83%, 52.83%, and 26.00%, respectively
(image-aligned-v2 spatial references; 26.17% under the superseded mask-grid references).

| Artifact role | Retained local path | SHA-256 |
|---|---|---|
| Requests | `experiments/runs/jsg_scale_requests_pixel5d_v2_legacy_prompt_controlled.jsonl` | `7878cbaa3fbe31edc657252068e412ed9e5944e0faf528013c674613994040fd` |
| Raw model outputs | `experiments/runs/jsg_scale_outputs_pixel5d_v2_legacy_prompt_controlled.jsonl` | `6c675c5f8b6189e75dce141ba0a748ea92cb508eacad1f8e08410b05213f7a50` |
| Normalized predictions | `experiments/runs/jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl` | `250d006163d1497eada9481604ae693de9418f16a72701b9e6515de74a6c68a7` |
| Per-report audit trace | `experiments/runs/jsg_scale_td_egfa_audit_trace_pixel5d_v2.jsonl` | `ec62d758bc9fc488d99831f28f58c27c77c84b031427dae2ea14fe7f0ed876a6` |

The following retained reports provide independent hash-bearing or numerical
checks for this family.

| Manifest or report | SHA-256 | Check supplied |
|---|---|---|
| `experiments/reports/spatial_construct_controls.json` | `44ce02e624790c53802a7d9419e5063d8b090d331e5c832cfc6bab55c8b4a748` | Request and prediction hashes; spatial-control inputs |
| `experiments/reports/criterion_relaxation_analysis.json` | `c8dd747bf878c79e18a1e8f2e873a994ded6876dd11854cd2c9f5049c27cd4cd` | Trace hash and contract-level condition counts |
| `experiments/reports/pixel5d_v2_rerun_comparison.json` | `5c351cf99d5fb955a4649292b40323d13e707f88108946090a2034109497c603` | Paired old-versus-`pixel5d-v2` comparison and primary rates |
| `experiments/reports/jsg_scale_metrics_pixel5d_v2_legacy_prompt_controlled.json` | `0a62e5d37ed0d84dfec5557a97caf21baf1568266c791948a876041d64ff7372` | Primary evaluation summary |

## `prompt-legacy6d-v1`

This is the earlier six-feature candidate-package family used only for the
post-hoc prompt-specification sensitivity analysis. The `candidate_v1` arm is
the frozen baseline wording; `target_v2` and `artifact_v2` alter the specified
field prompt while retaining that earlier proposal package. The manuscript's
`candidate_v1` row reports Artifact, Target, BBox-pass at 0.1, and Joint at 0.1
rates of 79.50%, 58.00%, 52.00%, and 25.33%, respectively.

### `candidate_v1` baseline artifacts

| Artifact role | Retained local path | SHA-256 |
|---|---|---|
| Requests | `experiments/runs/sid_fa600_candidate_top3_requests.jsonl` | `c8e846bf1db705e51b42cff7d23ecc49c5fd4743d048b05ee118c631aa065e67` |
| Raw model outputs | `experiments/runs/sid_fa600_candidate_top3_qwen25vl_outputs.jsonl` | `f78010fc5bb0a509b03a1f8c22a8801e694de24b8815c17ef1c3dce129fcf74a` |
| Normalized predictions | `experiments/runs/sid_fa600_candidate_top3_postreview_predictions.jsonl` | `ea9a04d537a32293b76e943923f8b03021a2c170caa8bb5ea29ad0278840a938` |
| Per-report audit trace | `experiments/runs/jsg_scale_td_egfa_audit_trace_v1.jsonl` | `b48028bf38cb1738104fa5f66b41ff7fac0dd7e421f3740861f93daf4592513f` |

The first row of the per-report audit trace points to the three exact
`candidate_v1` artifacts above, and the trace contains 600 reports.

### Prompt-sensitivity manifests

| Manifest or trace | SHA-256 | Scope |
|---|---|---|
| `experiments/reports/recommended_suite/prompt_specification_sensitivity.json` | `4c5d3907d83c9f6cd1db314cf387910967a3a8f6adf46cdcb0abbe5fbfa7b742` | Hashes and summaries for `candidate_v1`, `target_v2`, and `artifact_v2` |
| `experiments/runs/recommended_suite/prompt_specification_sensitivity_trace.jsonl` | `0ba851aab66cd0148c41dbbb2563d3c5507112861a2f396e77041054d8efbc5a` | Same-ID, per-condition sensitivity outcomes |

For completeness, the two modified-prompt output hashes are
`84ee035572cf6b0ee99f90a961d1fd79932bbd900af8f9e7c1f94aa075b3cca6`
for `target_v2` and
`66b17fe6c902fa144e17bb7ed70e6d7919fb23864aa64769c251b0b450765527`
for `artifact_v2`; their request and normalized-prediction hashes are retained
in the prompt-sensitivity JSON manifest.

## `xfer-pixel5d-v2-unified-runtime`

This is the complete 100-report JSG-Xfer family used in the manuscript's main
and source-disaggregated tables. Its Artifact, Target, BBox-pass at 0.1, and
strict Joint at 0.1 rates are 47.00%, 54.00%, 68.00%, and 21.00%, respectively.
The trace labels all 100 reports
`known-tampered/proposal-grounded/B_top3/pixel5d-v2/unified-runtime`.

The run-level execution record documents a local Qwen2.5-VL-3B-Instruct run on
Apple MPS in float16 with greedy decoding, `max_new_tokens=512`, and
`max_pixels=602112`. All 100 Xfer requests were run in that controlled batch
without an inference error. The retained raw-output rows, however, use a
legacy schema that does not embed backend or software-version fields. The
family is therefore a single controlled output/trace family with incomplete
row-level runtime metadata; it is not the older 95-retained-plus-5-rerun mixed
artifact.

| Artifact role | Retained local path | SHA-256 |
|---|---|---|
| Low-level proposals | `experiments/runs/jsg_xfer_lowlevel_pixel5d_v2.jsonl` | `ef6a9a075885757a4466d05f0d01b7f94dcea43668fbb9524a74d945414eb1e3` |
| Requests | `experiments/runs/jsg_xfer_requests_pixel5d_v2.jsonl` | `5277f9d71bff2cea2912943e370ec3e867b9905742884a4a1939086ff7e9e8a6` |
| Raw model outputs | `experiments/runs/jsg_xfer_outputs_pixel5d_v2.jsonl` | `4718f65a2a460b5400e4a0186b9f1749ceef05be6cbe598500502220ebf30dfa` |
| Normalized predictions | `experiments/runs/jsg_xfer_predictions_pixel5d_v2.jsonl` | `f5d7d5724887c55a1f15248db28e0e9b1ed6d9a7df3eee0ad406bca3230cb515` |
| Per-report audit trace | `experiments/runs/jsg_xfer_td_egfa_audit_trace_pixel5d_v2.jsonl` | `42f09e4587bf9c8e296ee5bad06eb4ccb05cb165c9c82cbb63a6d080820602aa` |

| Manifest or report | SHA-256 | Check supplied |
|---|---|---|
| `experiments/reports/jsg_xfer_requests_pixel5d_v2_summary.json` | `26b6eec29feb017e616b5bbfb2dc4de863709e88b8f3abc4a941d5e9226a5fdc` | Request count, prompt family, and zero skipped reports |
| `experiments/reports/jsg_xfer_candidate_metrics_pixel5d_v2.json` | `bb306c652fc05dec520ab50e858d90ff3ebac258a4788a0a2de6a7c76cd8d3d4` | Parse, prediction, and spatial summaries |
| `experiments/reports/jsg_xfer_semantic_metrics_pixel5d_v2.json` | `451e15683a0456a03d05051fbb4cd660dea32c051c3f226928347cbf8794e57a` | Overall and source-disaggregated semantic/spatial rates |
| `experiments/reports/pixel5d_v2_rerun_comparison_and_manuscript_recommendations.md` | `4e30a4c0068349f689dcc27c1129ea08640162b4c63d3e554e0187c492c3063c` | Run-level model, device, dtype, decoding, and full-batch execution record |

## Interpretation rule

Use `scale-pixel5d-v2-primary` for the principal JSG-Scale estimates. Use
`prompt-legacy6d-v1` only when discussing sensitivity to the three earlier
prompt packages. Matching sample IDs enable paired checks within a declared
family; they do not make the two families the same run. Use
`xfer-pixel5d-v2-unified-runtime` for every 47.00/54.00/68.00/21.00 JSG-Xfer
result; do not describe those values as the older mixed-provenance Xfer run.
