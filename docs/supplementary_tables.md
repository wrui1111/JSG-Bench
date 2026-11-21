# Supplementary Material S1 — Detailed audit results

Report counts use the full declared cohort. Percentages are rounded to two decimal places. Brackets give 95% intervals unless otherwise stated.

## Table S1. Frozen-report threshold profiles

BBox-pass includes the structural parse gate. Joint is strict-primary. BBox and Joint are percentages, and J-AUC is a proportion integrated over the IoU threshold.

| Cohort | N | BBox@0.3 | Joint@0.3 [CI] | BBox@0.5 | Joint@0.5 [CI] | J-AUC [CI] |
|---|---:|---:|---|---:|---|---|
| Scale | 600 | 24.00 | 13.17 [10.91, 16.14] | 6.17 | 3.67 [2.58, 5.78] | 0.0888 [0.0766, 0.1009] |
| Core | 100 | 30.00 | 14.00 [9.38, 22.77] | 7.00 | 0.00 [0.00, 6.66] | 0.0986 [0.0750, 0.1228] |
| Xfer | 100 | 34.00 | 11.00 [6.88, 19.53] | 6.00 | 1.00 [0.18, 8.85] | 0.0695 [0.0452, 0.0960] |

## Table S2. Semantic field diagnostics

Accuracy, frequency of the majority reference label, and macro recall are percentages. Macro recall averages recall over labels present in the cohort. Cohen’s κ compares predictions with the fixed reference.

| Cohort | Field | Accuracy | Majority | Macro recall | κ |
|---|---|---:|---:|---:|---:|
| Scale | Artifact | 80.67 | 84.33 | 20.14 | 0.071 |
| Scale | Target | 56.83 | 53.33 | 39.93 | 0.233 |
| Core | Artifact | 71.00 | 68.00 | 54.71 | 0.220 |
| Core | Target | 94.00 | 98.00 | 96.94 | 0.385 |
| Xfer | Artifact | 47.00 | 30.00 | 37.36 | 0.299 |
| Xfer | Target | 54.00 | 45.00 | 53.40 | 0.305 |

## Table S3. Core adjustment-prompt control

The adjustment prompt yielded 79 parseable tampered predictions and 21 unparseable reports on 100 images. Both box scores use this run’s shared gate and the full denominator. Differences subtract the supplied candidate score from the model-emitted adjustment score. Holm correction covers the three thresholds.

| Metric (%) | Supplied candidate | Emitted adjustment | Difference [CI], pp | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| Mean IoU × 100 | 17.01 | 15.76 | −1.24 [−3.12, 0.54] | — | — |
| IoU-pass@0.1 | 53.00 | 49.00 | −4.00 [−8.00, −1.00] | 0.125 | 0.375 |
| IoU-pass@0.3 | 26.00 | 22.00 | −4.00 [−9.00, 0.00] | 0.219 | 0.438 |
| IoU-pass@0.5 | 6.00 | 7.00 | +1.00 [−3.00, 6.00] | 1.000 | 1.000 |

## Table S4. Three-class routing on SID-Explain-300

The same-runtime candidate-conditioned branch evaluated 100 images per reference class. All entries are percentages.

| Metric | Estimate | 95% CI |
|---|---:|---|
| Parse rate | 99.00 | [96.45, 99.56] |
| Accuracy / balanced accuracy | 33.00 | [31.52, 34.77] |
| Macro F1 | 16.62 | [16.33, 16.84] |
| Non-tampered specificity | 0.00 | [0.00, 2.62] |
| Equal-weight negative false-tampered emission | 99.00 | [95.89, 99.72] |
| Real recall | 0.00 | [0.00, 3.70] |
| Full synthetic recall | 0.00 | [0.00, 3.70] |
| Tampered recall | 99.00 | [94.55, 99.82] |

## Table S5. Negative-class recognition on the same 2,000 images

The real and fully synthetic classes have equal weight. The two branches use different prompts, views, and output vocabularies. Entries are percentages.

| Metric | Candidate-conditioned three-class [CI] | Dedicated binary [CI] |
|---|---|---|
| Parse rate | 99.55 [99.11, 99.76] | 99.90 [99.57, 99.96] |
| Fixed-mixture accuracy | 0.00 [0.00, 0.27] | 89.00 [87.53, 90.26] |
| Real recall | 0.00 [0.00, 0.38] | 85.90 [83.61, 87.92] |
| Full synthetic recall | 0.00 [0.00, 0.38] | 92.10 [90.26, 93.62] |
| False-tampered emission | 99.55 [99.11, 99.76] | N/A |

## Table S6. Free evidence cards on SID-Explain-300

Artifact and target use all 300 images per model, including 200 negative-class images. All rates are percentages. Replaying the archived raw responses with explicit field-validity flags changed zero of 1,800 artifact and target decisions across the three models.

| Model | Parse | Prediction accuracy | Predicted real | Artifact | Target | Artifact correct | Target correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 100.00 | 64.33 | 49.67 | 45.33 | 33.67 | 136 | 101 |
| Qwen2.5-VL-7B | 100.00 | 38.00 | 95.33 | 34.33 | 35.67 | 103 | 107 |
| MiniCPM-V 4.5 | 100.00 | 36.00 | 97.33 | 34.00 | 35.00 | 102 | 105 |

## Table S7. Proposal-conditioned reports on SID-FA-200

All 200 images per model are known tampered. All rates are percentages. Inherited counts refer to valid evaluated proposal boxes. The other reports had no valid evaluated box.

| Model | Parse | Tampered recall | Artifact | Target | BBox-pass@0.1 | Strict Joint@0.1 | Inherited boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 99.00 | 99.00 | 80.50 | 57.50 | 52.50 | 26.00 | 198 |
| Qwen2.5-VL-7B | 100.00 | 4.50 | 0.00 | 2.50 | 1.50 | 0.00 | 9 |
| MiniCPM-V 4.5 | 100.00 | 0.00 | 0.00 | 1.50 | 0.00 | 0.00 | 0 |

## Table S8. Proposal-package reassignment on JSG-Scale

The five allocations share 600 recipients and the matched baseline. Each row reports observed image pairs for one fixed allocation. BBox, semantic and Joint are percentages. Both, exits, entries, changed, and spatial-only are image counts. Spatial-only counts changed images whose unsuccessful run failed only the spatial check. The exploratory two-sided exact McNemar tests condition on each allocation. Holm adjustment covers the five donor comparisons. The unchanged repeat supplies the baseline reproducibility check.

| Run | BBox | Semantic | Joint | Both | Exits | Entries | Changed | Spatial-only | Jaccard | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unchanged repeat | 52.83 | 45.50 | 26.00 | 156 | 0 | 0 | 0 | 0 | 1.000 | — | — |
| Donor 20260729 | 46.50 | 44.67 | 23.33 | 94 | 62 | 46 | 108 | 67 | 0.465 | 0.14858 | 0.22184 |
| Donor 20260730 | 45.67 | 45.00 | 21.67 | 91 | 65 | 39 | 104 | 68 | 0.467 | 0.01384 | 0.06918 |
| Donor 20260731 | 46.17 | 44.83 | 21.83 | 92 | 64 | 39 | 103 | 64 | 0.472 | 0.01761 | 0.07044 |
| Donor 20260732 | 47.50 | 45.83 | 23.17 | 97 | 59 | 42 | 101 | 58 | 0.490 | 0.11092 | 0.22184 |
| Donor 20260733 | 44.83 | 46.50 | 22.33 | 94 | 62 | 40 | 102 | 57 | 0.480 | 0.03707 | 0.11120 |

Across allocations, 213 images changed status at least once relative to the matched run. Thirty-seven passed under every reassignment. Of these, 33 also passed with the matched input.

## Table S9. Prompt differences in the six-feature Scale control

The entries summarize the instructions sent with the same 600 marked images and crops. Strict passes use the original artifact and target mapping and IoU 0.1. The example fields below are class, target, and artifact, in that order.

| Feature | Baseline | Target-focused prompt | Artifact-focused prompt |
|---|---|---|---|
| Example answer | real / none / none | tampered / person / geometry_structure | tampered / object / boundary_seam |
| Label guidance | Enumerates candidate signs of local editing | Defines target categories and the treatment of human body parts | Lists seven artifact checks in order and gives a specific rule for geometry_structure |
| Decision instruction | Route globally generated images to full_synthetic | Decide target first, then artifact | Choose the most specific supported artifact, then assess weak or normal evidence |
| Other text | Gives local-evidence examples and explicit full_synthetic routing | Shortens the local-evidence instruction, replaces the explicit routing instruction, and adds the marked_crop mode line | Makes the same local-evidence and mode changes, replaces the explicit routing instruction, and rewords the weak-evidence rule |
| Strict passes | 152/600 | 208/600 | 15/600 |
