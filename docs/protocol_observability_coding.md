# Table I Protocol-Observability Coding Audit

Audit date: 2026-07-19

This sheet supports Table I and the detector-protocol discussion in
`ieee-journal-latex-package/main_en.tex`. Table I is restricted to the nearest
structured-output comparators; the detector-only cells remain below as the
evidence base for the preceding prose. The sheet codes documented protocol
observability, not system quality or every capability an implementation might
have. The unit of coding is one resource-by-criterion cell.

## Decision Rules

- `Yes`: the inspected protocol explicitly represents the criterion and
  evaluates it against a reference or declared acceptance rule.
- `Partial`: a related output, annotation, proxy, or qualitative analysis is
  present, but at least one required correctness check is absent.
- `ND`: no explicit coverage was found in the inspected public sources. ND
  means "no documented coverage" and does not mean that the method cannot
  provide the capability.
- `OOS`: the criterion is explicitly outside the declared protocol scope and
  is therefore not presented as an evaluated capability.

The four criteria are fixed as follows.

1. `Free-text rationale evaluated`: a human-readable forensic rationale is
   scored or otherwise checked for correctness. Merely emitting or retaining
   text is Partial.
2. `Evidence region evaluated`: a predicted evidence region is quantitatively
   compared with a spatial reference. Qualitative saliency visualization is
   Partial.
3. `Structured artifact--target attribution evaluated`: explicit coverage
   requires separate normalization and reference-based evaluation of both the
   forensic artifact cue and the affected semantic target. Spatial grounding
   is covered by the Evidence region criterion; free text, a manipulation
   class alone, localization alone, or a combined description score is
   Partial.
4. `Executable same-sample joint pass and trace`: the public protocol executes
   a same-sample conjunction over applicable fields and retains sample
   identity, condition outcomes or failed fields, and evaluated-region
   provenance.

## Sources Inspected

| Resource | Archival source | Additional public artifact | Version inspected |
|---|---|---|---|
| CLIP-ADA | [IEEE DOI](https://doi.org/10.1109/TIFS.2026.3676657), TIFS 21:3588-3601 | None required for the reported metric scope | Paper accessed 2026-07-18 |
| Component-specific prompt tuning | [IEEE DOI](https://doi.org/10.1109/TIFS.2026.3678021), TIFS 21:3424-3435 | None required for the reported metric scope | Paper accessed 2026-07-18 |
| Degradation inconsistency | [IEEE DOI](https://doi.org/10.1109/TIFS.2026.3695433), TIFS 21:5627-5642 | Supplementary material linked by IEEE | Paper accessed 2026-07-18 |
| Dual-frequency branches | [IEEE DOI](https://doi.org/10.1109/TIFS.2026.3664000), TIFS 21:2313-2325 | [Official code](https://github.com/HorizonTEL/DFFreq-main) | Paper and repository accessed 2026-07-18 |
| Propose-and-Rectify | [IEEE DOI](https://doi.org/10.1109/TIFS.2026.3677584), TIFS 21:3396-3409 | [Official code](https://github.com/JacksonZKY/Propose-Rectify) | Commit `7536e2d8f02352f588baa9d08ae8215a2d57e611` |
| SIDA | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_SIDA_Social_Media_Image_Deepfake_Detection_Localization_and_Explanation_with_CVPR_2025_paper.html) | [Official code](https://github.com/hzlsaber/SIDA) | Commit `6b1c3aa9097a79849ea7aae95601854763014a9d` |
| FakeShield | [ICLR paper](https://openreview.net/forum?id=pAQzEY7M03) | [Official code](https://github.com/zhipeixu/FakeShield) | Commit `b22717d7d524cbf617b8841d488d1edf56ceb7f3` |
| DGM4 | [IEEE DOI](https://doi.org/10.1109/TPAMI.2024.3367749), TPAMI 46(8):5556-5574 | [Official code](https://github.com/rshaojimmy/MultiModal-DeepFake) | Commit `33fc04f3a1f550f923848737c864eea32556f699` |
| Omni-Fake | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Omni-Fake_Benchmarking_Unified_Multimodal_Social_Media_Deepfake_Detection_CVPR_2026_paper.html) | [Official supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Li_Omni-Fake_Benchmarking_Unified_CVPR_2026_supplemental.pdf) | Paper and supplement accessed 2026-07-18 |
| JSG-Bench | Manuscript contract definition | `configs/jsg_audit_contract_v1.json`, `experiments/export_jsg_audit_traces.py`, and `docs/run_provenance.md` | Contract `jsg-bench-audit-v1.0`; submission release verified 2026-08-01 |

## Cell-Level Evidence

| Resource | Criterion | Code | Evidence location | Rationale |
|---|---|---:|---|---|
| CLIP-ADA | Free-text rationale evaluated | ND | Sec. IV-B, evaluation metrics | The reported endpoints are accuracy and average precision; no rationale metric is specified. |
| CLIP-ADA | Evidence region evaluated | ND | Sec. IV-B, evaluation metrics | The protocol evaluates image-level synthetic-image detection, not an output region. |
| CLIP-ADA | Structured artifact--target attribution evaluated | ND | Sec. IV-B, evaluation metrics | Artifact representations are learned features, not separately checked report fields. |
| CLIP-ADA | Executable same-sample joint pass and trace | ND | Sec. IV-B and result tables | Aggregate ACC/AP do not define a report-level conjunction or failure trace. |
| Component-specific prompt tuning | Free-text rationale evaluated | ND | Sec. IV-B1, p. 3430 | Component prompts guide feature extraction, while ACC/AUC score the final real/fake prediction. |
| Component-specific prompt tuning | Evidence region evaluated | ND | Secs. III-IV | Eyes, nose, and mouth prompts are inputs; no predicted region is evaluated against spatial ground truth. |
| Component-specific prompt tuning | Structured artifact--target attribution evaluated | ND | Sec. IV-B1 | The protocol does not separately score an artifact type and affected target. |
| Component-specific prompt tuning | Executable same-sample joint pass and trace | ND | Sec. IV and result tables | ACC/AUC are aggregate detector metrics without a per-case field conjunction. |
| Degradation inconsistency | Free-text rationale evaluated | ND | Sec. IV and result tables | The reported endpoint is detector AUC; no textual-rationale check is specified. |
| Degradation inconsistency | Evidence region evaluated | Partial | Grad-CAM analysis, p. 5639 | Qualitative activation maps are compared visually with ground truth, but no localization metric or acceptance rule is reported. |
| Degradation inconsistency | Structured artifact--target attribution evaluated | ND | Sec. IV | Degradation inconsistency is a learned cue, not a separately scored artifact-target report. |
| Degradation inconsistency | Executable same-sample joint pass and trace | ND | Sec. IV and result tables | Cross-dataset and cross-manipulation AUC do not retain a multi-field pass set. |
| Dual-frequency branches | Free-text rationale evaluated | ND | Experimental setup, physical PDF p. 6 | The stated evaluation metrics are ACC and AP. |
| Dual-frequency branches | Evidence region evaluated | ND | Experimental setup and official evaluator | Local/frequency features support classification; no output evidence region is scored. |
| Dual-frequency branches | Structured artifact--target attribution evaluated | ND | Experimental setup and result tables | No separate artifact and affected-target fields are evaluated. |
| Dual-frequency branches | Executable same-sample joint pass and trace | ND | Result tables and official evaluator | The released endpoint aggregates ACC/AP and does not emit a condition-level audit trace. |
| Propose-and-Rectify | Free-text rationale evaluated | Partial | Abstract and proposal stage | The MLLM produces initial manipulation analysis, but the published evaluation does not separately score that analysis. |
| Propose-and-Rectify | Evidence region evaluated | Yes | Sec. IV-B and localization tables | Predicted masks are evaluated with pixel-level IoU/F1. |
| Propose-and-Rectify | Structured artifact--target attribution evaluated | Partial | Proposal/rectification description and Sec. IV-B | Semantic analysis and forensic cues are present, but the published evaluator does not separately score artifact type and affected target scope as report fields. |
| Propose-and-Rectify | Executable same-sample joint pass and trace | ND | Sec. IV-B; official `train.py` test path | Image F1/AUC and pixel IoU/F1 are reported separately; no same-sample failed-field/provenance trace was found. |
| SIDA | Free-text rationale evaluated | Partial | Paper abstract; Sec. 5 evaluation metrics | SIDA emits textual explanations, but the published metrics cover detection ACC/F1 and localization AUC/F1/IoU. |
| SIDA | Evidence region evaluated | Yes | Sec. 5; official `train_SIDA.py` evaluator | The tamper mask is quantitatively evaluated with spatial metrics. |
| SIDA | Structured artifact--target attribution evaluated | Partial | Native description format; official README | Area, content, and inconsistency text can be emitted, but the published evaluator does not separately score a normalized artifact cue and semantic target scope. |
| SIDA | Executable same-sample joint pass and trace | ND | Sec. 5; official `train_SIDA.py` evaluator | The evaluator reports classification and mask metrics and derives a combined F1 from aggregate accuracy and IoU; neither defines a per-case conjunction or a failed-condition/provenance trace. |
| FakeShield | Free-text rationale evaluated | Yes | Sec. 4.3, Table 3 | Explanation text is scored by cosine semantic similarity against GPT-4o-generated descriptions. |
| FakeShield | Evidence region evaluated | Yes | Sec. 4.4, Table 4 | Tamper localization is evaluated with IoU and F1. |
| FakeShield | Structured artifact--target attribution evaluated | Partial | Secs. 3.1 and 4.3 | Descriptions include tamper content, location, and visible artifacts, but the full-text similarity score does not separately verify artifact and target fields. |
| FakeShield | Executable same-sample joint pass and trace | ND | Secs. 4.3-4.4 and official repository | Explanation, detection, and localization are reported as separate tasks; no complete-report pass set and failure trace was found. |
| DGM4 | Free-text rationale evaluated | ND | Sec. 5.1 and official evaluator | The protocol scores detection and grounding outputs but does not generate or score a human-readable forensic rationale. |
| DGM4 | Evidence region evaluated | Yes | Sec. 5.1 and official evaluator | Image-region boxes are evaluated with overlap-based metrics, and manipulated text tokens are evaluated with precision, recall, and F1. |
| DGM4 | Structured artifact--target attribution evaluated | Partial | Task definition and Sec. 5.1 | Manipulation operations and grounded content are represented, but artifact cues and affected semantic targets are not separately matched as report fields. |
| DGM4 | Executable same-sample joint pass and trace | ND | Sec. 5.1 and official evaluator | Classification and grounding metrics are aggregated by task; no complete-report conjunction with failed-field and region-provenance traces was found. |
| Omni-Fake | Free-text rationale evaluated | Yes | Secs. 4.3 and 5.1; supplementary Sec. S1.1 | Rationales are evaluated with ROUGE-L, cosine semantic similarity, and expert judgments of factual correctness and usefulness. |
| Omni-Fake | Evidence region evaluated | Yes | Secs. 4.3 and 5.1 | Predicted visual boxes/masks and temporal intervals are quantitatively compared with spatial or temporal references using IoU and localization F1. |
| Omni-Fake | Structured artifact--target attribution evaluated | Partial | Structured-output definition and explanation evaluation | Structured outputs include task labels and localization, but the published evaluator does not independently match an artifact cue and semantic target scope as two report fields. |
| Omni-Fake | Executable same-sample joint pass and trace | Partial | Training reward and evaluation protocol | A deterministic parser feeds an additive per-response reward over format, detection, spatial, and temporal terms. This weighted multi-task proxy is not a conjunction, and the paper and supplement do not expose a binary passing set with sample identities, failed conditions, and region provenance; no official evaluator code was public at inspection. |
| JSG-Bench | Free-text rationale evaluated | OOS | Manuscript protocol scope and limitations | The report schema may retain evidence text, but audit-v1.0 does not score free-text faithfulness; the manuscript therefore marks this criterion outside scope. |
| JSG-Bench | Evidence region evaluated | Yes | Manuscript Eq. (1), the IoU definition and descriptive J-AUC readout in Sec. IV-C; contract spatial rule | The declared report region is checked against the reference region at fixed thresholds and its provenance is retained. |
| JSG-Bench | Structured artifact--target attribution evaluated | Yes | Manuscript Secs. III-A-III-C | Artifact cue and target scope have separate applicability and matching conditions. |
| JSG-Bench | Executable same-sample joint pass and trace | Yes | `configs/jsg_audit_contract_v1.json`, `experiments/export_jsg_audit_traces.py`, and `docs/run_provenance.md` | The evaluator executes the same-sample conjunction and records identifiers, condition vectors, failed fields, passing sets, versions, and region provenance. Full generated traces are not distributed in the public repository. |

## Interpretation Boundary

This audit does not compare accuracy or claim that an ND-coded method lacks a
capability. A future public artifact, evaluator, or protocol revision can change
a cell. Any update should record the new source, version, access date, and
cell-specific rationale before Table I is changed. Omni-Fake's final-column
Partial specifically denotes a per-response weighted multi-task proxy; it does
not satisfy the binary pass-set and retained-trace requirements for explicit
coverage. JSG-Bench is OOS for free-text rationale because audit-v1.0 does not
score free-text faithfulness; retaining text in the report schema does not
change that declared boundary.
