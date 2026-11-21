#!/bin/zsh
# Regenerate every manuscript artifact after the reference-geometry correction
# (image-aligned-v2). Run from the repository root. Frozen inputs are archived in
# experiments/runs/frozen_reference_v1/. No VLM inference is performed here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
R=experiments/runs
S=$R/recommended_suite
RS=experiments/reports/recommended_suite

step() { echo "\n=== $1"; }

step "1. Patch annotations and evaluate-format predictions (idempotent)"
$PY experiments/apply_reference_geometry_correction.py > /dev/null

step "2. Main-cohort metrics reports"
for c in core scale; do
  if [[ $c == core ]]; then A=$R/sid_explain300_tampered100_annotations_revised.jsonl; else A=$R/sid_fa600_annotations_postreview_eval.jsonl; fi
  $PY experiments/candidate_evidence_explanation.py evaluate --annotations $A \
    --outputs $R/jsg_${c}_outputs_pixel5d_v2_legacy_prompt_controlled.jsonl \
    --requests $R/jsg_${c}_requests_pixel5d_v2_legacy_prompt_controlled.jsonl \
    --output experiments/reports/jsg_${c}_metrics_pixel5d_v2_legacy_prompt_controlled.json \
    --summary-output experiments/reports/jsg_${c}_summary_pixel5d_v2_legacy_prompt_controlled.md > /dev/null
done

step "3. SIDA native adapter audit"
$PY experiments/evaluate_sida_jsg_core.py > /dev/null

step "4. Per-report audit traces"
$PY experiments/export_jsg_audit_traces.py > /dev/null

step "5. Recommended-suite variant analyses"
$PY experiments/analyze_recommended_suite.py variants --design sid_explain_300_views \
  --annotations SID_Set/annotations/SID-Explain-300.jsonl \
  --variant "matched_marked_crop|$S/sid_explain300_top3_marked_crop_legacy_predictions.jsonl|$S/sid_explain300_top3_marked_crop_legacy_requests.jsonl|$S/sid_explain300_top3_marked_crop_legacy_outputs.jsonl" \
  --variant "marked_only|$S/sid_explain300_top3_marked_only_legacy_predictions.jsonl|$S/sid_explain300_top3_marked_only_legacy_requests.jsonl|$S/sid_explain300_top3_marked_only_legacy_outputs.jsonl" \
  --variant "shuffled_crop|$S/sid_explain300_top3_shuffled_crop_legacy_predictions.jsonl|$S/sid_explain300_top3_shuffled_crop_legacy_requests.jsonl|$S/sid_explain300_top3_shuffled_crop_legacy_outputs.jsonl" \
  --variant "crop_only|$S/sid_explain300_top3_crop_only_legacy_predictions.jsonl|$S/sid_explain300_top3_crop_only_legacy_requests.jsonl|$S/sid_explain300_top3_crop_only_legacy_outputs.jsonl" \
  --baseline matched_marked_crop --primary-comparison marked_only --primary-comparison shuffled_crop \
  --output $RS/view_decoupling.json --summary-output $RS/view_decoupling.md \
  --trace-output $S/view_decoupling_trace.jsonl > /dev/null
$PY experiments/analyze_recommended_suite.py variants --design jsg_core_proposal \
  --annotations $R/sid_explain300_tampered100_annotations_revised.jsonl \
  --variant "matched_proposal_package|$R/jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl|$R/jsg_core_requests_pixel5d_v2_legacy_prompt_controlled.jsonl|$R/jsg_core_outputs_pixel5d_v2_legacy_prompt_controlled.jsonl" \
  --variant "shuffled_proposal_package|$S/jsg_core_shuffled_proposal_legacy_predictions.jsonl|$S/jsg_core_shuffled_proposal_legacy_requests.jsonl|$S/jsg_core_shuffled_proposal_legacy_outputs.jsonl" \
  --baseline matched_proposal_package --primary-comparison shuffled_proposal_package \
  --output $RS/proposal_package_specificity.json --summary-output $RS/proposal_package_specificity.md \
  --trace-output $S/proposal_package_specificity_trace.jsonl > /dev/null
$PY experiments/analyze_recommended_suite.py variants --design sid_fa600_prompt_sensitivity \
  --annotations $R/sid_fa600_annotations_postreview_eval.jsonl \
  --variant "candidate_v1|$R/sid_fa600_candidate_top3_postreview_predictions.jsonl|$R/sid_fa600_candidate_top3_requests.jsonl|$R/sid_fa600_candidate_top3_qwen25vl_outputs.jsonl" \
  --variant "target_v2|$R/sid_fa600_target_v2_top3_postreview_predictions.jsonl|$R/sid_fa600_target_v2_top3_requests.jsonl|$R/sid_fa600_target_v2_top3_qwen25vl_outputs.jsonl" \
  --variant "artifact_v2|$R/sid_fa600_artifact_v2_top3_postreview_predictions.jsonl|$R/sid_fa600_artifact_v2_top3_requests.jsonl|$R/sid_fa600_artifact_v2_top3_qwen25vl_outputs.jsonl" \
  --baseline candidate_v1 \
  --output $RS/prompt_specification_sensitivity.json --summary-output $RS/prompt_specification_sensitivity.md \
  --trace-output $S/prompt_specification_sensitivity_trace.jsonl > /dev/null
$PY experiments/analyze_recommended_suite.py variants --design sid_fa200_model_runtime \
  --annotations $R/sid_fa200_annotations.jsonl \
  --variant "qwen25vl_3b_transformers|$R/sid_fa200_candidate_top3_qwen25vl_predictions.jsonl|$R/sid_fa200_candidate_top3_requests.jsonl|$R/sid_fa600_candidate_top3_qwen25vl_outputs.jsonl" \
  --variant "qwen25vl_7b_ollama|$R/sid_fa200_candidate_top3_ollama7b_predictions.jsonl|$R/sid_fa200_candidate_top3_requests.jsonl|$R/sid_fa200_candidate_top3_ollama7b_outputs.jsonl" \
  --variant "minicpm_v45_ollama|$R/sid_fa200_candidate_top3_minicpm45_predictions.jsonl|$R/sid_fa200_candidate_top3_requests.jsonl|$R/sid_fa200_candidate_top3_minicpm45_outputs.jsonl" \
  --baseline qwen25vl_3b_transformers \
  --output $RS/multivlm_joint_diagnostic.json --summary-output $RS/multivlm_joint_diagnostic.md \
  --trace-output $S/multivlm_joint_diagnostic_trace.jsonl > /dev/null
$PY experiments/analyze_recommended_suite.py bbox-refine \
  --annotations $S/sid_explain300_tampered100_annotations.jsonl \
  --requests $S/jsg_core_bbox_refine_requests.jsonl --outputs $S/jsg_core_bbox_refine_outputs.jsonl \
  --output $RS/bbox_refinement_diagnostic.json --trace-output $S/bbox_refinement_trace.jsonl > /dev/null

step "6. Criterion relaxation, spatial controls, paired contrasts, thresholds"
$PY experiments/criterion_relaxation_analysis.py > /dev/null
# matplotlib lives in the system interpreter; the Qwen venv lacks it.
/usr/bin/python3 experiments/spatial_construct_controls.py --figure-stem /tmp/jsg_fig5 > /dev/null
cp /tmp/jsg_fig5.pdf ieee-journal-latex-package/figures/fig5.pdf && rm -f /tmp/jsg_fig5.{pdf,png,svg,tiff}
$PY experiments/paired_benchmark_contrasts.py > /dev/null
$PY experiments/joint_threshold_multivlm_analysis.py > /dev/null
$PY experiments/table_consolidation_statistics.py > /dev/null

step "7. Gate-only release records"
$PY JSG-Bench/experiments/export_gate_verification.py --source-root . --output-dir JSG-Bench/verification > /dev/null
(cd JSG-Bench && $PY experiments/verify_gate_release.py)

step "8. Figures driven by gate records"
/usr/bin/python3 scripts/redraw_fig3_failure_signatures.py JSG-Bench/verification/gate_records.jsonl ieee-journal-latex-package/figures/fig3.pdf > /dev/null

echo "\nreference-geometry regeneration complete"
