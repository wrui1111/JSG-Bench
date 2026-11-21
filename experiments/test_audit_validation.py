"""Small in-memory conformance fixtures; no images, gold corpus, or model needed."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import candidate_evidence_explanation as candidate
import evidence_explanation_experiment as free
import evaluate_sida_jsg_core as sida
from audit_validation import check_output_ids, image_dimensions, index_rows, valid_bbox
from export_jsg_audit_traces import EXPECTED_JOINT_PASSES, profile, td_trace_rows


class AuditValidationTests(unittest.TestCase):
    def test_bbox_validation(self):
        self.assertEqual(valid_bbox([0, 0, 100, 80], 100, 80), [0, 0, 100, 80])
        for box in ([0, 0, float("inf"), 20], [0, 0, float("nan"), 20],
                    [-1, 0, 10, 10], [0, 0, 101, 20], [0, 0, 20, 81],
                    [2, 0, 2, 3], [0, 0, 2], [False, 0, 2, 3]):
            with self.subTest(box=box):
                for module in (candidate, free):
                    self.assertIsNone(module.valid_bbox(box, 100, 80))

    def test_fields_and_failed_inference(self):
        for module in (candidate, free):
            def parse(row):
                return module.prediction_item(row, None, 100, 80) if module is candidate else module.prediction_item(row, 100, 80)
            missing = parse({"vlm_result": {"prediction": "real"}})
            self.assertTrue(missing["parsed"])
            self.assertFalse(missing["artifact_present"])
            self.assertFalse(missing["target_present"])
            explicit = parse({"vlm_result": {"prediction": "real", "artifact_type": "none", "target_scope": "none", "evidence_bbox": None}})
            self.assertTrue(explicit["artifact_present"])
            self.assertTrue(explicit["target_present"])
            self.assertTrue(explicit["bbox_explicit_null"])
            self.assertFalse(parse({"error": "inference failed", "vlm_result": {"prediction": "real"}})["parsed"])
            invalid = parse({"vlm_result": {"prediction": "unknown", "artifact_type": "unsupported", "target_scope": ["object"]}})
            self.assertFalse(invalid["prediction_present"])
            self.assertFalse(invalid["artifact_present"])
            self.assertFalse(invalid["target_present"])

    def test_candidate_inheritance_uses_valid_dimensions(self):
        row = {"vlm_result": {"prediction": "tampered", "evidence_bbox": [2, 2, 8, 8]}}
        request = {"metadata_for_evaluation_only": {"candidate_bbox": [0, 0, 101, 80], "width": 100, "height": 80}}
        self.assertEqual(candidate.prediction_item(row, request)["evidence_bbox"], [2, 2, 8, 8])
        self.assertEqual(candidate.prediction_item(row, request)["region_source"], "model_emitted")
        request["metadata_for_evaluation_only"]["candidate_bbox"] = [0, 0, 100, 80]
        self.assertEqual(candidate.prediction_item(row, request)["evidence_bbox"], [0, 0, 100, 80])
        self.assertEqual(candidate.prediction_item(row, request)["region_source"], "inherited_proposal")
        self.assertIsNone(candidate.prediction_item({"error": "failed", **row}, request)["evidence_bbox"])
        self.assertEqual(candidate.prediction_item({"error": "failed", **row}, request)["region_source"], "missing")

    def test_field_presence_cases(self):
        for module in (candidate, free):
            for value, expected in (("none", True), ("", False), ("unsupported", False), (None, False)):
                row = {"vlm_result": {"prediction": "real", "artifact_type": value, "target_scope": value}}
                pred = module.prediction_item(row, None) if module is candidate else module.prediction_item(row)
                for key, flag in (("artifact_type", "artifact_present"), ("target_scope", "target_present")):
                    self.assertEqual(bool(pred["parsed"] and pred[flag] and pred[key] == "none"), expected)
            empty = {"vlm_result": {}}
            pred = module.prediction_item(empty, None) if module is candidate else module.prediction_item(empty)
            self.assertFalse(pred["parsed"])

    def test_native_structural_gate(self):
        report = {"runtime_status": "ok", "native_text_normalized": "Tampered Content: object"}
        self.assertTrue(sida.report_parse_valid(report))
        self.assertIsNone(sida.native_prediction(report))
        gates = sida.condition_record(True, False, True, True, True)["conditions"]
        self.assertEqual(gates, {"parse": True, "prediction": False, "artifact": True, "target": True, "spatial": True})
        self.assertFalse(any(sida.condition_record(False, True, True, True, True)["conditions"].values()))
        trace = profile({"parse": False, "prediction": True, "artifact": True, "target": True, "spatial": True},
                        ["parse", "prediction", "artifact", "target", "spatial"])
        self.assertEqual(trace["pass_vector"], [False] * 5)
        for row in ({}, {"runtime_status": "ok", "native_text_normalized": " "},
                    {"runtime_status": "failed", "native_text_normalized": "class tampered"}):
            self.assertFalse(sida.report_parse_valid(row))

    def test_none_and_missing_end_to_end(self):
        outputs = [
            {"img_id": "explicit", "vlm_result": {"prediction": "real", "artifact_type": "none", "target_scope": "none"}},
            {"img_id": "missing", "vlm_result": {"prediction": "real"}},
            {"img_id": "blank", "vlm_result": {"prediction": "real", "artifact_type": "", "target_scope": ""}},
            {"img_id": "illegal", "vlm_result": {"prediction": "real", "artifact_type": "unsupported", "target_scope": "unsupported"}},
        ]
        annotations = [{"img_id": row["img_id"], "human_label": "real", "width": 100, "height": 80,
                        "dominant_artifact_type": "none", "target_scope": "none", "evidence_items": []}
                       for row in outputs]
        for module in (candidate, free):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ann, out, metrics = (root / name for name in ("ann.jsonl", "out.jsonl", "metrics.json"))
                ann.write_text("\n".join(json.dumps(row) for row in annotations) + "\n")
                out.write_text("\n".join(json.dumps(row) for row in outputs) + "\n")
                args = argparse.Namespace(annotations=ann, outputs=out, requests=None, output=metrics,
                                          predictions_output=None, summary_output=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    module.evaluate(args)
                result = json.loads(metrics.read_text())
                self.assertEqual(result["samples"], 4)
                self.assertEqual(result["parse_rate"], 1.0)
                self.assertEqual(result["artifact_type_accuracy"], 0.25)
                self.assertEqual(result["target_scope_accuracy"], 0.25)

    def test_model_box_source_survives_trace_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = {"img_id": "fallback", "metadata_for_evaluation_only": {
                "candidate_bbox": [0, 0, 101, 80], "width": 100, "height": 80}}
            output = {"img_id": "fallback", "vlm_result": {"prediction": "tampered",
                "artifact_type": "geometry_structure", "target_scope": "object", "evidence_bbox": [2, 2, 8, 8]}}
            normalized = candidate.prediction_item(output, request)
            frozen = {"img_id": "fallback", "prediction": normalized, "gold": {
                "prediction": "tampered", "artifact_type": "geometry_structure",
                "target_scope": "object", "evidence_bbox": [2, 2, 8, 8]},
                "prediction_correct": True, "artifact_correct": True, "target_correct": True, "bbox_iou": 1.0}
            spec = {"system_id": "fixture", "input_condition": "fixture"}
            for name, row in (("predictions", frozen), ("requests", request), ("raw_outputs", output)):
                spec[name] = root / (name + ".jsonl")
                spec[name].write_text(json.dumps(row) + "\n")
            with patch("export_jsg_audit_traces.ROOT", root):
                trace = td_trace_rows("fixture", spec)[0]
            self.assertEqual(trace["evaluated_region"]["provenance"], "model_emitted")
            self.assertTrue(trace["audit_profiles"]["primary_strict"]["joint"])

    def test_inherited_region_is_not_a_null_bbox_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann, out, req, metrics = (root/name for name in ("ann.jsonl", "out.jsonl", "req.jsonl", "metrics.json"))
            ann.write_text(json.dumps({"img_id":"negative", "human_label":"real", "width":100, "height":80,
                                      "dominant_artifact_type":"none", "target_scope":"none", "evidence_items":[]})+"\n")
            out.write_text(json.dumps({"img_id":"negative", "vlm_result":{"prediction":"tampered", "evidence_bbox":None}})+"\n")
            req.write_text(json.dumps({"img_id":"negative", "metadata_for_evaluation_only":{"candidate_bbox":[0,0,50,40]}})+"\n")
            args = argparse.Namespace(annotations=ann, outputs=out, requests=req, output=metrics,
                                      predictions_output=None, summary_output=None)
            with contextlib.redirect_stdout(io.StringIO()):
                candidate.evaluate(args)
            self.assertEqual(json.loads(metrics.read_text())["null_bbox_accuracy_when_expected"], 0.0)

    def test_manifest_validation(self):
        with self.assertRaises(ValueError):
            index_rows([{"img_id": "a"}, {"img_id": "a"}], "fixture")
        with self.assertRaises(ValueError):
            check_output_ids({"a": {}}, {"b": {}})
        with self.assertRaises(ValueError):
            image_dimensions({"img_id": "a"})

    def test_missing_reports_keep_denominator(self):
        annotations = [{"img_id": key, "human_label": "real", "width": 100, "height": 80,
                        "dominant_artifact_type": "none", "target_scope": "none", "evidence_items": []}
                       for key in ("present", "absent")]
        for module in (candidate, free):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ann, out, metrics, predictions = (root / name for name in ("ann.jsonl", "out.jsonl", "metrics.json", "pred.jsonl"))
                ann.write_text("\n".join(json.dumps(row) for row in annotations) + "\n")
                out.write_text(json.dumps({"img_id": "present", "vlm_result": {"prediction": "real"}}) + "\n")
                args = argparse.Namespace(annotations=ann, outputs=out, requests=None, output=metrics,
                                          predictions_output=predictions, summary_output=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    module.evaluate(args)
                report = json.loads(metrics.read_text())
                self.assertEqual(report["samples"], 2)
                self.assertEqual(report["missing_outputs"], 1)
                self.assertEqual(report["parse_rate"], 0.5)
                self.assertEqual(report["prediction_accuracy"], 0.5)
                self.assertEqual(report["artifact_type_accuracy"], 0.0)
                self.assertEqual(report["target_scope_accuracy"], 0.0)
                self.assertEqual(len(predictions.read_text().splitlines()), 2)

    def test_xfer_criteria_have_separate_expected_counts(self):
        expected = EXPECTED_JOINT_PASSES["JSG-Xfer/TD-EGFA"]
        self.assertEqual(expected["set_compatible"], expected["primary_strict"])
        self.assertEqual(expected["primary_or_secondary"], 30)


if __name__ == "__main__":
    unittest.main()
