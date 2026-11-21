import unittest
import json
from pathlib import Path
from verify_gate_release import replay, replay_pairs, replay_xfer, replay_stability, replay_mapping


class GateReleaseTest(unittest.TestCase):
    def test_stability_and_target_mappings(self):
        directory = Path(__file__).resolve().parents[1] / "verification"
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(len(replay_stability(directory, manifest)), 5)
        mapped = replay_mapping(directory, manifest)
        self.assertEqual(mapped["original_target"]["both"], 25)
        self.assertEqual(mapped["coarse_target"]["jaccard"], 0.600)

    def test_shipped_record_replay(self):
        result = replay(Path(__file__).resolve().parents[1]/"verification")
        self.assertEqual(sum(r["n"] for r in result.values()), 900)
        self.assertEqual(result["JSG-Xfer/TD-EGFA"]["set_count"], 21)
        self.assertEqual(result["JSG-Xfer/TD-EGFA"]["primary_or_secondary_count"], 30)

    def test_supplement_tables_vii_and_xiv(self):
        directory = Path(__file__).resolve().parents[1] / "verification"
        manifest = json.loads((directory / "manifest.json").read_text())
        primary = [json.loads(line) for line in (directory / "gate_records.jsonl").read_text().splitlines()]
        counts, changed = replay_pairs(directory, manifest, primary)
        self.assertEqual(counts, {"both": 94, "exit": 62, "entry": 46, "neither": 398})
        self.assertEqual(len({p["report_id"] for p in changed}), 108)
        self.assertEqual(sum(p["permuted_gates"] == [True, True, True, True, False]
                             for p in changed if p["transition"] == "exit"), 37)
        self.assertEqual(sum(p["matched_gates"] == [True, True, True, True, False]
                             for p in changed if p["transition"] == "entry"), 30)
        self.assertEqual(replay_xfer(primary, manifest["xfer_pool_counts"]), {
            "selected_pct": 21.0, "pool_weighted_pct": 10.92, "equal_source_pct": 10.59,
            "bootstrap_95_pct": [6.21, 16.25], "wilson_mover_95_pct": [7.44, 20.90]})


if __name__ == "__main__":
    unittest.main()
