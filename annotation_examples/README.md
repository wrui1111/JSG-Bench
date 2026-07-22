# IAA annotation examples

These four JSONL files are small schema examples derived from the local IAA overlap files. Each contains exactly one record from each SID image class in the fixed order `real`, `full_synthetic`, and `tampered`.

| File | Real ID | Full-synthetic ID | Tampered ID |
|---|---|---|---|
| `SID-Explain-IAA-reference-sample3.jsonl` | `a4382bd9c2dadf02` | `full_synthetic_006188` | `tampered_08313` |
| `SID-Explain-IAA-template-sample3.jsonl` | `a4382bd9c2dadf02` | `full_synthetic_006188` | `tampered_08313` |
| `SID-Hard-IAA-reference-sample3.jsonl` | `15d8d0876db5a6bf` | `full_synthetic_001545` | `tampered_07616` |
| `SID-Hard-IAA-template-sample3.jsonl` | `15d8d0876db5a6bf` | `full_synthetic_001545` | `tampered_07616` |

The paired files use matching IDs so readers can inspect the first-pass reference and blinded second-pass schema for the same samples. Image and mask files are not included. Download SID-Set separately as described in the repository README.

These examples are for schema inspection only. They are not a benchmark split, must not be used for tuning, and are insufficient to reproduce the paper's reported agreement statistics.
