# JSG audit v1.5

This release pairs the audit code with Supplementary Material S1 v1.5. It reproduces five Scale donor assignments, the matched-input repeat, and two Core target mappings.

- Each Scale assignment has 600 anonymous donor IDs. The verifier checks within-band bijections, absence of self-pairs, gate outcomes, and spatial-only transitions. Repeat rows contain paired SHA-256 hashes of the raw UTF-8 responses.
- Original Core mapping: 61 TD-EGFA passes, 32 SIDA passes, 25 common. Coarse mapping (person and animal to object on both sides): 61, 59, 45.
- Table S8 includes both raw and Holm-adjusted conditional McNemar tests. Table S9 lists the full prompt differences. The annotation notes distinguish eight annotation choices from the nine-label model protocol.
- The scorer uses explicit field validity, parse-gated downstream fields, and the selected region source. Native replay preserves 1,800 archived semantic judgments, 800 region decisions, and 100 native parse decisions.

Run `python experiments/verify_gate_release.py` from the extracted S1 with NumPy installed. The archive contains all input records for this replay. Rebuilding from native responses additionally requires the local source archive described in the run ledger.

Validation: 25 code tests, 3 standalone archive tests, and byte-identical native rebuild of all six exported data files. The manuscript contains 14 pages, 12 tables, and 5 figures, with zero unresolved references or citations. The attached release manifest records the final manuscript and archive digests.
