#!/usr/bin/env python3
"""Compatibility entry point for the canonical JSG-Bench trace exporter.

The canonical pixel5d-v2 traces, frozen-contract validation, and manuscript
count assertions live in ``export_jsg_audit_traces.py``.  Keeping this alias
preserves earlier command lines without maintaining a second trace schema.
"""

from export_jsg_audit_traces import main


if __name__ == "__main__":
    main()
