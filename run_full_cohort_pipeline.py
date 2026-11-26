"""Legacy medicine-level experimental pipeline.

This runner belongs to the former transaction-level dataset and expects MEDICINE_ID,
DEMAND_CLEAN and inventory fields that do not exist in the SCMD source. It is
intentionally disabled to prevent mixing incompatible datasets or inventing VEN
clinical classifications. Use run_phase1_scmd.py through run_phase5_8_scmd.py.
"""
raise SystemExit(
    "This is a legacy non-SCMD runner and is disabled for the SCMD migration. "
    "Run the SCMD phase runners in order instead."
)
