"""Legacy medicine-level experimental pipeline.

This runner belongs to the former transaction-level dataset and expects MEDICINE_ID,
DEMAND_CLEAN and inventory fields that do not exist in SCMD. It is intentionally
disabled as an executable entry point to prevent incompatible datasets being mixed
or synthetic VEN clinical classifications being generated.
"""
def main():
    raise SystemExit(
        "This is a legacy non-SCMD runner and is disabled for the SCMD migration. "
        "Run the SCMD phase runners in order instead."
    )

if __name__ == "__main__":
    main()
