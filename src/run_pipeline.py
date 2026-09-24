"""CLI entry point: runs the full multi-agent pipeline on a sample project and
prints a clean trace + final decision to the console. Also used by the
Streamlit app's "run" button under the hood.

Usage: python src/run_pipeline.py [project_id]
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rngd.orchestrator import run_pipeline  # noqa: E402


def main():
    project_id = sys.argv[1] if len(sys.argv) > 1 else "project_001"
    print(f"=== RNGD Concept-to-Schematic Multi-Agent Pipeline — {project_id} ===\n")
    state = run_pipeline(project_id)

    print("--- Agent Trace ---")
    for entry in state.trace:
        print(f"[{entry.step:02d}] {entry.agent:16s} {entry.action:35s} {entry.status:8s} {entry.latency_ms:8.1f}ms  {entry.summary}")

    print("\n--- Final Decision (Anthropic Arbiter) ---")
    fd = state.final_decision
    print(f"Decision: {fd.decision}  (confidence {fd.confidence:.2f})")
    print(f"Narrative: {fd.narrative}")
    print("Unresolved issues:")
    for u in fd.unresolved_issues:
        print(f"  - {u}")
    print("Recommended next steps:")
    for s in fd.recommended_next_steps:
        print(f"  - {s}")

    print(f"\nOutputs written to: outputs/{project_id}/")


if __name__ == "__main__":
    main()
