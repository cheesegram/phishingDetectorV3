import json
import time
from pathlib import Path

PROGRESS_FILE = Path("trained_models") / "training_progress.json"


def make_bar(percent, width=40):
    percent = max(0.0, min(100.0, float(percent)))
    done = int(width * percent / 100.0)
    return "[" + ("#" * done) + ("-" * (width - done)) + "]"


def fmt_phase(name, phase):
    pct = phase.get("percent", 0.0)
    status = phase.get("status", "pending")
    detail = phase.get("detail", "")
    return f"{name.upper():6} {make_bar(pct, 28)} {pct:6.2f}%  {status:9}  {detail}"


def clear_screen():
    print("\033[2J\033[H", end="")


def main():
    while True:
        if not PROGRESS_FILE.exists():
            clear_screen()
            print("Waiting for progress file: trained_models/training_progress.json")
            time.sleep(1)
            continue

        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        overall = state.get("overall_percent", 0.0)
        status = state.get("status", "running")
        phases = state.get("phases", {})

        clear_screen()
        print("PHISHING DETECTOR TRAINING - LIVE PROGRESS")
        print()
        print(f"OVERALL {make_bar(overall)} {overall:6.2f}%  status={status}")
        print()
        print(fmt_phase("bert", phases.get("bert", {})))
        print(fmt_phase("url", phases.get("url", {})))
        print(fmt_phase("vision", phases.get("vision", {})))

        if status in {"completed", "failed"}:
            err = state.get("error")
            if err:
                print()
                print(f"ERROR: {err}")
            break

        time.sleep(1)


if __name__ == "__main__":
    main()
