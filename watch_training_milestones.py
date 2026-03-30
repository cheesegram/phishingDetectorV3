import json
import time
from datetime import datetime
from pathlib import Path

PROGRESS_FILE = Path("trained_models") / "training_progress.json"
LOG_FILE = Path("trained_models") / "milestone_updates.log"
MILESTONES = [10, 25, 50, 75, 100]
POLL_SECONDS = 15


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_line(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_progress() -> dict | None:
    if not PROGRESS_FILE.exists():
        return None
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    write_line("Milestone watcher started.")
    reached = set()
    last_status = None

    while True:
        state = load_progress()
        if state is None:
            time.sleep(POLL_SECONDS)
            continue

        overall = float(state.get("overall_percent", 0.0))
        status = str(state.get("status", "running"))

        if status != last_status:
            write_line(f"Training status: {status}")
            last_status = status

        for m in MILESTONES:
            if overall >= m and m not in reached:
                reached.add(m)
                write_line(f"Milestone reached: {m}% (current: {overall:.2f}%)")

        if status in {"completed", "failed"}:
            if status == "failed":
                err = state.get("error", "Unknown error")
                write_line(f"Training failed: {err}")
            else:
                write_line("Training completed successfully.")
            break

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
