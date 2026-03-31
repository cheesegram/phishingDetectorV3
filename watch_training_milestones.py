import json
import time
import re
from datetime import datetime
from pathlib import Path

PROGRESS_FILE = Path("trained_models") / "training_progress.json"
LOG_FILE = Path("trained_models") / "milestone_updates.log"
POLL_SECONDS = 15
STEP_INTERVAL = 5


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


def parse_bert_step(detail: str) -> tuple[int, int] | tuple[None, None]:
    # Expected format: "Step X/Y"
    match = re.search(r"Step\s+(\d+)\s*/\s*(\d+)", str(detail))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    write_line(f"Step watcher started (interval={STEP_INTERVAL}).")
    reached_steps = set()
    last_status = None

    while True:
        state = load_progress()
        if state is None:
            time.sleep(POLL_SECONDS)
            continue

        overall = float(state.get("overall_percent", 0.0))
        status = str(state.get("status", "running"))
        bert = state.get("phases", {}).get("bert", {})
        bert_detail = bert.get("detail", "")
        current_step, total_steps = parse_bert_step(bert_detail)

        if status != last_status:
            write_line(f"Training status: {status}")
            last_status = status

        if current_step is not None and total_steps is not None:
            if current_step % STEP_INTERVAL == 0 and current_step not in reached_steps:
                reached_steps.add(current_step)
                write_line(
                    f"BERT step milestone: {current_step}/{total_steps} "
                    f"(overall: {overall:.2f}%)"
                )

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
