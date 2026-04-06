"""
Real-time terminal GUI for monitoring phishing detector training progress.
Displays live progress bars, ETA, and detailed metrics for all training phases.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from threading import Thread


PROGRESS_FILE = Path("trained_models") / "training_progress.json"
ANSI_CLEAR = "\033[2J\033[H"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_CYAN = "\033[96m"
ANSI_GRAY = "\033[90m"


def make_progress_bar(percent, width=50, filled_char="█", empty_char="░"):
    """Create a colorized progress bar."""
    percent = max(0.0, min(100.0, float(percent)))
    filled = int(width * percent / 100.0)
    empty = width - filled
    
    # Color based on progress
    if percent == 100.0:
        color = ANSI_GREEN
    elif percent >= 75:
        color = ANSI_CYAN
    elif percent >= 50:
        color = ANSI_YELLOW
    else:
        color = ANSI_YELLOW
    
    bar = color + (filled_char * filled) + ANSI_GRAY + (empty_char * empty) + ANSI_RESET
    return bar


def format_time(seconds):
    """Convert seconds to human-readable format."""
    if seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def load_state():
    """Load the current training state from JSON file."""
    try:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        pass
    return None


def draw_phase(name, phase_data, overall_started=None):
    """Draw a single training phase with progress, status, and ETA."""
    percent = phase_data.get("percent", 0.0)
    status = phase_data.get("status", "pending")
    detail = phase_data.get("detail", "")
    
    # Status color
    if status == "completed":
        status_color = ANSI_GREEN
        status_symbol = "✓"
    elif status == "running":
        status_color = ANSI_CYAN
        status_symbol = "▶"
    elif status == "failed":
        status_color = ANSI_RED
        status_symbol = "✗"
    else:
        status_color = ANSI_GRAY
        status_symbol = "◯"
    
    # Build the line
    name_str = f"{name.upper():8}"
    bar = make_progress_bar(percent, width=45)
    pct_str = f"{percent:6.1f}%"
    status_str = f"{status_color}{status_symbol} {status:9}{ANSI_RESET}"
    
    line = f"  {name_str} {bar} {pct_str} {status_str}"
    if detail:
        line += f" {ANSI_GRAY}{detail}{ANSI_RESET}"
    
    return line


def draw_screen(state):
    """Draw the entire training status screen."""
    lines = []
    
    # Header
    lines.append("")
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}╔════ PHISHING DETECTOR - TRAINING MONITOR ════╗{ANSI_RESET}")
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}")
    
    # Overall progress
    overall = state.get("overall_percent", 0.0)
    overall_bar = make_progress_bar(overall, width=50)
    overall_status = state.get("status", "running")
    
    if overall_status == "completed":
        overall_symbol = f"{ANSI_GREEN}✓ COMPLETED{ANSI_RESET}"
    elif overall_status == "failed":
        overall_symbol = f"{ANSI_RED}✗ FAILED{ANSI_RESET}"
    else:
        overall_symbol = f"{ANSI_CYAN}● RUNNING{ANSI_RESET}"
    
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}  OVERALL    {overall_bar} {overall:6.1f}%  {overall_symbol}")
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}")
    
    # Started time
    started = state.get("started_at")
    if started:
        elapsed = time.time() - started
        elapsed_str = format_time(elapsed)
        lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}  Elapsed: {elapsed_str}")
    
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}")
    
    # Phase details
    phases = state.get("phases", {})
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}  PHASES:")
    
    for phase_name in ["bert", "url", "vision"]:
        phase = phases.get(phase_name, {})
        line = draw_phase(phase_name, phase, started)
        lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}{line}")
    
    # Footer
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}")
    updated = state.get("updated_at")
    if updated:
        update_time = datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}  Updated: {update_time}")
    
    # Error message if present
    if state.get("error"):
        lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}")
        lines.append(f"{ANSI_BOLD}{ANSI_CYAN}║{ANSI_RESET}  {ANSI_RED}ERROR: {state.get('error')}{ANSI_RESET}")
    
    lines.append(f"{ANSI_BOLD}{ANSI_CYAN}╚═══════════════════════════════════════════════╝{ANSI_RESET}")
    lines.append("")
    
    return "\n".join(lines)


def wait_for_progress_file(timeout=60):
    """Wait for the progress file to be created by the training process."""
    start_time = time.time()
    print(f"\n{ANSI_CYAN}Waiting for training to start...{ANSI_RESET}")
    print(f"{ANSI_GRAY}(Looking for progress file: {PROGRESS_FILE}){ANSI_RESET}\n")
    
    while time.time() - start_time < timeout:
        if PROGRESS_FILE.exists():
            time.sleep(0.5)  # Small delay to ensure file is written
            return True
        time.sleep(0.5)
    
    return False


def monitor_training():
    """Main monitoring loop."""
    print(ANSI_CLEAR, end="", flush=True)
    
    # Wait for training to start
    if not wait_for_progress_file():
        print(f"{ANSI_RED}✗ Training did not start within timeout. Check that train_models.py is running.{ANSI_RESET}\n")
        sys.exit(1)
    
    print(ANSI_CLEAR, end="", flush=True)
    
    try:
        while True:
            state = load_state()
            
            if state is None:
                print(f"{ANSI_RED}✗ Error reading progress file{ANSI_RESET}")
                time.sleep(1)
                continue
            
            # Draw screen
            screen = draw_screen(state)
            print(ANSI_CLEAR + screen, end="", flush=True)
            
            # Check if training is done
            status = state.get("status", "running")
            if status in ("completed", "failed"):
                if status == "completed":
                    print(f"\n{ANSI_GREEN}✓ Training completed successfully!{ANSI_RESET}\n")
                else:
                    print(f"\n{ANSI_RED}✗ Training failed. Check error message above.{ANSI_RESET}\n")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n{ANSI_YELLOW}Monitor stopped by user.{ANSI_RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{ANSI_RED}✗ Monitor error: {e}{ANSI_RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    monitor_training()
