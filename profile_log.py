"""
Lightweight logger for PROFILE_TIME=1 timing output.

All [PROFILE] lines are written to both stdout and a unique timestamped file
under profiling/ (or the directory set by PROFILE_LOG_DIR), so runs never
overwrite each other.  The file is opened lazily on first write.
"""
import os
import sys
import time

_log_file = None
_log_path = None


def _open_log():
    global _log_file, _log_path
    log_dir = os.environ.get('PROFILE_LOG_DIR', 'profiling')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    _log_path = os.path.join(log_dir, f'profile_{timestamp}.log')
    _log_file = open(_log_path, 'a', buffering=1)  # line-buffered
    print(f'[PROFILE] saving to {_log_path}', file=sys.stderr, flush=True)


def plog(msg: str) -> None:
    """Print msg to stdout and append it to the run's profile log file."""
    global _log_file
    print(msg, flush=True)
    if _log_file is None:
        _open_log()
    _log_file.write(msg + '\n')
