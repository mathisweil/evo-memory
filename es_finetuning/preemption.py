"""Spot preemption handler for Google Cloud TPU/GPU VMs.

Google Cloud sends SIGTERM ~30 seconds before terminating a spot VM.
This module registers a handler that sets a flag, which the training
loop checks after each iteration to trigger an emergency checkpoint.
"""

import signal
import threading


class PreemptionHandler:
    """Detects spot VM preemption via SIGTERM."""

    def __init__(self):
        self.is_preempted = False
        self._lock = threading.Lock()
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        with self._lock:
            self.is_preempted = True
        print("\n*** PREEMPTION SIGNAL RECEIVED — "
              "will save checkpoint after current iteration ***")

    def check(self):
        """Check if preemption has been signaled."""
        return self.is_preempted
