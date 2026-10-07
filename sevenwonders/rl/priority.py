"""Lower the OS scheduling priority of the current process.

With a lower priority, normal-priority programs (a game, the desktop) always get the CPU
first and training only uses what is left over.
"""

from __future__ import annotations

import os
import sys

_WINDOWS_CLASSES = {"below_normal": 0x00004000, "idle": 0x00000040}
_NICE = {"below_normal": 10, "idle": 19}


def keep_awake() -> bool:
    """Ask Windows not to idle-sleep while this process runs (released when it exits).

    Does not change any power settings; the display may still turn off. Returns False on
    other platforms, where nothing is done."""
    if sys.platform != "win32":
        return False
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def lower_priority(level: str) -> None:
    """level: "below_normal" or "idle" (only runs when nothing else wants the CPU)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _WINDOWS_CLASSES[level])
    else:
        os.nice(_NICE[level])
