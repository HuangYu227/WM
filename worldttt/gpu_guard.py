"""Safe GPU discovery and experiment-process bookkeeping.

The WorldTTT host is shared by several users.  This module intentionally does
not expose a ``kill_all``/pattern based cleanup API: a launcher records the
exact PIDs and command lines it created, and cleanup is allowed only after the
owner and command line still match that record.  GPU selection is by UUID (or
an explicit index translated to a UUID), never by an implicit "first free"
assumption.

The helpers are usable on CPU-only CI; ``query_gpus`` raises a descriptive
``GpuProbeError`` when ``nvidia-smi`` is unavailable instead of pretending that
the cards are idle.
"""
from __future__ import annotations

import getpass
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


class GpuProbeError(RuntimeError):
    """The host did not provide a trustworthy NVIDIA inventory."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_gpu: int
    persistence_mode: str
    compute_mode: str

    @property
    def free_fraction(self) -> float:
        return self.memory_free_mib / max(self.memory_total_mib, 1)


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    owner: str
    command: str
    started_at: float


_GPU_QUERY = (
    "index,uuid,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,persistence_mode,compute_mode"
)


def _run_smi(args: Iterable[str]) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuProbeError(f"nvidia-smi probe failed: {exc}") from exc
    return proc.stdout


def _integer(value: str) -> int:
    # nvidia-smi appends units in CSV output (for example ``43432 MiB``).
    return int(value.strip().split()[0])


def query_gpus() -> list[GpuInfo]:
    """Return the current read-only GPU inventory.

    A second process query is deliberately separate; callers can record both
    snapshots immediately before and after a run.
    """
    raw = _run_smi([f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"])
    rows: list[GpuInfo] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 9:
            raise GpuProbeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            GpuInfo(
                index=int(fields[0]),
                uuid=fields[1],
                name=fields[2],
                memory_total_mib=_integer(fields[3]),
                memory_used_mib=_integer(fields[4]),
                memory_free_mib=_integer(fields[5]),
                utilization_gpu=_integer(fields[6]),
                persistence_mode=fields[7],
                compute_mode=fields[8],
            )
        )
    if not rows:
        raise GpuProbeError("nvidia-smi returned an empty GPU inventory")
    return rows


def query_compute_apps() -> list[dict[str, str | int]]:
    """Return compute-app rows without making ownership assumptions."""
    raw = _run_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    result = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise GpuProbeError(f"unexpected compute-app row: {line!r}")
        result.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": _integer(fields[3]),
            }
        )
    return result


def choose_gpu(*, min_free_mib: int = 0, max_utilization: int | None = None) -> GpuInfo:
    """Choose the least-used suitable card deterministically.

    This is a *selection aid*, not a claim that a card is reserved.  The
    caller must record the returned UUID and re-check occupancy immediately
    before launching a job.
    """
    candidates = [g for g in query_gpus() if g.memory_free_mib >= min_free_mib]
    if max_utilization is not None:
        candidates = [g for g in candidates if g.utilization_gpu <= max_utilization]
    if not candidates:
        raise GpuProbeError("no GPU satisfies the requested free-memory/utilization gate")
    return max(candidates, key=lambda g: (g.memory_free_mib, -g.utilization_gpu, -g.index))


def write_inventory(path: str | os.PathLike[str]) -> dict:
    """Persist a timestamped, read-only inventory for an experiment manifest."""
    payload = {
        "timestamp": time.time(),
        "user": getpass.getuser(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [asdict(g) for g in query_gpus()],
        "compute_apps": query_compute_apps(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def record_process(pid: int) -> ProcessRecord:
    """Record one child PID; reject processes owned by another user."""
    owner = getpass.getuser()
    status = Path(f"/proc/{pid}/status")
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not status.is_file() or not cmdline.is_file():
        raise ProcessLookupError(pid)
    text = status.read_text(errors="replace")
    uid_line = next((line for line in text.splitlines() if line.startswith("Uid:")), None)
    if uid_line is None:
        raise PermissionError(f"cannot verify owner of pid {pid}")
    # Comparing the effective uid through /proc is intentionally conservative;
    # callers cannot register an inaccessible or foreign process.
    effective_uid = int(uid_line.split()[1])
    if effective_uid != os.geteuid():
        raise PermissionError(f"pid {pid} is not owned by the current user")
    command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    return ProcessRecord(pid, owner, command, time.time())


def terminate_owned(record: ProcessRecord, *, grace_seconds: float = 10.0) -> None:
    """Terminate exactly one previously recorded child, after revalidation.

    No process-name or user-wide matching is performed.  If the PID has been
    recycled or its command line changed, the function refuses to signal it.
    """
    current = record_process(record.pid)
    if current.owner != record.owner or current.command != record.command:
        raise RuntimeError(f"refusing to signal pid {record.pid}: owner/command changed")
    os.kill(record.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not Path(f"/proc/{record.pid}").exists():
            return
        time.sleep(0.1)
    # It is still the exact same verified process; use SIGKILL only for this
    # PID, never a pattern or a username.
    current = record_process(record.pid)
    if current.owner != record.owner or current.command != record.command:
        raise RuntimeError(f"refusing SIGKILL for recycled pid {record.pid}")
    os.kill(record.pid, signal.SIGKILL)


__all__ = [
    "GpuInfo",
    "GpuProbeError",
    "ProcessRecord",
    "choose_gpu",
    "query_compute_apps",
    "query_gpus",
    "record_process",
    "terminate_owned",
    "write_inventory",
]
