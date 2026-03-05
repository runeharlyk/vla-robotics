"""Unique run identifier derived from the HPC job scheduler or a timestamp fallback."""

from __future__ import annotations

import os
from datetime import datetime


def run_id() -> str:
    """Return a short, unique run identifier.

    Resolution order:
      1. ``$LSB_JOBID``  - LSF (DTU HPC / bsub)
      2. ``$SLURM_JOB_ID`` - Slurm
      3. ``$PBS_JOBID`` - PBS/Torque
      4. Timestamp ``YYYYMMDD_HHMMSS`` as fallback (local / interactive runs)
    """
    for var in ("LSB_JOBID", "SLURM_JOB_ID", "PBS_JOBID"):
        jid = os.environ.get(var)
        if jid:
            return jid
    return datetime.now().strftime("%Y%m%d_%H%M%S")
