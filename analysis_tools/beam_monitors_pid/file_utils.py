"""
File I/O utilities for WCTE beam analysis.

This module provides functions for staging files locally from EOS (CERN storage)
and converting between EOS paths and XRootD URLs. Originally written by Sahar.
"""

import os
import hashlib
import shutil
import subprocess
from typing import Optional


def stage_local(
    src_eos_path: str, min_free_gb: int = 20, min_bytes: int = 1_000_000
) -> str:
    """
    Stage a file from EOS to local disk for faster access.

    Checks if there's enough free space in /tmp (default 20 GB minimum),
    then stages the file using xrdcp. Uses a hash-based naming scheme
    to avoid collisions across different directories.

    Parameters
    ----------
    src_eos_path : str
        Path to the file on EOS (must start with "/eos/")
    min_free_gb : int, optional
        Minimum free space required in /tmp (GB). Default is 20.
    min_bytes : int, optional
        Minimum file size after staging (bytes). Default is 1,000,000.

    Returns
    -------
    str
        Local path to the staged file, or empty string if no space in /tmp.

    Raises
    ------
    OSError
        If the local file is too small after xrdcp.
    AssertionError
        If the source path doesn't start with "/eos/".
    """
    st = shutil.disk_usage("/tmp")
    if st.free / 1e9 < min_free_gb:
        print("Not enough /tmp space; will stream from EOS")
        return ""

    # Use a unique local name to avoid collisions across different directories
    h = hashlib.md5(src_eos_path.encode()).hexdigest()[:8]
    local = f"/tmp/{os.path.basename(src_eos_path)}.{h}"

    def good(p):
        return os.path.exists(p) and os.path.getsize(p) >= min_bytes

    if not good(local):
        if os.path.exists(local):
            print("Cached local copy is too small; re-staging…")
            os.remove(local)
        print("Staging ROOT file to local disk (xrdcp)…")
        subprocess.run(
            ["xrdcp", "-f", to_xrootd(src_eos_path), local], check=True
        )
        if not good(local):
            raise OSError(f"Local file too small after xrdcp: {local}")

    print("Using local copy:", local)
    return local


def to_xrootd(path: str) -> str:
    """
    Convert an EOS path to an XRootD URL.

    Parameters
    ----------
    path : str
        EOS path (must start with "/eos/")

    Returns
    -------
    str
        XRootD URL for accessing the file via xrdcp

    Raises
    ------
    AssertionError
        If the path doesn't start with "/eos/"
    """
    assert path.startswith("/eos/")
    return "root://eosuser.cern.ch//eos" + path[4:]


def make_blocks(idx, max_block: int):
    """
    Partition an array of indices into contiguous blocks.

    Given a sorted array of indices, partitions them into blocks such that
    the span of indices in each block is less than max_block.

    Parameters
    ----------
    idx : np.ndarray
        Sorted array of indices
    max_block : int
        Maximum span of indices in a single block

    Returns
    -------
    list of tuple
        List of (start, end) tuples defining contiguous blocks
    """
    import numpy as np

    if idx.size == 0:
        return []

    blocks = []
    start = idx[0]
    last = idx[0]

    for v in idx[1:]:
        # if extending the block stays ≤ max_block, keep extending
        if (v - start) < max_block:
            last = v
        else:
            blocks.append((int(start), int(last) + 1))
            start = last = v

    blocks.append((int(start), int(last) + 1))
    return blocks
