"""
Detector and TDC (Time-to-Digital Converter) utilities for WCTE beam analysis.

This module provides functions for processing TDC hits, removing duplicates,
and checking detector requirements.
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict


class NotTheFirstHit(Exception):
    """Exception raised when TDC hits are not ordered correctly."""

    pass


def deduplicate_tdc_hits(
    ids: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Remove duplicate TDC hits by keeping only the first hit per channel.

    When a TDC channel has multiple hits (which shouldn't happen but sometimes does),
    this function keeps only the first (earliest) hit and removes later ones.
    Returns cleaned arrays and a count of removed duplicates per channel.

    Parameters
    ----------
    ids : np.ndarray
        Array of channel IDs
    times : np.ndarray
        Array of hit times corresponding to each channel ID

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, dict)
        - ids_clean : Deduplicated channel IDs (same dtype as input)
        - times_clean : Deduplicated hit times (same dtype as input)
        - duplicates_removed : Dict mapping channel ID -> count of removed duplicates

    Raises
    ------
    NotTheFirstHit
        If a duplicate channel appears with an earlier time than the first occurrence,
        indicating the hits are not ordered correctly.

    Notes
    -----
    - Maintains the original dtype of input arrays
    - If no duplicates found, returns original arrays (even if they're lists)
    """
    seen = set()
    keep_ids = []
    keep_times = []
    duplicates_removed = defaultdict(int)
    stored_times = defaultdict(int)

    for ch, t in zip(ids, times):
        if ch in seen:
            # Count how many events are being removed per channel
            duplicates_removed[ch] += 1
            if stored_times[ch] > t:
                # If they are not ordered correctly, this is important
                raise NotTheFirstHit(
                    f"Channel {ch}: stored time {stored_times[ch]} > new time {t}"
                )
            continue

        seen.add(ch)
        keep_ids.append(ch)
        keep_times.append(t)
        stored_times[ch] = t

    # If we're not removing any channels, give it back as is
    if not duplicates_removed:
        return ids, times, duplicates_removed

    # Ensure that the format on the way out is the same as on the way in
    if isinstance(ids, np.ndarray):
        ids_clean = np.asarray(keep_ids, dtype=ids.dtype)
    else:
        ids_clean = np.asarray(keep_ids)

    if isinstance(times, np.ndarray):
        times_clean = np.asarray(keep_times, dtype=times.dtype)
    else:
        times_clean = np.asarray(keep_times)

    return ids_clean, times_clean, duplicates_removed


def tdc_requirement_met(group: Dict, tdc_set: set) -> bool:
    """
    Check if a detector group's TDC requirement is met.

    Supports two modes:
    - "all": All channels in the group must be present in tdc_set
    - "any_pair": At least one pair of channels must be completely present

    Parameters
    ----------
    group : dict
        Group specification with keys:
        - "mode" : str ("all" or "any_pair")
        - "channels" : list of channel IDs or list of channel pairs (for "any_pair")
    tdc_set : set
        Set of channel IDs that have TDC hits

    Returns
    -------
    bool
        True if the requirement is met, False otherwise
    """
    if group["mode"] == "any_pair":
        # At least one pair must be completely present
        return any(all(ch in tdc_set for ch in pair) for pair in group["channels"])
    # Default: all channels must be present
    return all(ch in tdc_set for ch in group["channels"])
