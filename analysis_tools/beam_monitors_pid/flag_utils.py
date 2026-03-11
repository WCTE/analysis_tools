"""
Bitmask and flag utilities for event quality tracking in WCTE beam analysis.

This module provides functions to pack and unpack event quality flags into/from
integer bitmasks for efficient storage and handling.
"""

from typing import Dict


def make_flag_map(flags):
    """
    Build a deterministic mapping from flag name to bit index.

    Creates a reproducible mapping by using sorted order of flag names.
    This ensures that the mapping is consistent across different runs.

    Parameters
    ----------
    flags : iterable of str
        Flag names (can be list, dict keys, etc.)

    Returns
    -------
    dict
        Mapping of flag name -> bit index (0, 1, 2, ...)
    """
    return {name: idx for idx, name in enumerate(sorted(flags))}


def write_event_quality_mask(flag_dict: Dict[str, bool], flag_map: Dict[str, int]) -> int:
    """
    Pack flags into an integer bitmask.

    Each flag is assigned a bit position according to flag_map. A flag is set
    (bit = 1) if its value is truthy, unset (bit = 0) otherwise.

    Parameters
    ----------
    flag_dict : dict[str, bool]
        Mapping of flag name -> truthy/falsey value (bool, 0/1, etc.)
    flag_map : dict[str, int]
        Mapping flag name -> bit index. If None, a deterministic mapping
        from sorted(flag_dict.keys()) is used.

    Returns
    -------
    int
        Integer bitmask where bit i (value 2**i) is set when the corresponding flag is true.

    Notes
    -----
    Missing flags are treated as False (not set). Modify the code if you want an error instead.
    """
    if flag_map is None:
        flag_map = make_flag_map(flag_dict.keys())

    mask = 0
    for flag_name, bit_idx in flag_map.items():
        if flag_name not in flag_dict:
            # Missing flags are treated as False (not set)
            continue
        if bool(flag_dict[flag_name]):
            mask |= 1 << bit_idx

    return mask


def read_event_quality_mask(
    mask: int, flag_map: Dict[str, int]
) -> Dict[str, bool]:
    """
    Unpack an integer bitmask back into a flag dictionary.

    Reverses the operation of write_event_quality_mask, recovering the
    original flag dictionary from a bitmask.

    Parameters
    ----------
    mask : int
        Integer bitmask created by write_event_quality_mask
    flag_map : dict[str, int]
        Mapping flag name -> bit index used when the mask was created

    Returns
    -------
    dict[str, bool]
        Flag name -> bool (True if that bit is set, False otherwise)
    """
    result: Dict[str, bool] = {}
    for flag_name, bit_idx in flag_map.items():
        result[flag_name] = bool(mask & (1 << bit_idx))
    return result
