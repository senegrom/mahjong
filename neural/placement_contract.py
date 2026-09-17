"""Placement representation identities shared by artifacts and teacher replay.

Versions 1–3 consume frozen policy features. Version 4 has its own tower and
consumes a known observation layout instead. A head-file SHA-256 identifies
its trained weights independently of either representation identity.
"""
from __future__ import annotations

import re

FEATURE_VERSIONS = (1, 2, 3, 4)
OBSERVATION_PLANES = (97, 1012)  # Engine and Mortal-v4 layouts, respectively.


def validate_features(features: str, version: int, *, planes: int | None = None) -> None:
    """Reject unknown versions and identities, optionally against served planes."""
    if type(version) is not int or version not in FEATURE_VERSIONS:
        raise ValueError("unsupported placement feature version")
    if not isinstance(features, str):
        raise ValueError("placement head needs an immutable feature identity")
    if version == 4:
        choices = OBSERVATION_PLANES if planes is None else (planes,)
        if (any(type(n) is not int or n not in OBSERVATION_PLANES for n in choices)
                or features not in tuple(f"planes-{n}" for n in choices)):
            raise ValueError("placement head observation identity is incompatible")
    elif re.fullmatch(r"[0-9a-f]{16}", features) is None:
        raise ValueError("placement head needs an immutable feature fingerprint")


def validate_provenance(head: dict, *, planes: int) -> None:
    """A replay names both the exact trained file and its representation."""
    if (not isinstance(head, dict) or not isinstance(head.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", head["sha256"]) is None):
        raise ValueError("placement teacher needs immutable head provenance")
    validate_features(head.get("features"), head.get("feature_version"), planes=planes)
