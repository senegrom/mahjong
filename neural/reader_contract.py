"""Explicit producer declarations for learned proposal likelihood readers.

Saving or loading a model does not calibrate it. Only a producer that has fitted
and checked a reader for the stated proposal should set reader_proposal_version.
Missing, stale and unknown declarations remain loadable; the declaration is
carried through checkpoints, never invented.
"""
from __future__ import annotations

from collections.abc import Mapping

FIELD = "reader_proposal_version"


def metadata(net) -> dict:
    """Round-trip an explicit declaration; never label a fresh model implicitly."""
    if not hasattr(net, FIELD):
        return {}
    version = getattr(net, FIELD)
    if type(version) is not int or not 1 <= version < 2**32:
        raise ValueError("reader_proposal_version must be a positive u32 integer")
    return {FIELD: version}


def restore(net, payload: Mapping) -> None:
    """Validate a loaded declaration before setting it on the network."""
    if FIELD in payload:
        version = payload[FIELD]
        if type(version) is not int or not 1 <= version < 2**32:
            raise ValueError("reader_proposal_version must be a positive u32 integer")
        setattr(net, FIELD, version)
