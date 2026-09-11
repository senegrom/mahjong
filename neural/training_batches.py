"""A reported learning generation must actually execute optimizer updates."""
from __future__ import annotations


def validate_learning(batch: int, epochs: int) -> None:
    for name, value in (("batch", batch), ("epochs", epochs)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def require_trainable_round(decisions: int, batch: int, epochs: int) -> None:
    validate_learning(batch, epochs)
    if decisions < batch:
        raise ValueError(
            f"No optimizer updates possible: rollout has {decisions} decisions but batch is {batch}. "
            "Use a smaller --batch or more --games; no learned generation will be saved."
        )


def require_updates(updates: int) -> None:
    if updates <= 0:
        raise RuntimeError("No optimizer updates were executed; refusing to save a learned generation")
