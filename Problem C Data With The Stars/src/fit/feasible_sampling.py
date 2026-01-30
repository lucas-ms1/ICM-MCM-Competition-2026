"""
H3. Feasible set volume (inverse-only variant).

Implement only if "Option B" (constrained feasibility / inverse-only) is implemented.
There is no current Option B in the repo (no feasibility constraints or inverse sampling).

If implemented:
- Define the feasible set of fan share vectors consistent with observed eliminations
  (and optionally other constraints).
- Sample from this set (e.g. MCMC or rejection sampling on the simplex).
- For each (season, week), report dispersion of f_i across samples (e.g. std or
  credible intervals). Small dispersion => more certainty.

This module is a stub/placeholder until Option B exists.
"""

from __future__ import annotations

# Placeholder: no Option B (constrained feasibility) in the codebase yet.
# When Option B is implemented, add:
# - feasible_set_definition(elimination_events, rule_per_season) -> constraints
# - sample_feasible_fan_shares(constraints, n_samples, ...) -> list of f arrays
# - feasible_dispersion_per_week(samples_by_week) -> DataFrame with season, week, f_std, etc.


def feasible_fan_share_intervals_placeholder(
    cw=None,
    raw=None,
    elimination_events=None,
):
    """
    Placeholder for H3: uncertainty via feasible set volume (inverse-only).
    Requires Option B (constrained feasibility) to be implemented first.
    Returns None and a message until then.
    """
    return None, (
        "H3 (feasible set volume / dispersion) is not implemented: "
        "Option B (constrained feasibility / inverse-only) is not in the codebase. "
        "When Option B exists, sample feasible vote vectors and report dispersion "
        "(e.g. std or credible intervals) of f_i per (season, week)."
    )
