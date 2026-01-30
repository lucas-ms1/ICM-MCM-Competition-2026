"""
Industry-based communities for small-world network layer.
Defines G nodes: one per distinct celebrity_industry plus one "general audience" node.
"""

import numpy as np
import pandas as pd

INDUSTRY_COL = "celebrity_industry"
GENERAL_LABEL = "general"


def industry_communities_from_raw(
    raw: pd.DataFrame,
    *,
    include_general: bool = True,
    min_count: int = 0,
) -> tuple[int, list[str], dict[str, int]]:
    """
    Infer community indices from raw data using celebrity_industry.

    Parameters
    ----------
    raw : DataFrame
        Must have column celebrity_industry (or equivalent).
    include_general : bool
        If True, reserve index G-1 for "general" audience; else G = num industries.
    min_count : int
        Industries with fewer than this many rows can be grouped into "Other" (optional; 0 = no grouping).

    Returns
    -------
    G : int
        Number of communities.
    industry_order : list[str]
        Ordered list of industry labels; index g in 0..G-2 maps to industry_order[g];
        if include_general, index G-1 is "general".
    industry_to_g : dict[str, int]
        Mapping from normalized industry string to community index g.
    """
    if INDUSTRY_COL not in raw.columns:
        if include_general:
            return 1, [GENERAL_LABEL], {GENERAL_LABEL: 0}
        return 0, [], {}

    series = raw[INDUSTRY_COL].astype(str).str.strip().str.lower()
    series = series.replace("", np.nan).dropna()
    unique = series.unique().tolist()
    unique = [x for x in unique if x]
    if not unique:
        if include_general:
            return 1, [GENERAL_LABEL], {GENERAL_LABEL: 0}
        return 0, [], {}

    # Optional: group rare industries into "other"
    if min_count > 0:
        counts = series.value_counts()
        keep = [x for x in counts[counts >= min_count].index.tolist() if x]
        other_labels = [x for x in unique if x not in keep]
        if other_labels:
            unique = sorted(keep) + ["other"]
            industry_to_g = {k: i for i, k in enumerate(unique)}
            industry_to_g.update({lab: len(unique) - 1 for lab in other_labels})
        else:
            unique = sorted(unique)
            industry_to_g = {k: i for i, k in enumerate(unique)}
    else:
        unique = sorted(unique)
        industry_to_g = {k: i for i, k in enumerate(unique)}

    if include_general:
        industry_order = unique + [GENERAL_LABEL]
        industry_to_g[GENERAL_LABEL] = len(industry_order) - 1
        G = len(industry_order)
    else:
        industry_order = unique
        G = len(industry_order)

    # Rebuild industry_to_g for all industries in raw (normalized)
    series_all = raw[INDUSTRY_COL].astype(str).str.strip().str.lower()
    for v in series_all.dropna().unique():
        if not v:
            continue
        if v not in industry_to_g:
            if min_count > 0 and v not in keep:
                industry_to_g[v] = industry_order.index("other") if "other" in industry_order else 0
            else:
                industry_to_g[v] = industry_order.index(v) if v in industry_order else (G - 1 if include_general else 0)
    return G, industry_order, industry_to_g


def community_weights_from_raw(
    raw: pd.DataFrame,
    *,
    include_general: bool = True,
    mode: str = "empirical",
    general_fraction: float = 0.1,
    dirichlet_alpha: float = 1.0,
    rng: np.random.Generator | None = None,
    min_count: int = 0,
) -> tuple[int, np.ndarray, dict[str, int] | None]:
    """
    Compute community count G and weights w_g from raw data (industry-based + optional general).

    Parameters
    ----------
    raw : DataFrame
        Must have celebrity_industry.
    include_general : bool
        If True, add one "general" community.
    mode : "empirical" | "uniform" | "dirichlet"
        empirical: proportional to row counts per industry (+ general_fraction for general).
        uniform: w_g = 1/G.
        dirichlet: sample w ~ Dir(alpha,...,alpha); use for sensitivity.
    general_fraction : float
        When mode=="empirical" and include_general, fraction of total weight for general node (in (0,1)).
    dirichlet_alpha : float
        When mode=="dirichlet", concentration parameter per community.
    rng : np.random.Generator or None
        Used for mode=="dirichlet".
    min_count : int
        Passed to industry_communities_from_raw for optional "Other" grouping.

    Returns
    -------
    G : int
        Number of communities.
    w_g : array (G,)
        Non-negative weights summing to 1.
    industry_to_g : dict or None
        Mapping industry -> g; None if no industry column.
    """
    G, industry_order, industry_to_g = industry_communities_from_raw(
        raw, include_general=include_general, min_count=min_count
    )
    if G == 0:
        return 0, np.array([]), industry_to_g

    if mode == "uniform":
        w_g = np.ones(G) / G
        return G, w_g, industry_to_g

    if mode == "dirichlet":
        rng = rng or np.random.default_rng()
        w_g = rng.dirichlet(np.full(G, float(dirichlet_alpha)))
        return G, w_g, industry_to_g

    if mode == "empirical":
        if INDUSTRY_COL not in raw.columns:
            w_g = np.ones(G) / G
            return G, w_g, industry_to_g
        series = raw[INDUSTRY_COL].astype(str).str.strip().str.lower()
        counts = series.value_counts()
        count_per_g = np.zeros(G)
        for ind, cnt in counts.items():
            if not ind:
                continue
            g = industry_to_g.get(ind, G - 1 if include_general else 0)
            if g < G and (not include_general or g < G - 1):
                count_per_g[g] += cnt
        if include_general:
            industry_total = count_per_g[: G - 1].sum()
            if industry_total <= 0:
                w_g = np.ones(G) / G
            else:
                w_g = np.zeros(G)
                w_g[: G - 1] = (1.0 - general_fraction) * count_per_g[: G - 1] / industry_total
                w_g[G - 1] = general_fraction
        else:
            total = count_per_g.sum()
            w_g = count_per_g / total if total > 0 else np.ones(G) / G
        return G, w_g, industry_to_g

    raise ValueError('mode must be "empirical", "uniform", or "dirichlet"')
