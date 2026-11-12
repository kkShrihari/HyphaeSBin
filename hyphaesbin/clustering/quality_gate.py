"""
hyphaesbin.clustering.quality_gate -- per-stage clustering QUALITY GATE + WIDE coarse-to-fine PARAMETER SEARCH.

Per clustering stage (inside run_clustering):
  1. run the stage with the configured values (default), then search the stage's assignment-controlling parameters
  2. compare every candidate with the LAST ACCEPTED clustering
  3. ACCEPT the best passing candidate, or REJECT the stage (labels revert) and continue with the next stage
  4. every candidate (labels, parameters, metrics, decision) is saved to gate_candidates/<stage>/

LIMITATION (stated, not hidden): all gate metrics are internal (computed on the latent the clustering used). They show
consistency, not biological correctness. On CAMISIM, score the saved candidates with tools/gt_stage_report.py and
calibrate the acceptance rules / search ranges from that before trusting them on real data.

Search (per stage, parameters in STAGE_PARAMS, ranges in SEARCH_SPACE, overridable with gate_search_overrides)
  * screen one parameter at a time (others at their current best):
      coarse  : the parameter's own broad list (+ the configured default), e.g. min_cluster_size 5..100
      refine  : around the best TWO regions, between the neighbouring coarse values, with the parameter's own step,
                nearest values first (so a budget cut keeps the most informative ones)
  * combination grid: the top-2 passing values of each screened parameter, crossed (<= gate_combo_max new evals)
  * dependent settings stay valid: support/vote minima are scaled with k and never exceed k
  * identical label vectors are evaluated once; a parameter whose coarse values all give identical labels is not refined
  * the chosen value lying on the edge of the tested range is reported as 'boundary optimum - range may be insufficient'
  * hard caps: evaluations and seconds per stage (HDBSCAN refits have their own, larger caps)
  * a candidate that raises an error is recorded as failed and the search continues
  * DISABLED stages are never searched (run_clustering passes no parameters for them)

v10 changes
  * only CH, kNN agreement and assigned bp can FAIL a candidate (fail_metrics); silhouette / silhouette_macro / DB
    and the collective mean only enter the score (they rejected better candidates far more often than worse ones)
  * a split needs TNF AND coverage evidence (split_require_both); either one alone split single-genome bins
  * split pieces are re-joined only if intermixed in the TNF latent AND in the clustering latent (a split the
    clustering latent separates is no longer undone on TNF overlap alone)
  * split ranking uses the weakest accepted piece and sequence retention
  * rescue stages are ranked mainly by recovered sequence (rescue_assignment_weight)
  * collapse guard: a candidate that loses > max_bin_loss_frac (30 %) of the bins fails -- except in merge_stages
    (04d), whose job is merging fragments (mock test: 14 TE-split fragments -> 8 true genomes is a correct merge);
    in merge stages only kNN agreement + assigned bp can fail a candidate (merge_fail_metrics)
  * split_knn_max_drop_pp: separate per-piece TNF threshold for splits (default = knn_max_drop_pp)

Gate rules
  * internal indices (silhouette, silhouette_macro, CH higher = better; DB lower = better): relative change,
      FAIL if any < -tolerance (10 %)
  * kNN agreement: absolute percentage-point drop, FAIL if > knn_max_drop_pp (3 pp)
  * collective mean relative change >= -tolerance/2
  * assigned sequence may not fall by more than gate_max_bp_loss_pp (2 pp): discarding contigs must not be what
      makes a candidate look better
  * every comparison carries a 1e-9 numerical tolerance (an exactly 3.0 pp drop passes)
  * SPLIT stages: decided PER AFFECTED BIN (gate_split_per_bin), then the composite is checked globally:
      - fragments of one genome are re-joined (pieces intermixed in TNF space AND without a coverage difference)
      - contigs the split left unassigned are re-attached only with support: majority of TNF neighbours in one piece,
        not a TNF outlier (neighbour distance within the 95 % quantile of the bin's own), coverage shape fits the piece
        (|z| <= gate_split_reattach_cov_z); otherwise they stay unassigned (gate_split_reattach: false = never)
      - a bin's split is kept if its pieces are separable in TNF (local kNN@10 drop <= knn_max_drop_pp) OR differ in
        coverage shape (z >= gate_split_cov_z), and the bin loses <= gate_split_max_bp_loss_frac of its sequence
      - global: no bin disappears. Every decision is written to split_diagnostics.tsv
  * a candidate equal to the last accepted labels is 'accepted_no_change' (explicit)
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hyphaesbin.utils.logger import get_logger

log = get_logger("hyphaesbin.clustering.quality_gate")

EPS = 1e-9
METRIC_COLUMNS = ["n_clusters", "noise_pct", "noise_pct_eligible", "assigned_bp_pct", "size_gini", "silhouette",
                  "silhouette_macro", "calinski_harabasz", "davies_bouldin", "knn_agree@10", "knn_agree@30",
                  "knn_agree@50"]
# metric -> (direction, kind)  direction +1 = higher is better, -1 = lower is better; kind 'rel' | 'pp'
QUALITY_METRICS: Dict[str, Tuple[int, str]] = {
    "silhouette": (+1, "rel"), "silhouette_macro": (+1, "rel"), "calinski_harabasz": (+1, "rel"),
    "davies_bouldin": (-1, "rel"), "knn_agree@10": (+1, "pp"), "knn_agree@30": (+1, "pp"),
    "knn_agree@50": (+1, "pp"),
}
KNN_KS = (10, 30, 50)
NON_STAGE_NAMES = {"eligible_mask", "membership_probabilities", "contig_ids", "06_final_bin_ids"}

# ------------------------------------------------------------------------------------------------ search space
# kind: int | float | cat.  coarse: broad list.  step: refine step (None = no refine).  lo/hi: HARD validity bounds
# (None = open). Starting ranges, to be calibrated on CAMISIM -- not established optima.
_PROB = dict(kind="float", coarse=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], step=0.025, lo=0.0, hi=1.0)
_DIST = dict(kind="float", coarse=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5], step=0.05, lo=0.01, hi=2.0)
_KNN = dict(kind="int", coarse=[5, 10, 15, 20, 30, 50], step=1, lo=2, hi=None)
_SEP = dict(kind="float", coarse=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], step=0.1, lo=0.05, hi=None)
_MISFIT = dict(kind="float", coarse=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], step=0.025, lo=0.0, hi=1.0)
_Z = dict(kind="float", coarse=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0], step=0.25, lo=0.25, hi=None)
SEARCH_SPACE: Dict[str, Dict] = {
    "hdbscan_min_cluster_size": dict(kind="int", coarse=[5, 10, 20, 30, 40, 50, 60, 80, 100], step=1, lo=2, hi=None),
    "hdbscan_min_samples": dict(kind="int", coarse=[1, 3, 5, 10, 20, 30, 50], step=1, lo=1, hi=None),
    "hdbscan_method": dict(kind="cat", coarse=["eom", "leaf"], step=None, lo=None, hi=None),
    "hdbscan_epsilon": dict(kind="float", coarse=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8], step=0.025, lo=0.0, hi=None),
    "hdbscan_soft_prob_min": dict(kind="float", coarse=[0.0, 0.2, 0.4, 0.6, 0.8, 0.95], step=0.025, lo=0.0, hi=0.99),
    "noise_local_agreement": dict(_PROB),
    "noise_reference_prob_min": dict(kind="float", coarse=[0.0, 0.2, 0.4, 0.6, 0.8], step=0.025, lo=0.0, hi=1.0),
    "noise_neighbor_max_distance": dict(_DIST),
    "noise_local_k": dict(_KNN),
    "refinement_max_distance": dict(_DIST),
    "subcluster_merge_pieces": dict(kind="cat", coarse=[True, False], step=None, lo=None, hi=None),
    "subcluster_min_bin_contigs": dict(kind="int", coarse=[250, 500, 1000, 2000, 4000], step=None, lo=10, hi=None),
    "subcluster_min_cluster_size": dict(kind="int", coarse=[5, 10, 15, 20, 30, 50, 100], step=1, lo=2, hi=None),
    "subcluster_min_samples": dict(kind="int", coarse=[1, 3, 5, 10, 20], step=1, lo=1, hi=None),
    "subcluster_tnf_sep_max": dict(_SEP),
    "subcluster_cov_merge_tau": dict(_SEP),
    "subcluster_max_misfit": dict(_MISFIT),
    "prune_shape_z": dict(_Z),
    "prune_max_frac": dict(kind="float", coarse=[0.05, 0.1, 0.2, 0.3, 0.5], step=0.025, lo=0.0, hi=1.0),
    "binmerge_tnf_sep_max": dict(_SEP),
    "binmerge_cov_tau": dict(_SEP),
    "binmerge_max_misfit": dict(_MISFIT),
    "binmerge_misfit_z": dict(_Z),
    "binmerge_min_tnf_mix": dict(kind="float", coarse=[0.05, 0.1, 0.2, 0.3, 0.5], step=0.02, lo=0.0, hi=1.0),
    "reassign_max_z": dict(kind="float", coarse=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], step=0.5, lo=0.25, hi=None),
    "reassign_k": dict(_KNN),
    "tnf_consensus_min": dict(kind="float", coarse=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], step=0.025, lo=0.0, hi=1.0),
    "recruit_agreement": dict(_PROB),
    "recruit_max_dist": dict(_DIST),
    "recruit_k": dict(_KNN),
}
# k -> the support / vote minimum that must stay <= k (scaled proportionally with k)
DEPENDENT: Dict[str, str] = {"noise_local_k": "noise_local_min_support", "reassign_k": "reassign_min_votes",
                             "recruit_k": "recruit_min_support"}
_RESCUE = ["noise_local_agreement", "noise_neighbor_max_distance", "noise_reference_prob_min", "noise_local_k"]
STAGE_PARAMS: Dict[str, List[str]] = {
    "01_hdbscan_raw": ["hdbscan_method", "hdbscan_min_cluster_size", "hdbscan_min_samples", "hdbscan_epsilon"],
    "02_probability_filtered": ["hdbscan_soft_prob_min"],
    "03_after_rescue": _RESCUE,
    "04_after_refinement": ["refinement_max_distance"],
    "04b_after_subcluster": ["subcluster_merge_pieces", "subcluster_tnf_sep_max", "subcluster_cov_merge_tau", "subcluster_min_cluster_size",
                             "subcluster_min_samples", "subcluster_max_misfit", "subcluster_min_bin_contigs"],
    "04c_after_prune": ["prune_shape_z", "prune_max_frac"],
    "04d_after_binmerge": ["binmerge_tnf_sep_max", "binmerge_cov_tau", "binmerge_max_misfit", "binmerge_misfit_z",
                           "binmerge_min_tnf_mix"],
    "04d2_after_reassign": ["reassign_max_z", "reassign_k"],
    "04d3_after_tnf_consensus": ["tnf_consensus_min"],
    "04d4_after_late_rescue": _RESCUE,
    "04e_after_recruit": ["recruit_agreement", "recruit_max_dist", "recruit_k"],
}


def search_spec(field: str, overrides: Optional[Dict] = None) -> Optional[Dict]:
    spec = dict(SEARCH_SPACE.get(field) or {})
    ov = (overrides or {}).get(field)
    if isinstance(ov, (list, tuple)):
        spec["coarse"] = list(ov)
    elif isinstance(ov, dict):
        spec.update(ov)
    return spec or None


def _norm_val(v, kind):
    if kind == "int":
        return int(round(float(v)))
    if kind == "float":
        return round(float(v), 6)
    return v


def _clip(v, spec):
    if spec["kind"] == "cat":
        return v
    lo, hi = spec.get("lo"), spec.get("hi")
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return _norm_val(v, spec["kind"])


def coarse_values(spec: Dict, default) -> List:
    vals = []
    for v in [default] + list(spec["coarse"]):
        v = _clip(_norm_val(v, spec["kind"]), spec)
        if v not in vals:
            vals.append(v)
    return vals


def refine_values(spec: Dict, centre, tested_coarse: List) -> List:
    """Values between the coarse neighbours of `centre`, with the parameter's own step, nearest to the centre first."""
    if spec["kind"] == "cat" or not spec.get("step"):
        return []
    grid = sorted(set(tested_coarse))
    i = grid.index(centre)
    lo = grid[i - 1] if i > 0 else centre
    hi = grid[i + 1] if i + 1 < len(grid) else centre
    step = spec["step"]
    n = int(round((hi - lo) / step))
    vals = [_clip(_norm_val(lo + k * step, spec["kind"]), spec) for k in range(n + 1)]
    vals = [v for v in dict.fromkeys(vals) if v != centre]
    return sorted(vals, key=lambda v: abs(v - centre))


def on_boundary(spec: Dict, value, tested: List) -> bool:
    """True when `value` is the lowest/highest value tested AND not a hard validity bound."""
    if spec["kind"] == "cat" or len(tested) < 2:
        return False
    lo_t, hi_t = min(tested), max(tested)
    at_lo = value == lo_t and (spec.get("lo") is None or value > spec["lo"] + EPS)
    at_hi = value == hi_t and (spec.get("hi") is None or value < spec["hi"] - EPS)
    return bool(at_lo or at_hi)


def apply_value(cfg, field: str, value, base_cfg):
    """dataclasses.replace with dependent settings kept valid (support/vote minimum <= k, scaled with k)."""
    upd = {field: value}
    dep = DEPENDENT.get(field)
    if dep and hasattr(cfg, dep) and hasattr(base_cfg, field):
        k0, s0 = max(1, int(getattr(base_cfg, field))), int(getattr(base_cfg, dep))
        upd[dep] = int(max(1, min(int(value), round(s0 * int(value) / k0))))
    return dataclasses.replace(cfg, **upd)


def _cos_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, dtype=float))
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def _labels_key(labels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(labels, dtype=np.int64)).tobytes()).hexdigest()


class StageMetrics:
    """Computes the metric columns for a label vector. Built ONCE per clustering run.
    noise_pct          : % of the POOL (eligible + recruitable short contigs) that is unassigned
    noise_pct_eligible : % of the ELIGIBLE (clustered) contigs that is unassigned -- the HDBSCAN-comparable number"""

    def __init__(self, latent: np.ndarray, pool_mask: np.ndarray, lengths: np.ndarray, seed: int = 42,
                 silhouette_n: int = 10_000, knn_query_n: int = 10_000, knn_ref_max: int = 150_000,
                 threads: int = 8, eligible_mask: Optional[np.ndarray] = None):
        self.x = _cos_norm(latent)
        self.pool = np.asarray(pool_mask, dtype=bool)
        self.elig = np.asarray(eligible_mask, dtype=bool) if eligible_mask is not None else self.pool
        self.L = np.asarray(lengths, dtype=float)
        self.pool_bp = float(self.L[self.pool].sum()) or 1.0
        self.seed = int(seed)
        self.sil_n, self.q_n, self.r_max = int(silhouette_n), int(knn_query_n), int(knn_ref_max)
        self.threads = max(1, int(threads))
        self._cache: Dict[str, Dict] = {}
        try:
            import faiss  # noqa: F401
            self._faiss = True
        except Exception:
            self._faiss = False

    def _sample(self, idx: np.ndarray, n: int, salt: int) -> np.ndarray:
        if len(idx) <= n:
            return idx
        rng = np.random.default_rng(self.seed + salt)
        return np.sort(rng.choice(idx, size=n, replace=False))

    def _knn(self, ref: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
        if self._faiss:
            import faiss
            faiss.omp_set_num_threads(self.threads)
            index = faiss.IndexFlatL2(ref.shape[1])
            index.add(np.ascontiguousarray(ref, dtype=np.float32))
            _, nn = index.search(np.ascontiguousarray(q, dtype=np.float32), k)
            return nn
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=self.threads).fit(ref)
        return nn.kneighbors(q, return_distance=False)

    def compute(self, labels: np.ndarray) -> Dict[str, float]:
        labels = np.asarray(labels)
        key = _labels_key(labels)
        if key in self._cache:
            return dict(self._cache[key])
        out: Dict[str, float] = {c: float("nan") for c in METRIC_COLUMNS}
        assigned = np.flatnonzero(labels >= 0)
        out["noise_pct"] = 100.0 * float(((labels < 0) & self.pool).sum()) / max(1, int(self.pool.sum()))
        out["noise_pct_eligible"] = 100.0 * float(((labels < 0) & self.elig).sum()) / max(1, int(self.elig.sum()))
        out["assigned_bp_pct"] = 100.0 * float(self.L[(labels >= 0) & self.pool].sum()) / self.pool_bp
        lab_a = labels[assigned]
        uniq, inv = np.unique(lab_a, return_inverse=True)
        out["n_clusters"] = float(len(uniq))
        if len(uniq):
            out["size_gini"] = _gini(np.bincount(inv, weights=self.L[assigned]))
        if len(uniq) >= 2 and len(assigned) > len(uniq):
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples
            X = self.x[assigned]
            try:
                out["calinski_harabasz"] = float(calinski_harabasz_score(X, inv))
                out["davies_bouldin"] = float(davies_bouldin_score(X, inv))
            except Exception:
                pass
            s_idx = self._sample(np.arange(len(assigned)), self.sil_n, 1)
            if len(np.unique(inv[s_idx])) >= 2:
                try:
                    sv = silhouette_samples(X[s_idx], inv[s_idx])
                    out["silhouette"] = float(sv.mean())
                    out["silhouette_macro"] = float(np.mean([sv[inv[s_idx] == c].mean()
                                                             for c in np.unique(inv[s_idx])]))
                except Exception:
                    pass
            r_idx = self._sample(np.arange(len(assigned)), self.r_max, 2)
            q_idx = self._sample(r_idx, self.q_n, 3)
            kmax = min(max(KNN_KS) + 1, len(r_idx))
            if kmax >= 2:
                nn = self._knn(X[r_idx], X[q_idx], kmax)
                pos = np.searchsorted(r_idx, q_idx)                 # q is a subset of r -> drop self hit
                ref_lab = inv[r_idx]
                qlab = inv[q_idx]
                not_self = nn != pos[:, None]
                for k in KNN_KS:
                    kk = min(k, kmax - 1)
                    agree = np.full(len(q_idx), np.nan)
                    for row in range(len(q_idx)):
                        nb = nn[row][not_self[row]][:kk]
                        if len(nb):
                            agree[row] = (ref_lab[nb] == qlab[row]).mean()
                    out[f"knn_agree@{k}"] = float(np.nanmean(agree))
        self._cache[key] = dict(out)
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return out


# ------------------------------------------------------------------------------------------------ gate rule


# ------------------------------------------------------------------------------------------------ gate rule
def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def metric_changes(ref: Dict[str, float], cand: Dict[str, float],
                   metrics: Optional[Sequence[str]] = None) -> Dict[str, Tuple[str, float, float, float]]:
    """{metric: (kind, before, after, change)}. 'rel': signed relative change (> 0 better).
    'pp': signed ABSOLUTE change of the agreement fraction (0.99 -> 0.95 = -0.04 = -4 pp)."""
    out = {}
    for m, (direction, kind) in QUALITY_METRICS.items():
        if metrics is not None and m not in metrics:
            continue
        a, b = ref.get(m), cand.get(m)
        if not (_finite(a) and _finite(b)):
            continue
        a, b = float(a), float(b)
        ch = (b - a) if kind == "pp" else direction * (b - a) / max(abs(a), 1e-9)
        out[m] = (kind, a, b, ch)
    return out


DEFAULT_FAIL_METRICS = ("calinski_harabasz", "knn_agree@10", "knn_agree@30", "knn_agree@50")


def judge(ref: Dict[str, float], cand: Dict[str, float], tol: float, assign_w: float,
          knn_max_drop_pp: float = 3.0, max_bp_loss_pp: float = 2.0,
          fail_metrics: Optional[Sequence[str]] = DEFAULT_FAIL_METRICS, collective_rule: bool = False,
          max_bin_loss_frac: Optional[float] = 0.30) -> Dict:
    """Only metrics in `fail_metrics` can fail a candidate; all metrics enter the score."""
    use = list(QUALITY_METRICS)
    fset = set(fail_metrics) if fail_metrics is not None else set(use)
    ch = metric_changes(ref, cand, use)
    ref_defined = [m for m in use if _finite(ref.get(m))]
    if not ch:
        bad = bool(ref_defined)
        return {"passed": not bad, "worst": -1.0 if bad else 0.0, "worst_metric": "metrics_undefined" if bad else "",
                "collective": -1.0 if bad else 0.0, "score": -1e6 if bad else 0.0, "changes": {},
                "fail_reasons": "metrics undefined (collapse)" if bad else ""}
    failures = [(m, -1.0, "became undefined") for m in ref_defined if m not in ch and m in fset]
    rel_terms = []
    for m, (kind, a, b, c) in ch.items():
        if kind == "pp":
            if m in fset and -c * 100.0 > knn_max_drop_pp + EPS:
                failures.append((m, c, f"{a:.3f}->{b:.3f} ({100 * c:+.2f} pp > {knn_max_drop_pp:g} pp)"))
            rel_terms.append((b - a) / max(abs(a), 1e-9))
        else:
            if m in fset and c < -tol - EPS:
                failures.append((m, c, f"{a:.4g}->{b:.4g} ({100 * c:+.1f} % < -{100 * tol:.0f} %)"))
            rel_terms.append(c)
    collective = float(np.mean(rel_terms)) if rel_terms else 0.0
    if collective_rule and collective < -tol / 2.0 - EPS:
        failures.append(("collective", collective, f"mean relative change {100 * collective:+.1f} %"))
    # discarding sequence must not be what makes a candidate look better
    dbp = float(cand.get("assigned_bp_pct") or 0) - float(ref.get("assigned_bp_pct") or 0)
    if _finite(cand.get("assigned_bp_pct")) and dbp < -max_bp_loss_pp - EPS:
        failures.append(("assigned_bp", dbp / 100.0, f"assigned bp {dbp:+.2f} pp (> {max_bp_loss_pp:g} pp lost)"))
    # no stage may collapse the clustering: losing > max_bin_loss_frac of the bins is a failure (merges included).
    # Needed because silhouette/DB no longer fail candidates and CH/kNN reward fewer, larger clusters
    # (CAMISIM: hdbscan_epsilon=0.4 turned 41 bins into 4 and passed).
    n_ref, n_new = float(ref.get("n_clusters") or 0), float(cand.get("n_clusters") or 0)
    if max_bin_loss_frac is not None and n_ref >= 5 and n_new < (1.0 - max_bin_loss_frac) * n_ref - EPS:
        failures.append(("bin_collapse", (n_new - n_ref) / n_ref,
                         f"bins {int(n_ref)} -> {int(n_new)} (> {100 * max_bin_loss_frac:.0f} % lost)"))
    worst_m, (_, _, _, worst_c) = min(ch.items(), key=lambda kv: kv[1][3])
    gain = (float(cand.get("assigned_bp_pct") or 0) - float(ref.get("assigned_bp_pct") or 0)) / 100.0
    return {"passed": not failures, "worst": float(worst_c), "worst_metric": worst_m, "collective": collective,
            "score": collective + assign_w * gain, "changes": ch,
            "fail_reasons": "; ".join(f"{m}: {d}" for m, _, d in failures)}


def judge_split_global(ref: Dict[str, float], cand: Dict[str, float], n_bins_split: int, n_bins_tried: int,
                       weakest_piece_agreement: float = 0.0) -> Dict:
    """Rank supported splits by their weakest piece and retained sequence."""
    fails = []
    if float(cand.get("n_clusters") or 0) + EPS < float(ref.get("n_clusters") or 0):
        fails.append("bins disappeared")
    gain = (float(cand.get("assigned_bp_pct") or 0) - float(ref.get("assigned_bp_pct") or 0)) / 100.0
    return {"passed": not fails, "worst": gain, "worst_metric": "assigned_bp", "collective": 0.0,
            "score": weakest_piece_agreement + 0.5 * gain, "changes": {}, "fail_reasons": "; ".join(fails),
            "split_summary": f"{n_bins_split}/{n_bins_tried} affected bin(s) split"}


# ------------------------------------------------------------------------------------------------ per-bin splits
def _knn_idx(X_ref: np.ndarray, X_q: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    k = max(1, min(k, len(X_ref)))
    return NearestNeighbors(n_neighbors=k).fit(X_ref).kneighbors(X_q, return_distance=False)


def _cov_z(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Coverage-shape difference between two pieces: |median_a - median_b| / pooled MAD, mean over samples."""
    if P is None or P.shape[1] == 0 or len(a) == 0 or len(b) == 0:
        return float("nan")
    ma, mb = np.median(P[a], 0), np.median(P[b], 0)
    both = np.concatenate([P[a] - ma, P[b] - mb])
    mad = np.maximum(np.median(np.abs(both), 0) * 1.4826, 0.05)
    return float(np.mean(np.abs(ma - mb) / mad))


def split_per_bin(ref_l: np.ndarray, lab: np.ndarray, X: np.ndarray, P: Optional[np.ndarray], L: np.ndarray,
                  knn_max_drop_pp: float, cov_z_min: float, max_bp_loss_frac: float, rejoin_mix: float,
                  k: int = 10, sample_max: int = 4000, seed: int = 0, reattach: bool = True,
                  reattach_dist_q: float = 0.95, reattach_cov_z: float = 3.0, require_both: bool = True,
                  X_clust: Optional[np.ndarray] = None, rejoin: bool = True):
    """Decide a split candidate bin by bin. Returns (composite_labels, diagnostics_rows)."""
    rng = np.random.default_rng(seed)
    comp = ref_l.copy()
    rows = []
    parents = np.unique(ref_l[(ref_l >= 0) & (lab != ref_l)])
    for p in parents:
        idx = np.flatnonzero(ref_l == p)
        sub = lab[idx].copy()
        bp_par = float(L[idx].sum()) or 1.0
        lost_raw = float(L[idx][sub < 0].sum()) / bp_par
        n_raw = len(np.unique(sub[sub >= 0]))
        ass = np.flatnonzero(sub >= 0)
        r = {"parent_bin": int(p), "parent_contigs": len(idx), "parent_Mb": round(bp_par / 1e6, 3),
             "raw_pieces": n_raw, "raw_unassigned_bp_frac": round(lost_raw, 4)}
        if len(ass) < 3:
            rows.append({**r, "decision": "keep_parent", "reason": "split left < 3 assigned contigs"})
            continue
        s_ass = ass if len(ass) <= sample_max else np.sort(rng.choice(ass, sample_max, replace=False))
        Xs = X[idx[s_ass]]
        # 1) re-attach contigs the split left unassigned -- ONLY with support from all three checks:
        #    (a) majority (>= 50 %) of its TNF neighbours in one piece, (b) its mean TNF neighbour distance within the
        #    `reattach_dist_q` quantile of the assigned contigs' own neighbour distances (not an outlier),
        #    (c) its coverage shape fits that piece (mean |z| <= reattach_cov_z). Otherwise it stays unassigned.
        una = np.flatnonzero(sub < 0)
        n_reatt = n_refused = 0
        if len(una) and reattach:
            from sklearn.neighbors import NearestNeighbors
            kk = max(1, min(k, len(s_ass) - 1))
            nnm = NearestNeighbors(n_neighbors=kk + 1).fit(Xs)
            d_own, _ = nnm.kneighbors(Xs)
            d_max = float(np.quantile(d_own[:, 1:].mean(1), reattach_dist_q))
            d_u, nb = nnm.kneighbors(X[idx[una]], n_neighbors=kk)
            nl = sub[s_ass][nb]
            piece_model = {}
            for row, u in enumerate(una):
                vals, cnt = np.unique(nl[row], return_counts=True)
                tgt = vals[cnt.argmax()]
                ok = cnt.max() >= 0.5 * nl.shape[1] and float(d_u[row].mean()) <= d_max + EPS
                if ok and P is not None and P.shape[1]:
                    if tgt not in piece_model:
                        mem = idx[s_ass[sub[s_ass] == tgt]]
                        med = np.median(P[mem], 0)
                        piece_model[tgt] = (med, np.maximum(np.median(np.abs(P[mem] - med), 0) * 1.4826, 0.05))
                    med, sd = piece_model[tgt]
                    ok = float(np.mean(np.abs(P[idx[u]] - med) / sd)) <= reattach_cov_z + EPS
                if ok:
                    sub[u] = tgt
                    n_reatt += 1
                else:
                    n_refused += 1
        # 2) re-join fragments of one genome: pieces intermixed in TNF AND without a coverage difference
        nb = _knn_idx(Xs, Xs, min(k + 1, len(s_ass)))[:, 1:]
        nb_c = None
        if rejoin and X_clust is not None:               # neighbours in the latent the clustering used
            Xc = X_clust[idx[s_ass]]
            nb_c = _knn_idx(Xc, Xc, min(k + 1, len(s_ass)))[:, 1:]
        ls = sub[s_ass]
        pieces = list(np.unique(ls))
        parent_of = {q: q for q in pieces}

        def find(q):
            while parent_of[q] != q:
                q = parent_of[q]
            return q
        sizes = {q: int((ls == q).sum()) for q in pieces}
        components = {q: {q} for q in pieces}
        compatible = set()
        n_rejoin = 0
        for a, b in itertools.combinations(pieces if rejoin else (), 2):
            ia, ib = np.flatnonzero(ls == a), np.flatnonzero(ls == b)
            if len(ia) == 0 or len(ib) == 0:
                continue
            neighborhoods = (nb,) if nb_c is None else (nb, nb_c)
            mix = min(
                float(np.mean(ls[neighbors[source]] == target)) * (len(ls) - 1) / target_size
                for neighbors in neighborhoods
                for source, target, target_size in ((ia, b, len(ib)), (ib, a, len(ia)))
            )
            cz = _cov_z(P, idx[s_ass[ia]], idx[s_ass[ib]]) if P is not None else float("nan")
            if mix >= rejoin_mix and math.isfinite(cz) and cz < cov_z_min:
                compatible.add(frozenset((a, b)))
        for a, b in itertools.combinations(pieces if rejoin else (), 2):
            ra, rb = find(a), find(b)
            if ra != rb and all(frozenset((x, y)) in compatible
                                for x in components[ra] for y in components[rb]):
                keep, drop = (ra, rb) if sizes[ra] >= sizes[rb] else (rb, ra)
                parent_of[drop] = keep
                sizes[keep] += sizes[drop]
                components[keep].update(components.pop(drop))
                n_rejoin += 1
        roots = {q: find(q) for q in pieces}
        for q, rt in roots.items():
            if q != rt:
                sub[sub == q] = rt
        ls = sub[s_ass]
        final_pieces = np.unique(sub[sub >= 0])
        lost = float(L[idx][sub < 0].sum()) / bp_par
        # 3) evidence: local TNF separability and coverage difference of the two largest pieces
        neighbor_agreement = (ls[nb] == ls[:, None]).mean(axis=1)
        agree = min(float(neighbor_agreement[ls == q].mean()) if (ls == q).any() else 0.0
                    for q in final_pieces)
        drop_pp = 100.0 * (1.0 - agree)
        big = sorted(final_pieces, key=lambda q: -L[idx][sub == q].sum())[:2]
        cz = (_cov_z(P, idx[sub == big[0]], idx[sub == big[1]]) if (len(big) == 2 and P is not None)
              else float("nan"))
        ev_tnf = drop_pp <= knn_max_drop_pp + EPS
        ev_cov = math.isfinite(cz) and cz >= cov_z_min - EPS
        ok_loss = lost <= max_bp_loss_frac + EPS
        r.update({"reattached": n_reatt, "reattach_refused": n_refused, "rejoined_pairs": n_rejoin, "final_pieces": len(final_pieces),
                  "piece_Mb": ",".join(f"{L[idx][sub == q].sum() / 1e6:.2f}" for q in final_pieces),
                  "unassigned_bp_frac": round(lost, 4), "local_tnf_knn10_drop_pp": round(drop_pp, 3),
                  "cov_shape_z": round(cz, 3) if math.isfinite(cz) else "",
                  "min_piece_tnf_agree": round(agree, 4)})
        if len(final_pieces) < 2:
            dec, why = "keep_parent", "all pieces re-joined (one genome)"
        elif not ((ev_tnf and ev_cov) if require_both else (ev_tnf or ev_cov)):
            dec, why = "keep_parent", (f"evidence insufficient: TNF-separable {'yes' if ev_tnf else 'no'} "
                                       f"(drop {drop_pp:.1f} pp, max {knn_max_drop_pp:g}), coverage-distinct "
                                       f"{'yes' if ev_cov else 'no'} (z {cz:.2f}, min {cov_z_min:g})"
                                       f"{'; both required' if require_both else ''}")
        elif not ok_loss:
            dec, why = "keep_parent", f"split loses {100 * lost:.1f} % of the bin (> {100 * max_bp_loss_frac:.0f} %)"
        else:
            dec = "split"
            why = "+".join(n for n, ok in (("tnf_separable", ev_tnf), ("coverage_distinct", ev_cov)) if ok)
            r["sep_score"] = round(agree, 4)
            comp[idx] = sub
        rows.append({**r, "decision": dec, "reason": why})
    return comp, rows


# ------------------------------------------------------------------------------------------------ the gate
@dataclasses.dataclass
class GateSettings:
    enabled: bool = True
    sweep: bool = True
    tolerance: float = 0.10
    assignment_weight: float = 0.5
    max_evals_per_stage: int = 40
    stage_time_budget_s: float = 2400.0
    run_name: str = ""
    knn_max_drop_pp: float = 3.0
    split_stages: Tuple[str, ...] = ("04b_after_subcluster",)
    save_candidates: bool = True
    refine_regions: int = 2
    combo_max: int = 8
    split_per_bin: bool = True
    split_cov_z: float = 2.0
    split_max_bp_loss_frac: float = 0.05
    split_rejoin_mix: float = 0.3
    split_reattach: bool = True
    split_reattach_dist_q: float = 0.95
    split_reattach_cov_z: float = 3.0
    search_overrides: Optional[Dict] = None
    max_bp_loss_pp: float = 2.0
    exclude: Tuple[str, ...] = ()           # parameters never searched (configured value kept fixed)
    split_require_both: bool = True          # split needs TNF AND coverage evidence
    fail_metrics: Tuple[str, ...] = DEFAULT_FAIL_METRICS   # metrics allowed to FAIL a candidate
    collective_rule: bool = False            # old 'mean relative change >= -tolerance/2' fail rule
    rescue_stages: Tuple[str, ...] = ("03_after_rescue", "04d4_after_late_rescue", "04e_after_recruit")
    rescue_assignment_weight: float = 3.0
    max_bin_loss_frac: float = 0.30          # a candidate losing more than this fraction of bins fails (collapse guard)
    merge_stages: Tuple[str, ...] = ("04d_after_binmerge",)   # stages whose JOB is merging: collapse guard not applied
    # merging two latent sub-clusters of ONE genome always lowers CH/silhouette (they were apart in the latent), so in
    # merge stages only kNN agreement + assigned bp can fail a candidate (mock test: CH -12..-24 % blocked correct merges)
    merge_fail_metrics: Tuple[str, ...] = ("knn_agree@10", "knn_agree@30", "knn_agree@50")
    split_knn_max_drop_pp: Optional[float] = None   # split-only TNF threshold per piece; None = knn_max_drop_pp    # rescue stages are ranked mainly by recovered sequence


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


class QualityGate:
    def __init__(self, metrics: StageMetrics, settings: GateSettings, outdir: Path,
                 split_metrics: Optional[StageMetrics] = None, split_cov: Optional[np.ndarray] = None):
        self.M, self.S, self.outdir = metrics, settings, Path(outdir)
        self.Msplit = split_metrics          # metrics on the latent SPLITS are decided in (TNF); None -> fused rule
        self.split_cov = split_cov           # per-contig coverage shape (for split evidence); may be None
        self.accepted_labels: Optional[np.ndarray] = None
        self.accepted_metrics: Optional[Dict] = None
        self.accepted_stage = ""
        self.rows: List[Dict] = []
        self.cand_rows: List[Dict] = []
        self.split_rows: List[Dict] = []
        self.chosen_params: Dict[str, Dict] = {}
        self.boundary_warnings: List[str] = []
        self._evals_wd: List = []
        if self.S.save_candidates:
            shutil.rmtree(self.outdir / "gate_candidates", ignore_errors=True)

    def workdir(self) -> Path:
        return getattr(self, "_workdir", None) or self.outdir

    def _row(self, stage, metrics, decision, detail="", params=None):
        r = {"run": self.S.run_name, "stage": stage, **{c: metrics.get(c) for c in METRIC_COLUMNS},
             "decision": decision, "detail": detail, "params": json.dumps(params or {}, default=_jsonable)}
        self.rows.append(r)
        return r

    def baseline(self, stage: str, labels: np.ndarray) -> np.ndarray:
        m = self.M.compute(labels)
        self.accepted_labels, self.accepted_metrics, self.accepted_stage = labels.copy(), m, stage
        self._row(stage, m, "baseline")
        log.info(f"GATE  {stage:<26} baseline  {self._fmt(m)}")
        return labels

    def record_only(self, stage: str, labels: np.ndarray, why: str) -> np.ndarray:
        m = self.M.compute(labels)
        self.accepted_labels, self.accepted_metrics, self.accepted_stage = labels.copy(), m, stage
        self._row(stage, m, "accepted_not_gated", why)
        log.info(f"GATE  {stage:<26} not gated ({why})  {self._fmt(m)}")
        return labels

    @staticmethod
    def _fmt(m):
        def f(k, d=3):
            v = m.get(k)
            return "nan" if v is None or not _finite(v) else f"{float(v):.{d}f}"
        return (f"bins={int(m.get('n_clusters') or 0)} noise(pool)={f('noise_pct', 1)}% "
                f"noise(eligible)={f('noise_pct_eligible', 1)}% bp={f('assigned_bp_pct', 1)}% "
                f"sil={f('silhouette')} CH={f('calinski_harabasz', 0)} DB={f('davies_bouldin')} "
                f"kNN10={f('knn_agree@10')} kNN50={f('knn_agree@50')}")

    def step(self, stage: str, run_fn: Callable, base_cfg, params=None,
             max_evals: Optional[int] = None, time_budget_s: Optional[float] = None):
        """run_fn(cfg) -> (labels, extra). params: list of cfg field names (ranges from SEARCH_SPACE), or the legacy
        {field: (kind, lo, hi)} dict (field names are used, ranges come from SEARCH_SPACE), or None = no search.
        Returns (labels, extra, cfg_used, accepted: bool)."""
        prof = None
        manual_end = None
        try:
            from hyphaesbin.utils.resource_profiler import manual_start, manual_end
            prof = manual_start(f"gate/{self.S.run_name}/{stage}")
        except Exception:
            manual_end = None
        try:
            return self._step(stage, run_fn, base_cfg, params, max_evals, time_budget_s)
        finally:
            if prof is not None and manual_end is not None:
                try:
                    manual_end(prof)
                except Exception:
                    pass

    def _save_candidate(self, stage, cid, lab, pdesc, j, m, raw_lab=None):
        if not self.S.save_candidates:
            return
        try:
            d = self.outdir / "gate_candidates" / stage
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{cid}.npy", np.asarray(lab, dtype=np.int32))
            if raw_lab is not None:
                np.save(d / f"{cid}_raw_split.npy", np.asarray(raw_lab, dtype=np.int32))
            (d / f"{cid}.json").write_text(json.dumps({
                "stage": stage, "params": pdesc, "passed": j["passed"], "score": j["score"],
                "fail_reasons": j.get("fail_reasons", ""), "split_summary": j.get("split_summary", ""),
                "reference_stage": self.accepted_stage, "labels_sha256": _labels_key(lab),
                "reference_labels_sha256": (_labels_key(self.accepted_labels)
                                            if self.accepted_labels is not None else None),
                "metrics": {k: _jsonable(v) for k, v in m.items()}}, indent=1, default=_jsonable))
        except OSError as e:
            log.warning(f"could not save gate candidate {stage}/{cid}: {e}")

    def _step(self, stage, run_fn, base_cfg, params, max_evals, time_budget_s):
        _stale = self.outdir / f"REPORTS_FROM_REJECTED_STAGE_{stage}.txt"
        if _stale.exists():
            _stale.unlink()
        max_evals = int(max_evals or self.S.max_evals_per_stage)
        budget = float(time_budget_s or self.S.stage_time_budget_s)
        if isinstance(params, dict):
            params = list(params)
        params = [f for f in (params or []) if hasattr(base_cfg, f) and getattr(base_cfg, f) is not None
                  and search_spec(f, self.S.search_overrides) and f not in set(self.S.exclude or ())]
        ref_l, ref_m = self.accepted_labels, self.accepted_metrics
        ref_key = _labels_key(ref_l) if ref_l is not None else None
        split_stage = stage in tuple(self.S.split_stages or ())
        per_bin = (self.S.enabled and split_stage and self.S.split_per_bin
                   and self.Msplit is not None and ref_l is not None)
        t0 = time.time()
        evals: List[Tuple] = []            # (judge, labels, extra, cfg, pdesc, metrics, key)
        seen: Dict[str, Tuple] = {}
        cfg_seen: Dict[str, int] = {}      # config signature -> eval index (reuse)
        work_root = self.outdir / "_gate_work" / stage
        isolate = bool(params) and self.S.enabled and self.S.sweep
        shutil.rmtree(work_root, ignore_errors=True)
        self._evals_wd = []

        def sig(cfg_c):
            return json.dumps({f: getattr(cfg_c, f) for f in params}, sort_keys=True, default=str)

        n_attempts = [0]                    # EVERY attempted evaluation counts (failures included)
        eval_cid: List[str] = []            # candidate id of each successful evaluation

        def budget_left(reserve: int = 0):
            return n_attempts[0] < max_evals - reserve and time.time() - t0 <= budget

        def _try(cfg_c, pdesc):
            s = sig(cfg_c)
            if s in cfg_seen:
                return cfg_seen[s]
            cid = f"c{n_attempts[0]:02d}"
            n_attempts[0] += 1
            i = len(evals)
            self._workdir = (work_root / cid) if isolate else self.outdir
            if isolate:
                self._workdir.mkdir(parents=True, exist_ok=True)
            t_fit = time.time()
            try:
                lab, extra = run_fn(cfg_c)
            except Exception as ex:
                if i == 0:
                    raise                       # the configured default must work
                err = f"error: {type(ex).__name__}: {ex}"[:300]
                log.warning(f"GATE  {stage}: candidate {cid} {pdesc} raised {type(ex).__name__}: {ex} -> recorded as failed")
                self.cand_rows.append({"run": self.S.run_name, "stage": stage, "candidate": cid,
                                       "params": json.dumps(pdesc, default=_jsonable), "passed": False,
                                       "fail_reasons": err})
                if self.S.save_candidates:
                    try:
                        d = self.outdir / "gate_candidates" / stage
                        d.mkdir(parents=True, exist_ok=True)
                        (d / f"{cid}.json").write_text(json.dumps({"stage": stage, "params": pdesc, "passed": False,
                                                                   "error": err}, default=_jsonable))
                    except OSError:
                        pass
                cfg_seen[s] = -1
                return -1
            lab = np.asarray(lab)
            raw_lab = None
            split_rows = []
            if per_bin:
                raw_lab = lab
                lab, split_rows = split_per_bin(
                    ref_l, raw_lab, self.Msplit.x, self.split_cov, self.M.L,
                    (self.S.split_knn_max_drop_pp if self.S.split_knn_max_drop_pp is not None
                     else self.S.knn_max_drop_pp),
                    self.S.split_cov_z, self.S.split_max_bp_loss_frac, self.S.split_rejoin_mix,
                    reattach=self.S.split_reattach, reattach_dist_q=self.S.split_reattach_dist_q,
                    reattach_cov_z=self.S.split_reattach_cov_z, require_both=self.S.split_require_both,
                    X_clust=self.M.x, rejoin=getattr(cfg_c, "subcluster_merge_pieces", True))
                for r in split_rows:
                    self.split_rows.append({"run": self.S.run_name, "stage": stage, "candidate": cid,
                                            "params": json.dumps(pdesc, default=_jsonable), **r})
            key = _labels_key(lab)
            note = ""
            if ref_key is not None and key == ref_key:
                j = {"passed": True, "worst": 0.0, "worst_metric": "", "collective": 0.0, "score": 0.0,
                     "changes": {}, "unchanged": True, "fail_reasons": ""}
                m = ref_m
            elif key in seen:
                j, m = seen[key]
                note = "duplicate_labels"
            else:
                m = self.M.compute(lab)
                if per_bin:
                    nsplit = sum(r["decision"] == "split" for r in split_rows)
                    seps = [r.get("sep_score", 0.0) for r in split_rows if r["decision"] == "split"]
                    j = judge_split_global(ref_m, m, nsplit, len(split_rows), min(seps, default=0.0))
                elif ref_m:
                    aw = (self.S.rescue_assignment_weight if stage in tuple(self.S.rescue_stages or ())
                          else self.S.assignment_weight)
                    j = judge(ref_m, m, self.S.tolerance, aw, self.S.knn_max_drop_pp, self.S.max_bp_loss_pp,
                              fail_metrics=(self.S.merge_fail_metrics if stage in tuple(self.S.merge_stages or ())
                                            else self.S.fail_metrics),
                              collective_rule=self.S.collective_rule,
                              max_bin_loss_frac=(None if stage in tuple(self.S.merge_stages or ())
                                                 else self.S.max_bin_loss_frac))
                else:
                    j = {"passed": True, "worst": 0.0, "worst_metric": "", "collective": 0.0, "score": 0.0,
                         "changes": {}, "fail_reasons": ""}
                seen[key] = (j, m)
            raw = {}
            for k in KNN_KS:
                a_ = ref_m.get(f"knn_agree@{k}") if ref_m else None
                b_ = m.get(f"knn_agree@{k}")
                raw[f"knn@{k}_before"] = round(float(a_), 5) if _finite(a_) else ""
                raw[f"knn@{k}_after"] = round(float(b_), 5) if _finite(b_) else ""
                raw[f"knn@{k}_change_pp"] = (round(100 * (float(b_) - float(a_)), 3)
                                             if (_finite(a_) and _finite(b_)) else "")
            self.cand_rows.append({"run": self.S.run_name, "stage": stage, "candidate": cid,
                                   "params": json.dumps(pdesc, default=_jsonable),
                                   "seconds": round(time.time() - t_fit, 1),
                                   **{c: m.get(c) for c in METRIC_COLUMNS}, **raw,
                                   "unchanged": bool(j.get("unchanged")), "note": note, "passed": j["passed"],
                                   "score": round(j["score"], 5), "worst_metric": j["worst_metric"],
                                   "worst_change": round(j["worst"], 5), "collective_change": round(j["collective"], 5),
                                   "split_summary": j.get("split_summary", ""),
                                   "fail_reasons": j.get("fail_reasons", "")})
            self._save_candidate(stage, cid, lab, pdesc, j, m, raw_lab)
            evals.append((j, lab, extra, cfg_c, pdesc, m, key))
            eval_cid.append(cid)
            self._evals_wd.append(self._workdir)
            cfg_seen[s] = i
            return i

        def _round_robin(queues: Dict[str, List]):
            qs = {f: list(q) for f, q in queues.items()}
            while any(qs.values()):
                for f in list(qs):
                    if qs[f]:
                        yield f, qs[f].pop(0)

        _try(base_cfg, {"default": True})
        t_default = time.time() - t0
        do_sweep = bool(self.S.enabled and self.S.sweep and params and t_default <= budget / 3.0)
        if self.S.enabled and self.S.sweep and params and not do_sweep:
            log.info(f"GATE  {stage:<26} default run took {t_default:.0f}s (> 1/3 of {budget:.0f}s) -> no search")
        searched: Dict[str, List] = {}
        not_tested: Dict[str, List] = {}
        coarse: Dict[str, List] = {}
        if do_sweep:
            specs = {f: search_spec(f, self.S.search_overrides) for f in params}
            dflt = {f: _norm_val(getattr(base_cfg, f), specs[f]["kind"]) for f in params}
            coarse = {f: coarse_values(specs[f], dflt[f]) for f in params}
            results: Dict[str, Dict] = {f: {dflt[f]: 0} for f in params}
            reserve = (min(int(self.S.combo_max) + 1, max(0, max_evals // 4)) if len(params) >= 2 else 0)
            # phase 1: COARSE screen of EVERY parameter (round-robin, one at a time from the configured values), so no
            # parameter is starved by another's budget use
            for f, v in _round_robin({f: [v for v in coarse[f] if v not in results[f]] for f in params}):
                if not budget_left(reserve):
                    break
                results[f][v] = _try(apply_value(base_cfg, f, v, base_cfg), {f: v})
            # phase 2: REFINE around the best regions of every parameter (round-robin)
            rq = {}
            for f in params:
                spec = specs[f]
                ok = {v: i for v, i in results[f].items() if i >= 0}
                if len({evals[i][6] for i in ok.values()}) <= 1:
                    if len(ok) > 1:
                        log.info(f"GATE  {stage:<26} {f}: every tested value gave identical labels -> no refine")
                    continue
                if spec["kind"] == "cat" or not spec.get("step"):
                    continue
                passing = [v for v, i in ok.items() if evals[i][0]["passed"]]
                pool = passing or list(ok)
                tested = sorted(v for v in ok if v in coarse[f])
                centres = [c for c in sorted(pool, key=lambda v: evals[ok[v]][0]["score"], reverse=True)
                           if c in tested][:max(1, int(self.S.refine_regions))]
                qs = [refine_values(spec, c, tested) for c in centres]
                rq[f] = [v for v in dict.fromkeys(itertools.chain.from_iterable(itertools.zip_longest(*qs)))
                         if v is not None and v not in results[f]]
            for f, v in _round_robin(rq):
                if v in results[f]:
                    continue
                if not budget_left(reserve):
                    break
                results[f][v] = _try(apply_value(base_cfg, f, v, base_cfg), {f: v})
            def _summarise():
                bv, ft = {}, {}
                not_tested.clear()
                for f in params:
                    ok = {v: i for v, i in results[f].items() if i >= 0}
                    searched[f] = sorted(ok, key=lambda v: (str(type(v)), v))
                    miss = ([v for v in coarse[f] if v not in results[f]] +
                            [v for v in rq.get(f, []) if v not in results[f]])
                    if miss:
                        not_tested[f] = miss
                    passing = {v: i for v, i in ok.items() if evals[i][0]["passed"]}
                    ranked_v = sorted(passing, key=lambda v: evals[passing[v]][0]["score"], reverse=True)
                    ft[f] = ranked_v[:2]
                    if ranked_v:
                        bv[f] = ranked_v[0]
                return bv, ft
            best_val, field_top = _summarise()
            # phase 3: the COMBINED best (each parameter at its best screened value), then the top-2 combination grid
            cfg_comb = base_cfg
            for f, v in best_val.items():
                if v != dflt[f]:
                    cfg_comb = apply_value(cfg_comb, f, v, base_cfg)
            if sig(cfg_comb) not in cfg_seen and budget_left():
                _try(cfg_comb, {f: best_val[f] for f in best_val if best_val[f] != dflt[f]})
            tops = {f: v for f, v in field_top.items() if len(v) >= 2}
            n_combo = 0
            if len(tops) >= 2:
                fields = list(tops)
                for combo in itertools.product(*[tops[f] for f in fields]):
                    if n_combo >= int(self.S.combo_max) or not budget_left():
                        break
                    c = cfg_comb
                    for f, v in zip(fields, combo):
                        c = apply_value(c, f, v, base_cfg)
                    if sig(c) in cfg_seen:
                        continue
                    _try(c, dict(zip(fields, combo)))
                    n_combo += 1
            # phase 4: budget left after the combinations -> continue refining (round-robin)
            for f, v in _round_robin({f: [v for v in q if v not in results[f]] for f, q in rq.items()}):
                if not budget_left():
                    break
                results[f][v] = _try(apply_value(base_cfg, f, v, base_cfg), {f: v})
            _summarise()
            if not_tested:
                msg = "; ".join(f"{f}: {len(v)} value(s) not tested" for f, v in not_tested.items())
                log.warning(f"GATE  FLAG:SEARCH_BUDGET {stage}: budget ({max_evals} evals / {budget:.0f}s) reached -- {msg}")

        if not self.S.enabled:
            j, lab, extra, cfg_c, pdesc, m, _ = evals[0]
            self.accepted_labels, self.accepted_metrics, self.accepted_stage = lab.copy(), m, stage
            self._row(stage, m, "accepted_gate_off", params=pdesc)
            return lab, extra, cfg_c, True

        ranked = sorted(range(len(evals)), key=lambda i: evals[i][0]["score"], reverse=True)
        passing = [i for i in ranked if evals[i][0]["passed"]]
        changing = [i for i in passing if not evals[i][0].get("unchanged")]
        if passing:
            pick = passing[0]
            # tie-break only: an unchanged fallback loses to a changing candidate with an EQUAL score, never a lower one
            if (evals[pick][0].get("unchanged") and changing
                    and evals[changing[0]][0]["score"] >= evals[pick][0]["score"] - EPS):
                pick = changing[0]
            j, lab, extra, cfg_c, pdesc, m, _ = evals[pick]
            chosen = {f: _jsonable(getattr(cfg_c, f)) for f in params}
            bnd = []
            for f in params:
                spec = search_spec(f, self.S.search_overrides)
                # boundary = edge of the parameter's FULL search range (untested values from a budget cut are reported
                # separately as FLAG:SEARCH_BUDGET, not as a boundary optimum)
                if f in coarse and on_boundary(spec, _norm_val(getattr(cfg_c, f), spec["kind"]), coarse[f]):
                    bnd.append(f)
                    msg = (f"{stage}: {f}={getattr(cfg_c, f)} is at the edge of the search range "
                           f"[{min(coarse[f])}, {max(coarse[f])}] -- boundary optimum, range may be insufficient "
                           f"(widen it with gate_search_overrides)")
                    self.boundary_warnings.append(msg)
                    log.warning(f"GATE  FLAG:BOUNDARY_OPTIMUM {msg}")
            no_change = bool(j.get("unchanged"))
            rank = ranked.index(pick) + 1
            if no_change:
                detail = ("no assignment changed (the unchanged result scored best, or every changing candidate "
                          "failed -- see stage_quality_candidates.tsv)")
            elif j.get("split_summary"):
                detail = f"rank {rank}/{len(evals)}; {j['split_summary']} (see split_diagnostics.tsv)"
            else:
                detail = (f"rank {rank}/{len(evals)}; worst {j['worst_metric']} "
                          f"{(100 * j['worst']):+.2f}{' pp' if str(j['worst_metric']).startswith('knn') else ' %'}; "
                          f"mean {100 * j['collective']:+.1f} %")
            if bnd:
                detail += f"; boundary optimum: {','.join(bnd)}"
            wd = self._evals_wd[pick]
            if Path(wd) != self.outdir and Path(wd).exists():
                for f in Path(wd).iterdir():
                    dst = self.outdir / f.name
                    if f.is_file():
                        shutil.copyfile(f, dst)
                    elif f.is_dir():
                        shutil.rmtree(dst, ignore_errors=True)
                        shutil.copytree(f, dst)
            shutil.rmtree(work_root, ignore_errors=True)
            self.accepted_labels, self.accepted_metrics, self.accepted_stage = lab.copy(), m, stage
            self.chosen_params[stage] = {"values": chosen, "candidate": eval_cid[pick], "attempts": n_attempts[0],
                                         "successful_evaluations": len(evals), "boundary_optimum": bnd,
                                         "not_tested": {f: [_jsonable(x) for x in v] for f, v in not_tested.items()}}
            decision = "accepted_no_change" if no_change else "accepted"
            self._row(stage, m, decision, detail, chosen)
            log.info(f"GATE  {stage:<26} {decision.upper()} {chosen}  {self._fmt(m)}  ({detail}; {n_attempts[0]} attempt(s), "
                     f"{time.time() - t0:.0f}s)")
            return lab, extra, cfg_c, True
        shutil.rmtree(work_root, ignore_errors=True)
        try:
            (self.outdir / f"REPORTS_FROM_REJECTED_STAGE_{stage}.txt").write_text(
                f"Stage {stage} was REJECTED by the quality gate and its labels reverted. Any per-stage report files it "
                f"wrote describe a REJECTED candidate. All candidates are saved in gate_candidates/{stage}/.\n")
        except OSError:
            pass
        j0 = evals[ranked[0]][0]
        detail = (f"all {n_attempts[0]} candidate(s) failed; best: {j0.get('fail_reasons', '')} "
                  f"(tolerance {100 * self.S.tolerance:.0f} %, kNN max drop {self.S.knn_max_drop_pp:g} pp) -> "
                  f"reverted to '{self.accepted_stage}'")
        self._row(stage, ref_m, "rejected_reverted", detail)
        log.warning(f"GATE  {stage:<26} REJECT  {detail}")
        return ref_l.copy(), None, base_cfg, False

    def write(self):
        shutil.rmtree(self.outdir / "_gate_work", ignore_errors=True)
        try:
            import csv
            self.outdir.mkdir(parents=True, exist_ok=True)
            cols = ["run", "stage"] + METRIC_COLUMNS + ["decision", "detail", "params"]
            with open(self.outdir / "stage_quality.tsv", "w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=cols, delimiter="\t", extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)
            if self.cand_rows:
                raw = [f"knn@{k}_{s}" for k in KNN_KS for s in ("before", "after", "change_pp")]
                cols2 = (["run", "stage", "candidate", "params", "seconds"] + METRIC_COLUMNS + raw +
                         ["unchanged", "note", "passed", "score", "worst_metric", "worst_change", "collective_change",
                          "split_summary", "fail_reasons"])
                with open(self.outdir / "stage_quality_candidates.tsv", "w", newline="") as h:
                    w = csv.DictWriter(h, fieldnames=cols2, delimiter="\t", extrasaction="ignore", restval="")
                    w.writeheader()
                    w.writerows(self.cand_rows)
            if self.split_rows:
                cols3 = list(dict.fromkeys(k for r in self.split_rows for k in r))
                with open(self.outdir / "split_diagnostics.tsv", "w", newline="") as h:
                    w = csv.DictWriter(h, fieldnames=cols3, delimiter="\t", extrasaction="ignore", restval="")
                    w.writeheader()
                    w.writerows(self.split_rows)
            (self.outdir / "stage_quality_chosen_params.json").write_text(
                json.dumps({"tolerance": self.S.tolerance, "knn_max_drop_pp": self.S.knn_max_drop_pp,
                            "chosen": self.chosen_params, "boundary_warnings": self.boundary_warnings},
                           indent=2, default=_jsonable) + "\n")
        except OSError as e:
            log.warning(f"could not write stage_quality tables: {e}")

# ------------------------------------------------------------------------------------------------ calibration
def calibrate(table_tsv: str, stage_order: Optional[Sequence[str]] = None) -> Dict:
    """Read a stage-metrics table, print each transition's changes (relative for internal indices, percentage points
    for kNN agreement). `python -m hyphaesbin.clustering.quality_gate calibrate table.tsv`."""
    import pandas as pd
    d = pd.read_csv(table_tsv, sep="\t")
    d = d[~d["stage"].isin(NON_STAGE_NAMES)]
    if stage_order:
        d = d.set_index("stage").loc[[s for s in stage_order if s in set(d["stage"])]].reset_index()
    else:
        d = d.sort_values("stage", key=lambda s: s.str.replace("04d_after_binmerge", "04c~_after_binmerge"))
    rows = []
    for i in range(1, len(d)):
        a, b = d.iloc[i - 1].to_dict(), d.iloc[i].to_dict()
        ch = metric_changes(a, b)
        if not ch:
            continue
        rel = {m: v[3] for m, v in ch.items() if v[0] == "rel"}
        pp = {m: v[3] for m, v in ch.items() if v[0] == "pp"}
        wr = min(rel, key=rel.get) if rel else ""
        wp = min(pp, key=pp.get) if pp else ""
        rows.append({"stage": b["stage"], "worst_rel_metric": wr, "worst_rel_pct": round(100 * rel.get(wr, 0), 1),
                     "worst_knn_metric": wp, "worst_knn_pp": round(100 * pp.get(wp, 0), 2)})
    for r in rows:
        print(f"{r['stage']:<28} worst rel {r['worst_rel_metric']:<18} {r['worst_rel_pct']:+7.1f} %   "
              f"worst kNN {r['worst_knn_metric']:<14} {r['worst_knn_pp']:+6.2f} pp")
    return {"transitions": rows}


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "calibrate":
        calibrate(sys.argv[2])
    else:
        print("usage: python -m hyphaesbin.clustering.quality_gate calibrate <stage_metrics.tsv>")
