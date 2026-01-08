from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


# =========================
# 0) Logging
# =========================

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("electre_f1")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =========================
# 1) Utils
# =========================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def minmax_normalize(series: pd.Series) -> pd.Series:
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - smin) / (smax - smin)


def safe_parse_date(x) -> pd.Timestamp:
    return pd.to_datetime(x, dayfirst=True, errors="coerce")


def read_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
                break
            except Exception:
                continue
        else:
            raise

    df.columns = [c.strip() for c in df.columns]

    for col in ["latitude", "longitude", "temp_avg_c", "precip_mm_month", "d21", "d22"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = df["date"].apply(safe_parse_date)

    required = ["gp_name", "region", "date", "latitude", "longitude", "d21", "d22", "temp_avg_c", "precip_mm_month"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    return df.reset_index(drop=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# 2) ELECTRE IS params
# =========================

@dataclass(frozen=True)
class CriterionParams:
    w: float
    q: float
    p: float
    v: float


@dataclass(frozen=True)
class ElectreParams:
    s: float
    crit: Dict[str, CriterionParams]


# =========================
# 3) Step matrix construction (Option B)
# =========================

def compute_step_raw(
    df: pd.DataFrame,
    current_idx: int,
    candidate_indices: List[int],
    alpha: float = 0.15,
    v_eff_km_per_day: float = 850.0,
    handling_days: float = 2.0,
) -> pd.DataFrame:
    c = df.loc[current_idx]
    rows = []

    for idx in candidate_indices:
        a = df.loc[idx]

        d11 = haversine_km(c["latitude"], c["longitude"], a["latitude"], a["longitude"])
        d12 = alpha * d11
        d13 = 1.0 if (str(a["region"]) != str(c["region"])) else 0.0

        d32 = float((a["date"] - c["date"]).days) if pd.notna(a["date"]) and pd.notna(c["date"]) else float("nan")
        d14 = d32 - (d11 / v_eff_km_per_day + handling_days) if not math.isnan(d32) else float("nan")

        d21 = float(a["d21"])
        d22 = float(a["d22"])

        temp = float(a["temp_avg_c"])
        precip = float(a["precip_mm_month"])
        s_temp = max(0.0, 1.0 - abs(temp - 23.0) / 10.0)
        s_rain = max(0.0, 1.0 - precip / 200.0)
        d31 = 10.0 * (0.6 * s_temp + 0.4 * s_rain)

        rows.append({
            "idx": idx,
            "gp_name": a["gp_name"],
            "region": a["region"],
            "date": a["date"],
            "d11": d11,
            "d12": d12,
            "d13": d13,
            "d14": d14,
            "d21": d21,
            "d22": d22,
            "d31": d31,
            "d32": d32,
        })

    return pd.DataFrame(rows).set_index("idx")


def normalize_step(raw: pd.DataFrame) -> pd.DataFrame:
    n = raw.copy()

    # Costs -> invert
    for col in ["d11", "d12", "d13"]:
        n[col + "n"] = 1.0 - minmax_normalize(n[col])

    # Benefits
    for col in ["d14", "d21", "d22", "d31", "d32"]:
        n[col + "n"] = minmax_normalize(n[col])

    return n


# =========================
# 4) ELECTRE IS core
# =========================

def partial_concordance(delta: float, q: float, p: float) -> float:
    if math.isnan(delta):
        return 0.0
    if delta <= q:
        return 1.0
    if delta >= p:
        return 0.0
    return (p - delta) / (p - q) if p > q else 0.0


def electre_outranking(
    norm: pd.DataFrame,
    params: ElectreParams,
    debug_pairs: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crit_keys = ["d11", "d12", "d13", "d14", "d21", "d22", "d31", "d32"]
    idxs = norm.index.tolist()

    C = pd.DataFrame(0.0, index=idxs, columns=idxs)
    S = pd.DataFrame(0, index=idxs, columns=idxs, dtype=int)
    VETO = pd.DataFrame(0, index=idxs, columns=idxs, dtype=int)

    for a in idxs:
        for b in idxs:
            if a == b:
                continue

            csum = 0.0
            vetoed = False
            veto_reasons = []

            for k in crit_keys:
                cp = params.crit[k]
                ga = float(norm.loc[a, k + "n"])
                gb = float(norm.loc[b, k + "n"])
                delta = gb - ga

                cj = partial_concordance(delta, cp.q, cp.p)
                csum += cp.w * cj

                if not math.isnan(delta) and delta >= cp.v:
                    vetoed = True
                    veto_reasons.append(k)

            C.loc[a, b] = csum
            if vetoed:
                VETO.loc[a, b] = 1

            if (csum >= params.s) and (not vetoed):
                S.loc[a, b] = 1

            if logger and debug_pairs:
                logger.debug(
                    f"a={a} b={b} | C={csum:.3f} | veto={vetoed} "
                    f"{'(reasons: '+','.join(veto_reasons)+')' if veto_reasons else ''}"
                )

    return C, S, VETO


def non_dominated_set(S: pd.DataFrame) -> List[int]:
    idxs = list(S.index)
    indeg = {i: int(S[i].sum()) for i in idxs}
    return [i for i in idxs if indeg[i] == 0]


def tie_break_choice(step_raw: pd.DataFrame, candidates: List[int]) -> int:
    sub = step_raw.loc[candidates].copy()
    sub = sub.sort_values(by=["d14", "d22"], ascending=[False, False])
    return int(sub.index[0])


# =========================
# 5) Main loop (logs + CSV exports, NO graph)
# =========================

def electre_build_calendar_with_logs(
    df: pd.DataFrame,
    start_gp_name: str,
    params: ElectreParams,
    alpha: float = 0.15,
    out_root: str = "outputs",
    logger: Optional[logging.Logger] = None,
    debug_pairs: bool = False,
    top_k_preview: int = 8,
) -> Dict[str, Any]:
    logger = logger or setup_logger("INFO")
    ensure_dir(out_root)

    start_matches = df.index[df["gp_name"] == start_gp_name].tolist()
    if not start_matches:
        raise ValueError(f"Start GP name not found: {start_gp_name}")
    current_idx = int(start_matches[0])

    remaining = set(int(i) for i in df.index)
    remaining.remove(current_idx)

    sequence = [start_gp_name]

    iter_no = 1
    while remaining:
        c = df.loc[current_idx]
        cand_idx = sorted(list(remaining))

        logger.info("=" * 80)
        logger.info(f"Iteration {iter_no:02d} | current = {c['gp_name']} | candidates = {len(cand_idx)}")
        logger.info(f"Current date={c['date'].date()} region={c['region']} lat/lon=({c['latitude']},{c['longitude']})")

        # Build + normalize
        step_raw = compute_step_raw(df, current_idx, cand_idx, alpha=alpha)
        step_norm = normalize_step(step_raw)

        # ELECTRE
        C, S, VETO = electre_outranking(step_norm, params, debug_pairs=debug_pairs, logger=logger)

        # Preview
        preview_cols = ["gp_name", "region", "date", "d11", "d12", "d13", "d32", "d14", "d21", "d22", "d31"]
        preview = step_raw[preview_cols].sort_values(by=["d14", "d22"], ascending=[False, False]).head(top_k_preview)
        logger.info("Top candidats (tri d14 puis d22) :\n" + preview.to_string())

        arcs = int(S.values.sum())
        veto_count = int(VETO.values.sum())
        n = len(S.index)
        density = arcs / (n * (n - 1)) if n > 1 else 0.0
        logger.info(f"Surclassement: arcs={arcs} | densité={density:.3f} | veto déclenchés={veto_count}")

        # Recommended set
        rec_set = non_dominated_set(S)
        rec_names = step_raw.loc[rec_set, "gp_name"].tolist() if rec_set else []
        logger.info(f"Ensemble recommandé (non-surclassés) = {rec_names}")

        if not rec_set:
            logger.warning("Aucun non-surclassé (cycle). Départage par score net.")
            outdeg = S.sum(axis=1)
            indeg = S.sum(axis=0)
            net = (outdeg - indeg).sort_values(ascending=False)
            rec_set = [int(net.index[0])]
            rec_names = [step_raw.loc[rec_set[0], "gp_name"]]

        # Choose
        if len(rec_set) == 1:
            chosen_idx = rec_set[0]
            reason = "unique non-surclassé"
        else:
            chosen_idx = tie_break_choice(step_raw, rec_set)
            reason = "départage (max d14 puis d22)"

        chosen_name = str(step_raw.loc[chosen_idx, "gp_name"])
        logger.info(f"Choix c_{iter_no+1} = {chosen_name} ({reason})")

        # Export iteration artifacts
        iter_dir = os.path.join(out_root, f"iter_{iter_no:02d}")
        ensure_dir(iter_dir)
        step_raw.to_csv(os.path.join(iter_dir, "step_raw.csv"))
        step_norm.to_csv(os.path.join(iter_dir, "step_norm.csv"))
        C.to_csv(os.path.join(iter_dir, "C.csv"))
        S.to_csv(os.path.join(iter_dir, "S.csv"))
        VETO.to_csv(os.path.join(iter_dir, "VETO.csv"))

        # Update for next iteration
        sequence.append(chosen_name)
        remaining.remove(chosen_idx)
        current_idx = chosen_idx
        iter_no += 1

    # Save final sequence
    seq_path = os.path.join(out_root, "calendar_sequence.txt")
    with open(seq_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(sequence, 1):
            f.write(f"{i}\t{name}\n")

    logger.info("=" * 80)
    logger.info(f"Calendrier construit sauvegardé: {seq_path}")

    return {"sequence": sequence, "outputs_dir": out_root}


# =========================
# 6) Example configuration
# =========================

def default_params() -> ElectreParams:
    crit = {
        "d11": CriterionParams(w=0.12, q=0.05, p=0.15, v=0.45),
        "d12": CriterionParams(w=0.10, q=0.05, p=0.15, v=0.50),
        "d13": CriterionParams(w=0.08, q=0.00, p=0.50, v=0.90),
        "d14": CriterionParams(w=0.15, q=0.05, p=0.15, v=0.30),
        "d21": CriterionParams(w=0.12, q=0.05, p=0.15, v=0.60),
        "d22": CriterionParams(w=0.18, q=0.05, p=0.15, v=0.60),
        "d31": CriterionParams(w=0.15, q=0.05, p=0.15, v=0.45),
        "d32": CriterionParams(w=0.10, q=0.05, p=0.15, v=0.45),
    }
    return ElectreParams(s=0.65, crit=crit)


if __name__ == "__main__":
    logger = setup_logger(level="INFO")

    path = "f1_2025_dataset_raw.csv"
    df = read_dataset(path)

    params = default_params()

    result = electre_build_calendar_with_logs(
        df=df,
        start_gp_name="Australian Grand Prix",
        params=params,
        alpha=0.15,
        out_root="outputs",
        logger=logger,
        debug_pairs=False,
        top_k_preview=8,
    )

    logger.info("\nCalendrier (ordre final):")
    for i, gp in enumerate(result["sequence"], 1):
        logger.info(f"{i:02d}. {gp}")