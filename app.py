# app.py
# One-file PPS Village & Household Sampler (Stratified 30/30 per kebele)
# v4: PPS at village level, then per-kebele fixed 30 Eligible + 30 Non-eligible,
#      proportionally split across PPS-sampled villages, then systematic HH selection.

from pathlib import Path
import io
import math
import hashlib
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st

# --------------------- (Optional) page icon assets ---------------------
ASSETS_LOGO = Path("assets/wfp_logo.png")
ASSETS_FAVICON = Path("assets/favicon.png")
PAGE_ICON = str(ASSETS_FAVICON) if ASSETS_FAVICON.exists() else (str(ASSETS_LOGO) if ASSETS_LOGO.exists() else None)
st.set_page_config(page_title="PPS → 30/30 per Kebele → Systematic HH", page_icon=PAGE_ICON, layout="wide")

# --------------------- Helpers ---------------------
def norm_cols(cols):
    return [str(c).strip().lower().replace("\n"," ").replace("\t"," ") for c in cols]

def read_single_file(file) -> pd.DataFrame:
    name = getattr(file, "name", "uploaded").lower()
    ext = name.split(".")[-1]
    try:
        if ext in ["xlsx","xls"]:
            df = pd.read_excel(file, engine=None)
        else:
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="cp1252")
    except Exception as e:
        raise ValueError(f"Could not read input file: {e}")

    df.columns = norm_cols(df.columns)

    # Map common names → standard names
    rename = {}
    for c in df.columns:
        if c in ["woreda"]: rename[c] = "woreda"
        if c in ["kebele"]: rename[c] = "kebele"
        if c in ["village","village / ea","ea","enumeration area"]: rename[c] = "village"
        if c in ["eligibility","eligible_flag","status"]: rename[c] = "eligibility"
        if c in ["household head name","hh head name","head name","hh_name","household_name"]: rename[c] = "head_name"
        if c in ["hh_id","hh id","household id","registration #","registration","id"]: rename[c] = "hh_id"
        if c in ["phone","phone (optional)","phone_number"]: rename[c] = "phone"
        if c in ["other id","other_id","alt id","alt_id"]: rename[c] = "other_id"
    df = df.rename(columns=rename)

    required = ["woreda","kebele","village","eligibility","head_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Expected at minimum: {required}")

    for c in ["woreda","kebele","village","eligibility","head_name"]:
        df[c] = df[c].astype(str).str.strip()

    # Normalize eligibility labels
    df["eligibility"] = (
        df["eligibility"]
        .str.strip().str.lower()
        .map({
            "eligible":"Eligible",
            "non-eligible":"Non-eligible",
            "non eligible":"Non-eligible",
            "noneligible":"Non-eligible",
            "ineligible":"Non-eligible",
            "e":"Eligible",
            "ne":"Non-eligible"
        }).fillna(df["eligibility"].str.title())
    )

    for col in ["hh_id","phone","other_id"]:
        if col not in df.columns:
            df[col] = ""

    return df[["woreda","kebele","village","eligibility","head_name","hh_id","phone","other_id"]]

def rng_for_group(seed_base, *keys):
    if not seed_base:
        return np.random.default_rng()
    s = "::".join([str(seed_base)] + [str(k) for k in keys])
    h = int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)
    return np.random.default_rng(h)

def _systematic_indices(N: int, n: int, rng: np.random.Generator):
    if N <= 0 or n <= 0:
        return [], None, None
    n_eff = min(int(n), int(N))
    interval = N / n_eff
    start = rng.uniform(0.0, interval)
    idxs = [int(np.floor(start + k*interval)) for k in range(n_eff)]
    seen, out = set(), []
    for i in idxs:
        j = min(max(i,0), N-1)
        if j not in seen:
            seen.add(j); out.append(j)
        if len(out) == n_eff:
            break
    return out, interval, start

# --------------------- PPS frame & PPS selection ---------------------
def build_pps_frame(roster: pd.DataFrame, pps_base: str, dedup_names: bool):
    r = roster.copy()
    if dedup_names:
        r = (r.sort_values(["woreda","kebele","village","head_name"])
               .drop_duplicates(["woreda","kebele","village","head_name"], keep="first"))
    if pps_base == "All households":
        grp = r.groupby(["woreda","kebele","village"], as_index=False).size().rename(columns={"size":"hhs"})
    elif pps_base == "Eligible-only":
        grp = r[r["eligibility"]=="Eligible"].groupby(["woreda","kebele","village"], as_index=False).size().rename(columns={"size":"hhs"})
    else:
        grp = r[r["eligibility"]=="Non-eligible"].groupby(["woreda","kebele","village"], as_index=False).size().rename(columns={"size":"hhs"})
    grp["hhs"] = grp["hhs"].astype(float)
    grp = grp[grp["hhs"]>0].copy()
    return grp

def pps_select_systematic(cum_high, rng, m):
    n = len(cum_high)
    if n == 0: return []
    interval = 1.0 / max(m,1)
    start = rng.random() * interval
    idxs = []
    for k in range(m):
        u = start + k*interval
        if u >= 1.0:
            u -= math.floor(u)
        i = int(np.searchsorted(cum_high, u, side="left"))
        if i >= n: i = n-1
        idxs.append(i)
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def sample_villages_pps(vdf: pd.DataFrame, method: str, m_default: int, threshold_n: int, m_large: int,
                        use_fixed_m: bool, fixed_m: int, seed_base: str):
    sampled_rows, diag_rows, kebele_summary = [], [], []
    for (w,k), g in vdf.groupby(["woreda","kebele"], sort=False):
        g = g.reset_index(drop=True)
        nvill = len(g)
        m = max(int(fixed_m if use_fixed_m else (m_large if nvill >= int(threshold_n) else m_default)), 1)
        g = g.copy()
        total = g["hhs"].sum()
        g["p"] = g["hhs"] / total if total>0 else 0
        g["cum_high"] = g["p"].cumsum(); g["cum_low"] = g["cum_high"] - g["p"]
        rng = rng_for_group(seed_base, w, k)
        if method == "Systematic":
            idxs = pps_select_systematic(g["cum_high"].to_numpy(), rng, m)
        else:
            draws = rng.random(m)
            idxs = []
            for u in draws:
                i = int(np.searchsorted(g["cum_high"].to_numpy(), u, side="left"))
                if i >= nvill: i = nvill-1
                if i not in idxs:
                    idxs.append(i)
            c=0
            while len(idxs) < min(m, nvill) and c < m*3:
                u = rng.random()
                i = int(np.searchsorted(g["cum_high"].to_numpy(), u, side="left"))
                if i >= nvill: i = nvill-1
                if i not in idxs: idxs.append(i)
                c+=1

        sel = g.iloc[idxs][["woreda","kebele","village","hhs"]]
        sampled_rows.append(sel)

        d = g.copy()
        d["method"] = method; d["m_requested"] = m; d["m_final"] = len(idxs); d["selected"] = False
        d.loc[d.index.isin(idxs), "selected"] = True
        diag_rows.append(d)

        kebele_summary.append({"Woreda":w, "Kebele":k, "#Villages":nvill, "Method":method,
                               "m_used":int(m), "Total HHs":int(total)})

    sampled = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else vdf.iloc[0:0]
    diagnostics = pd.concat(diag_rows, ignore_index=True) if diag_rows else vdf.iloc[0:0]
    summary = pd.DataFrame(kebele_summary)
    return sampled, diagnostics, summary

# --------------------- Proportionate allocation (per kebele, per group) ---------------------
def _proportionate_allocate_per_kebele(roster: pd.DataFrame, sv: pd.DataFrame,
                                       kebele_nE: int, kebele_nNE: int):
    """
    For each kebele and group (Eligible / Non-eligible):
      - Compute per-village capacity (count of HH in that group in that village).
      - Allocate kebele target across the PPS-sampled villages proportionally by capacity.
      - Cap at capacity; distribute remaining by largest fractional remainder while respecting remaining capacity.
    """
    sub = roster.copy()
    sub['eligibility'] = sub['eligibility'].astype(str)

    gcounts = (sub.groupby(['woreda','kebele','village','eligibility'], as_index=False)
                 .size().rename(columns={'size':'N'}))

    sv_keys = sv[['woreda','kebele','village']].drop_duplicates()
    gcounts = gcounts.merge(sv_keys, on=['woreda','kebele','village'], how='inner')

    out_quota_rows = []
    for (w,k), gk in gcounts.groupby(['woreda','kebele'], sort=False):
        for group, kebele_target in [('Eligible', int(kebele_nE)), ('Non-eligible', int(kebele_nNE))]:
            gkg = gk[gk['eligibility']==group].copy()
            # If no capacity rows for this group in any sampled village, assign zeros
            if gkg.empty or kebele_target <= 0:
                villages = sv_keys[(sv_keys['woreda']==w)&(sv_keys['kebele']==k)]['village'].unique().tolist()
                for v in villages:
                    out_quota_rows.append({'woreda':w,'kebele':k,'village':v,'group':group,
                                           'n_target':0,'capacity':0})
                continue

            gkg = gkg.rename(columns={'N':'capacity'})
            total_cap = int(gkg['capacity'].sum())
            quota_eff = min(int(kebele_target), total_cap)
            if total_cap == 0 or quota_eff == 0:
                for v in gkg['village'].unique():
                    out_quota_rows.append({'woreda':w,'kebele':k,'village':v,'group':group,
                                           'n_target':0,'capacity':int(gkg[gkg['village']==v]['capacity'].iloc[0])})
                continue

            gkg['raw'] = gkg['capacity'] * (quota_eff / total_cap)
            gkg['floor'] = np.floor(gkg['raw']).astype(int)
            gkg['rem'] = gkg['raw'] - gkg['floor']

            # Ensure floors never exceed capacity
            over = gkg['floor'] > gkg['capacity']
            if over.any():
                gkg.loc[over, 'floor'] = gkg.loc[over, 'capacity']

            assigned = int(gkg['floor'].sum())
            left = int(quota_eff - assigned)

            # Distribute leftover by largest fractional remainder, tie-broken by remaining capacity
            while left > 0:
                gkg['remaining_cap'] = gkg['capacity'] - gkg['floor']
                elig = gkg[gkg['remaining_cap'] > 0].copy()
                if elig.empty:
                    break
                elig = elig.sort_values(['rem','remaining_cap'], ascending=[False, False])
                for idx, row in elig.iterrows():
                    if left == 0: break
                    gkg.at[idx,'floor'] += 1
                    left -= 1

            for _, r in gkg.iterrows():
                out_quota_rows.append({'woreda': w,'kebele': k,'village': r['village'],
                                       'group': group,'n_target': int(r['floor']),
                                       'capacity': int(r['capacity'])})

            # Add zero rows for sampled villages with zero capacity in this group
            present_vs = set(gkg['village'].unique())
            all_vs = set(sv_keys[(sv_keys['woreda']==w)&(sv_keys['kebele']==k)]['village'].unique())
            for v in (all_vs - present_vs):
                out_quota_rows.append({'woreda':w,'kebele':k,'village':v,'group':group,
                                       'n_target':0,'capacity':0})

    quotas_df = pd.DataFrame(out_quota_rows)
    return quotas_df

# --------------------- HH Sampling (stratified, systematic) ---------------------
def to_excel_bytes(df1: pd.DataFrame, name1: str, df2: pd.DataFrame, name2: str) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df1.to_excel(w, index=False, sheet_name=name1)
        df2.to_excel(w, index=False, sheet_name=name2)
    return out.getvalue()

def sample_households_stratified_prop(roster: pd.DataFrame, sampled_villages: pd.DataFrame,
                                      kebele_nE: int, kebele_nNE: int, order_by: str, seed_base: str):
    sv = sampled_villages[["woreda","kebele","village"]].drop_duplicates().copy()
    quotas = _proportionate_allocate_per_kebele(roster, sv, kebele_nE, kebele_nNE)

    out_rows, sum_rows = [], []
    for (w,k,v), gq in quotas.groupby(['woreda','kebele','village']):
        sub = roster[(roster['woreda']==w) & (roster['kebele']==k) & (roster['village']==v)].copy()
        for _, qrow in gq.iterrows():
            group = qrow['group']; n_target = int(qrow['n_target'])
            gdf = sub[sub['eligibility']==group].copy(); N = len(gdf)

            if N==0 or n_target<=0:
                sum_rows.append({'Woreda':w,'Kebele':k,'Village':v,'Group':group,
                                 'N':int(N),'n':int(n_target),'Interval':None,'Start':None,'Picked':0})
                continue

            col = (order_by or "").strip().lower()
            if col and col in gdf.columns:
                gdf = gdf.sort_values(col, kind='mergesort')
            else:
                # Stable fallback: head_name → hh_id
                fallback_cols = [c for c in ["head_name","hh_id"] if c in gdf.columns]
                gdf = gdf.sort_values(fallback_cols, na_position='last', kind='mergesort')

            rng = rng_for_group(seed_base, w,k,v,group)
            idxs, interval, start = _systematic_indices(N, int(n_target), rng)

            pick = gdf.iloc[idxs].copy()
            pick.insert(0, 'Sample_Order', range(1, len(pick)+1))
            pick.insert(0, 'Eligibility', group)
            pick.insert(0, 'Village', v)
            pick.insert(0, 'Kebele', k)
            pick.insert(0, 'Woreda', w)

            out_rows.append(pick[['Woreda','Kebele','Village','Eligibility','Sample_Order',
                                  'hh_id','head_name','phone','other_id']])
            sum_rows.append({'Woreda':w,'Kebele':k,'Village':v,'Group':group,
                             'N':int(N),'n':int(n_target),
                             'Interval':round(interval,3) if interval else None,
                             'Start':round(float(start),3) if start is not None else None,
                             'Picked':len(pick)})

    hh_sample = (pd.concat(out_rows, ignore_index=True)
                 if out_rows else pd.DataFrame(columns=['Woreda','Kebele','Village','Eligibility',
                                                        'Sample_Order','hh_id','head_name','phone','other_id']))
    hh_summary = pd.DataFrame(sum_rows)

    quotas_pivot = (quotas.pivot_table(index=['woreda','kebele','village'], columns='group',
                                       values='n_target', fill_value=0).reset_index())
    quotas_pivot.columns.name = None
