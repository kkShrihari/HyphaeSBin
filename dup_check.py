#!/usr/bin/env python3
"""
dup_check.py -- where does a bin's contamination come from? (alleles/strains vs a second organism)

For each bin it reads BUSCO's DUPLICATED single-copy genes and, for every pair of copies:
  * protein identity between the copies (multi_copy_busco_sequences/<id>.faa)
        >= 98 %  -> allele / strain copy of the SAME species
        90-98 %  -> close strain or recent paralog
        < 90 %   -> a DIFFERENT organism (contamination) or an old paralog
  * GC and coverage (SPAdes _cov_ in the contig name) of the contigs carrying the copies, as robust z-scores vs the
    whole bin -> a copy on an outlier contig (|z| > 2.5) is likely foreign
Verdict per bin from the share of low-identity pairs and outlier contigs.

Usage (hyphaesbin311 env):
  python3 dup_check.py --bins-dir <clusters_deduplicated> --busco-root <folder with the BUSCO runs> --bins bin_005 bin_008
  e.g.
  python3 dup_check.py \
    --bins-dir   /home/skkumara/zostera_1000/reclust/final2/runs/tetnfcov/clustering/clusters_deduplicated \
    --busco-root /home/skkumara/zostera_1000/reclust/final2/runs/tetnfcov/clustering \
    --bins bin_005 bin_008
The BUSCO folder of a bin is found automatically: any full_table.tsv below --busco-root whose path contains the bin
name (e.g. .../busco_*/bin_005*/run_eukaryota_odb12/full_table.tsv).
Output: printed tables + <bin>_dup_check.tsv in --out (default: current folder).
"""
import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs


def gc(s):
    s = s.upper()
    n = sum(s.count(c) for c in "ACGT")
    return (s.count("G") + s.count("C")) / n if n else np.nan


def cov_of(cid):
    m = re.search(r"_cov_([0-9.]+)", cid)
    return float(m.group(1)) if m else np.nan


def robust_z(x, ref):
    med = np.nanmedian(ref)
    mad = np.nanmedian(np.abs(ref - med)) * 1.4826
    return (x - med) / mad if mad > 0 else 0.0


def identity(a, b):
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.open_gap_score, al.extend_gap_score = -10, -0.5
    try:
        from Bio.Align import substitution_matrices
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        pass
    aln = al.align(a, b)[0]
    s1, s2 = str(aln[0]), str(aln[1])
    cols = [(x, y) for x, y in zip(s1, s2) if x != "-" and y != "-"]
    return 100 * sum(x == y for x, y in cols) / max(len(cols), 1), len(cols) / max(min(len(a), len(b)), 1)


NODE_RE = re.compile(r"(NODE_\d+_length_\d+_cov_[0-9.]+(?:_[A-Za-z0-9]+)*?)(?=[:|\s]|$)")


def contig_from_header(h, known):
    """BUSCO/metaeuk headers vary (contig:start-end, contig|..., ...) -> map back to a bin contig id."""
    h = h.split()[0]
    for tok in re.split(r"[|]", h):
        c = re.sub(r":\d+-\d+$", "", tok)
        if c in known:
            return c
    m = NODE_RE.search(h)
    if m:
        core = m.group(1)
        if core in known:
            return core
        hit = next((k for k in known if k.startswith(core) or core.startswith(k)), None)
        if hit:
            return hit
    return h


def check_bin(binname, a):
    fa = next((p for p in Path(a.bins_dir).glob(f"{binname}.fa*")), None)
    if fa is None:
        print(f"[{binname}] bin FASTA not found in {a.bins_dir}")
        return
    ft = [p for p in Path(a.busco_root).rglob("full_table.tsv") if binname in str(p)]
    if not ft:
        print(f"[{binname}] no BUSCO full_table.tsv under {a.busco_root} with '{binname}' in the path")
        return
    ft = ft[0]
    seqs = read_fasta(fa)
    G = {c: gc(s) for c, s in seqs.items()}
    C = {c: cov_of(c) for c in seqs}
    gref, cref = np.array(list(G.values())), np.array([np.log1p(v) for v in C.values()])

    recs = []
    with open(ft) as fh:                     # rows have 2-10 fields (Missing rows are short) -> parse by hand
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) > 1 and f[1] == "Duplicated":
                recs.append(f[:3])
    t = pd.DataFrame(recs, columns=[0, 1, 2])
    mdir = ft.parent / "busco_sequences" / "multi_copy_busco_sequences"
    rows, hdr_shown = [], False
    for bid, g in t.groupby(0):
        faa = mdir / f"{bid}.faa"
        if not faa.exists():
            continue
        prot = read_fasta(faa)
        copies = list(prot.items())
        for (h1, p1), (h2, p2) in combinations(copies, 2):
            ident, covr = identity(p1, p2)
            c1, c2 = contig_from_header(h1, seqs), contig_from_header(h2, seqs)
            if not hdr_shown:
                print(f"  [header example] {h1}  ->  {c1}  ({'matched' if c1 in seqs else 'NOT matched'})")
                hdr_shown = True
            zs = []
            for c in (c1, c2):
                zg = robust_z(G.get(c, np.nan), gref)
                zc = robust_z(np.log1p(C.get(c, np.nan)), cref)
                zs.append((zg, zc))
            out1 = abs(zs[0][0]) > 2.5 or abs(zs[0][1]) > 2.5
            out2 = abs(zs[1][0]) > 2.5 or abs(zs[1][1]) > 2.5
            cls = ("allele/strain (same species)" if ident >= 98 else "close strain/recent paralog" if ident >= 90
                   else "different organism or old paralog")
            rows.append({"busco": bid, "identity_pct": round(ident, 1), "aligned_frac": round(covr, 2), "class": cls,
                         "contig1": c1, "gc1": round(G.get(c1, np.nan), 3), "cov1": C.get(c1), "z_gc1": round(zs[0][0], 1), "z_cov1": round(zs[0][1], 1),
                         "contig2": c2, "gc2": round(G.get(c2, np.nan), 3), "cov2": C.get(c2), "z_gc2": round(zs[1][0], 1), "z_cov2": round(zs[1][1], 1),
                         "outlier_copy": out1 or out2})
    D = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / f"{binname}_dup_check.tsv", sep="\t", index=False)

    print(f"\n================ {binname}  ({fa.name}, {len(seqs):,} contigs; BUSCO: {ft})")
    if D.empty:
        print("no duplicated BUSCOs with sequences -> nothing to check")
        return
    n = len(D)
    low = (D.identity_pct < 90).mean()
    mid = ((D.identity_pct >= 90) & (D.identity_pct < 98)).mean()
    hi = (D.identity_pct >= 98).mean()
    outl = D.outlier_copy.mean()
    print(f"duplicated BUSCO copy pairs: {n}")
    print(f"  identity >= 98 % (allele/strain)        : {100 * hi:.0f} %")
    print(f"  identity 90-98 % (close strain/paralog)  : {100 * mid:.0f} %")
    print(f"  identity < 90 % (other organism/paralog) : {100 * low:.0f} %")
    print(f"  pairs with a copy on a GC/coverage outlier contig: {100 * outl:.0f} %")
    print(f"  median identity: {D.identity_pct.median():.1f} %")
    ok = D.gc1.notna() & D.gc2.notna()
    print(f"  pairs with both contigs matched to the bin: {ok.sum()}/{n}")
    if ok.any():
        dgc = (D.gc1 - D.gc2).abs()[ok]
        print(f"  median |GC difference| between the two copies' contigs: {100 * dgc.median():.1f} points "
              f"(bin GC median {100 * np.nanmedian(gref):.1f} %, IQR {100 * (np.nanpercentile(gref, 75) - np.nanpercentile(gref, 25)):.1f})")
    if low >= 0.5 and outl >= 0.3:
        v = "SECOND ORGANISM (chimera/contamination): most copies differ strongly and sit on deviating contigs"
    elif low >= 0.5:
        v = "LIKELY SECOND ORGANISM or old paralogs: copies differ strongly, but contigs look similar (related genome?)"
    elif hi >= 0.5:
        v = "ALLELES / STRAIN HETEROGENEITY of one species: copies near-identical -> not a chimera"
    else:
        v = "MIXED: close strains and/or paralogs; check the table"
    print(f"VERDICT: {v}")
    pd.set_option("display.width", 220)
    print(D.sort_values("identity_pct")[["busco", "identity_pct", "class", "gc1", "gc2", "cov1", "cov2",
                                          "outlier_copy"]].head(15).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins-dir", required=True)
    ap.add_argument("--busco-root", required=True)
    ap.add_argument("--bins", nargs="+", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    for b in a.bins:
        check_bin(b, a)


if __name__ == "__main__":
    main()
