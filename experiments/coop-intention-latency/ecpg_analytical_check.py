"""ECPG analytical consistency check (Path A -- no CARLA re-run).

The stored eval data only has episode-level aggregates, NOT per-frame geometry,
so the TRUE trajectory-replay ECPG cannot be computed from it. Instead this
script instantiates the ECPG *concept* analytically per configuration cell
(rule, num, latency k) with a small set of a-priori, scenario-reasoned constants
(NOT fit to the success matrix), then checks whether that single scalar tracks
the observed success/collision across the 44 cells.

ECPG(rule, k) = base_relevance[rule] x latency_survival(k, redundancy[rule])

  - base_relevance:  relevance x novelty of the shared set when fresh (k=0).
                     Set from scenario reasoning: how much of the decision-
                     critical oncoming-vehicle set the rule covers.
  - latency_survival(k, m): per-collaborator plan-coherence decay s(k)=1-k/kc,
                     aggregated over m (effective independent collaborators):
                       s>=0 : 1-(1-s)^m   (redundant coverage -> latency-robust)
                       s<0  : s/m         (stale plan misleads; redundancy dilutes harm)
                     kc = plan-coherence horizon; s<0 => info is actively misleading.

Reported: Spearman rank correlation (monotonic link; success saturates so Pearson
is inappropriate) between analytical ECPG and observed success / collision, plus
a scatter. This is a concept-SHAPE consistency check, NOT a from-trajectory
measurement; it cannot capture the random-vs-nearest temporal-stability effect
(Finding 3), which needs per-frame geometry (Path B).
"""

import csv
import json
import os

# ---- a-priori constants (scenario reasoning, not fit to success) ----
BASE_RELEVANCE = {  # fresh (k=0) coverage of the decision-critical oncoming set
    "all": 1.00,       # every oncoming vehicle's intention available
    "nearest2": 0.95,  # the 2 nearest = essentially the immediate blockers
    "nearest1": 0.72,  # 1 of the ~1-2 critical blockers
    "random1": 0.55,   # a random vehicle is the critical blocker only ~half the time
}
REDUNDANCY = {"all": 3, "nearest2": 2, "nearest1": 1, "random1": 1}  # effective independent collaborators
KC = 7.0        # plan-coherence horizon in steps (~0.7s @10Hz); past this a stale plan misleads
S_MIN = -0.6    # clip on per-collaborator survival (max misleading harm)


def latency_survival(k, m):
    s = max(S_MIN, 1.0 - k / KC)
    if s >= 0:
        return 1.0 - (1.0 - s) ** m   # union coverage: redundancy -> latency-robust
    return s / m                       # stale plan misleads; more collaborators dilute harm


def ecpg(rule, k):
    if rule == "none":
        return 0.0
    return BASE_RELEVANCE[rule] * latency_survival(k, REDUNDANCY[rule])


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def load_observed(outdir):
    rows = {}
    for d in sorted(os.listdir(outdir)):
        p = os.path.join(outdir, d, "metrics.jsonl")
        if not (os.path.isfile(p) and os.path.exists(os.path.join(outdir, d, "DONE"))):
            continue
        L = [json.loads(l) for l in open(p)]
        e = sum(1 for x in L if "episode/score" in x)
        if e == 0:
            continue
        rows[d] = dict(
            succ=sum(x.get("stats/sum_destination_reached", 0) for x in L) / e,
            coll=sum(x.get("stats/sum_is_collision", 0) for x in L) / e,
        )
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="logdir/eval/latency")
    ap.add_argument("--save", default="experiments/coop-intention-latency")
    args = ap.parse_args()

    obs = load_observed(args.outdir)
    recs = []
    for rule in ["all", "nearest2", "nearest1", "random1"]:
        for k in range(11):
            tag = f"{rule}_k{k}"
            if tag not in obs:
                continue
            recs.append(dict(tag=tag, rule=rule, k=k, ecpg=ecpg(rule, k),
                             succ=obs[tag]["succ"], coll=obs[tag]["coll"]))
    # k=0 code-path caveat: exclude the seeded k=0 cells from the primary
    # correlation (they used the v1 code path). Report both with/without.
    main_recs = [r for r in recs if r["k"] >= 1]

    def report(rs, label):
        E = [r["ecpg"] for r in rs]
        S = [r["succ"] for r in rs]
        C = [r["coll"] for r in rs]
        print(f"[{label}] n={len(rs)}  "
              f"Spearman(ECPG, success)={spearman(E, S):+.3f}   "
              f"Spearman(ECPG, collision)={spearman(E, C):+.3f}")

    print("=== analytical ECPG vs observed (Spearman rank correlation) ===")
    report(recs, "all cells incl k=0")
    report(main_recs, "k>=1 only (drops v1-code-path k=0)")

    csv_path = os.path.join(args.save, "ecpg_analytical.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "rule", "k", "ecpg", "succ", "coll"])
        w.writeheader()
        for r in sorted(recs, key=lambda r: (-r["ecpg"])):
            w.writerow({**r, "ecpg": round(r["ecpg"], 3), "succ": round(r["succ"], 3), "coll": round(r["coll"], 3)})
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable, skipping scatter")
        return
    colors = {"all": "tab:blue", "nearest2": "tab:green", "nearest1": "tab:orange", "random1": "tab:red"}
    for metric, fname, ylab in [("succ", "ecpg_vs_success.png", "observed success rate"),
                                ("coll", "ecpg_vs_collision.png", "observed collision rate")]:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for rule in colors:
            rs = [r for r in recs if r["rule"] == rule]
            ax.scatter([r["ecpg"] for r in rs], [r[metric] for r in rs],
                       c=colors[rule], label=rule, s=45, alpha=0.8)
            for r in rs:
                ax.annotate(f"k{r['k']}", (r["ecpg"], r[metric]), fontsize=6, alpha=0.6,
                            xytext=(2, 2), textcoords="offset points")
        rho = spearman([r["ecpg"] for r in main_recs], [r[metric] for r in main_recs])
        ax.set_xlabel("analytical ECPG (a-priori constants)")
        ax.set_ylabel(ylab)
        ax.set_title(f"analytical ECPG vs {ylab}\nSpearman rho (k>=1) = {rho:+.3f}")
        ax.legend(title="rule", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(args.save, "figures", fname)
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
