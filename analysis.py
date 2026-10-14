#!/usr/bin/env python3
"""
Reproduces every table and in-text number in
"A Three-Child Case Study on Classifier Use in Heritage and Monolingual Mandarin"
from the two token files only:

    data/children_tokens.csv    (Avia, Luna, XueXue)
    data/mothers_tokens.csv     (Avia's, Luna's and XueXue's mothers)
    data/nouns_in_paper.csv     (the nouns behind two counts in 3.3)

Figures 1 and 2 are made by figures.py from the same files.

Usage:   python analysis.py            (run from the repository root)
Output:  output/table1.csv, output/table2.csv, output/numbers.txt

Definitions
  个 share         = tokens with classifier_type == 'general' / all classifier tokens
                     (misselected specific classifiers count as specific)
  overt noun       = noun_status == 'overt'; noun omitted = noun_status == 'omitted'
  shared window    = 2;0-3;11 (24-47 months)
  Intervals        = children: 95% percentile interval from 5,000 bootstrap resamples of whole sessions;
                     mothers: exact (Clopper-Pearson) binomial interval.
"""
import io, os, re, json, zlib, collections
from math import comb
import numpy as np, pandas as pd
from scipy.stats import beta

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)
SEED, NBOOT, NPERM = 2026, 5000, 10000
WIN = (24, 47)
CHILDREN = ["Avia", "Luna", "XueXue"]
CONS = ["DEM+CL", "NUM+CL", "CL+N", "DEM+NUM+CL", "Other"]


# Bootstrap and permutation procedures use pseudo-random numbers. random_states.json stores the state of the random
# number generator (numpy PCG64, seed 2026) at the start of each procedure in the run that produced the reported
# values, so the printed digits reproduce exactly. A procedure without a stored state falls back to a fixed seed.
with open(os.path.join(ROOT, "random_states.json")) as _f:
    STATES = json.load(_f)


def rng_for(label):
    if label in STATES:
        g = np.random.Generator(np.random.PCG64())
        g.bit_generator.state = STATES[label]
        return g
    return np.random.default_rng([SEED, zlib.crc32(label.encode())])


# --------------------------------------------------------------------------- data
def load(path, mothers):
    d = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    d["gen"] = (d.classifier_type == "general").astype(int)
    d["spec"] = 1 - d.gen
    d["mis"] = (d.misselection == "yes").astype(int)
    d["overt"] = (d.noun_status == "overt").astype(int)
    d["year"] = d.age_months // 12
    d["month"] = d.age_months
    d["child"] = d.speaker.str.replace("'s mother", "", regex=False)
    d["role"] = "mother" if mothers else "child"
    return d


kids = load(os.path.join(DATA, "children_tokens.csv"), False)
moms = load(os.path.join(DATA, "mothers_tokens.csv"), True)
D = pd.concat([kids, moms], ignore_index=True)

SPEAKERS = [("Avia", "child"), ("Luna", "child"), ("XueXue", "child"),
            ("Avia", "mother"), ("Luna", "mother"), ("XueXue", "mother")]
LAB = {("Avia", "child"): "Avia", ("Luna", "child"): "Luna", ("XueXue", "child"): "XueXue",
       ("Avia", "mother"): "Avia’s mother", ("Luna", "mother"): "Luna’s mother", ("XueXue", "mother"): "XueXue’s mother"}

LOG = io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=LOG)


def fmt_age(m):
    return f"{int(m) // 12};{int(m) % 12}"


# --------------------------------------------------------------------------- intervals
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def clustered(sub, rng, B=NBOOT):
    g = sub.groupby("session").gen.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (B, len(g)))
    v = 100 * s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def share(sub, role, label):
    """'k/n, p% [lo, hi]' with the interval appropriate to the speaker."""
    n, k = len(sub), int(sub.gen.sum())
    if n == 0:
        return "–"
    lo, hi = clustered(sub, rng_for(label)) if role == "child" else cp(k, n)
    return f"{k}/{n}, {100 * k / n:.1f}% [{lo:.0f}, {hi:.0f}]"


def share_plain(sub):
    return f"{int(sub.gen.sum())}/{len(sub)}, {100 * sub.gen.mean():.1f}%" if len(sub) else "–"


def npct(k, n):
    return f"{k} ({100 * k / n:.1f}%)"


def spec_tokens(s):
    """specific-classifier tokens that were correctly selected (misselections excluded from diversity counts)"""
    return s[(s.spec == 1) & (s.mis == 0)].classifier


def expected_types(counts, m):
    """expected number of distinct types in a random draw of m tokens without replacement (Hurlbert, 1971)"""
    N = sum(counts)
    return sum(1 - comb(N - n_i, m) / comb(N, m) for n_i in counts)


# --------------------------------------------------------------------------- Table 1
shared = {c: kids[(kids.child == c) & kids.month.between(*WIN)] for c in CHILDREN}
M_RARE = min(len(spec_tokens(shared[c])) for c in CHILDREN)          # 13 (Luna)
rows = []


def header(t):
    rows.append([t] + [""] * 6)


def add(measure, fn, kids_only=False):
    r = [measure]
    for c, role in SPEAKERS:
        s = D[(D.child == c) & (D.role == role)]
        r.append("–" if (kids_only and role == "mother") else fn(s, role, f"{measure}|{c}|{role}"))
    rows.append(r)


header("A. Sample")
add("Recording sessions", lambda s, r, l: str(s.session.nunique()))
add("Ages", lambda s, r, l: f"{fmt_age(s.month.min())}–{fmt_age(s.month.max())}")
add("Classifier tokens (N)", lambda s, r, l: str(len(s)))

header("B. 个 share by age: 个 tokens/all tokens, % [95% interval]")
add("Shared window, 2;0–3;11", lambda s, r, l: share(s[s.month.between(*WIN)], r, l) if r == "child" else "–")
for y, lab in [(1, "Age 1 (1;8–1;11)"), (2, "Age 2 (2;0–2;11)"), (3, "Age 3 (3;0–3;11)"), (4, "Age 4 (4;0–4;11)")]:
    add(lab, lambda s, r, l, y=y: share(s[s.year == y], r, l))
add("All recorded months (mothers: all sampled sessions)", lambda s, r, l: share(s, r, l))

header("C. 个 share by noun status: 个 tokens/tokens, % [95% interval]")
add("Overt noun, all recorded months", lambda s, r, l: share(s[s.overt == 1], r, l))
add("Overt noun, shared window", lambda s, r, l: share(s[(s.overt == 1) & s.month.between(*WIN)], r, l) if r == "child" else "–")

header("D. Specific classifiers (misselections excluded)")
add("Tokens, shared window", lambda s, r, l: str(len(spec_tokens(s[s.month.between(*WIN)]))), kids_only=True)
add("Types, shared window", lambda s, r, l: str(spec_tokens(s[s.month.between(*WIN)]).nunique()), kids_only=True)
add(f"Types, rarefied to {M_RARE} tokens",
    lambda s, r, l: f"{expected_types(spec_tokens(s[s.month.between(*WIN)]).value_counts().tolist(), M_RARE):.1f}", kids_only=True)
add("Tokens, all months", lambda s, r, l: str(len(spec_tokens(s))))
add("Types, all months", lambda s, r, l: str(spec_tokens(s).nunique()))

header("E. Tokens by construction, n (% of the speaker's tokens)")
for c in CONS:
    lab = "Other (fillers, quantifiers, unclassified)" if c == "Other" else c
    add(lab, lambda s, r, l, c=c: npct(int((s.construction == c).sum()), len(s)))

header("F. 个 share within construction: 个 tokens/tokens in the construction, %")
for c in ["DEM+CL", "NUM+CL"]:
    add(c, lambda s, r, l, c=c: share_plain(s[s.construction == c]))

T1 = pd.DataFrame(rows, columns=["Measure"] + [LAB[s] for s in SPEAKERS])
T1.to_csv(os.path.join(OUT, "table1.csv"), index=False, encoding="utf-8-sig")
with pd.option_context("display.width", 300, "display.max_colwidth", 48, "display.max_rows", 100):
    say("=== TABLE 1 ===")
    say(T1.to_string(index=False))


# --------------------------------------------------------------------------- Table 2 (shared window)
def sess_table(c, mask):
    s = shared[c]
    s = s[mask(s)]
    g = s.groupby("session").agg(year=("year", "first"), k=("gen", "sum"), n=("gen", "size"))
    return g[g.n > 0].reset_index(drop=True)


def pooled(g):
    return 100 * g.k.sum() / g.n.sum()


def contrast(a, b, mask, rng, B=NBOOT, P=NPERM):
    ga, gb = sess_table(a, mask), sess_table(b, mask)
    obs = pooled(ga) - pooled(gb)
    bs = np.empty(B)
    for i in range(B):
        ra = ga.iloc[rng.integers(0, len(ga), len(ga))]
        rb = gb.iloc[rng.integers(0, len(gb), len(gb))]
        bs[i] = pooled(ra) - pooled(rb)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    both = pd.concat([ga.assign(g=0), gb.assign(g=1)], ignore_index=True)
    cnt = 0
    for _ in range(P):                                    # permute child labels across sessions within year of age
        lab = both.g.values.copy()
        for y in both.year.unique():
            ix = np.where(both.year.values == y)[0]
            lab[ix] = rng.permutation(lab[ix])
        t = pooled(both[lab == 0]) - pooled(both[lab == 1])
        cnt += abs(t) >= abs(obs) - 1e-12
    return obs, lo, hi, (cnt + 1) / (P + 1), len(ga), len(gb)


def holm(p):
    o = np.argsort(p)
    m = len(p)
    adj, run = np.empty(m), 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1, run)
    return adj


def fmt_p(p):
    """APA style: no leading zero; three decimals below .10, two otherwise"""
    if p < .001:
        return "<.001"
    return (f"{p:.3f}" if p < .1 else f"{p:.2f}").lstrip("0")


def signed(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.1f}"


PAIRS = [("Avia", "XueXue"), ("Luna", "XueXue"), ("Avia", "Luna")]
STAT = {}
for label, mask in [("overt", lambda s: s.overt == 1), ("num", lambda s: s.construction == "NUM+CL")]:
    res = [contrast(a, b, mask, rng_for(f"stat|{label}|{a}|{b}")) for a, b in PAIRS]
    STAT[label] = (res, holm([r[3] for r in res]))
t2 = []
for i, (a, b) in enumerate(PAIRS):
    ro, po = STAT["overt"][0][i], STAT["overt"][1][i]
    rn, pn = STAT["num"][0][i], STAT["num"][1][i]
    t2.append([f"{a} – {b}", f"{ro[4]}/{ro[5]}",
               f"{signed(ro[0])} [{signed(ro[1])}, {signed(ro[2])}]", fmt_p(po),
               f"{signed(rn[0])} [{signed(rn[1])}, {signed(rn[2])}]", fmt_p(pn)])
T2 = pd.DataFrame(t2, columns=["Contrast", "Sessions", "Overt-noun: difference, points [95% interval]", "Overt-noun: Holm p",
                               "NUM+CL: difference, points [95% interval]", "NUM+CL: Holm p"])
T2.to_csv(os.path.join(OUT, "table2.csv"), index=False, encoding="utf-8-sig")
say("\n=== TABLE 2 (shared window 2;0–3;11; sessions = sessions with overt-noun tokens, in the order of the contrast) ===")
say(T2.to_string(index=False))
say("(raw permutation p, unadjusted, overt noun:", [round(r[3], 4) for r in STAT["overt"][0]],
    "; NUM+CL:", [round(r[3], 4) for r in STAT["num"][0]], ")")

# --------------------------------------------------------------------------- numbers quoted in the text
say("\n=== 2.2 Identifying classifier tokens ===")
# Upstream filtering happened before the token files were created (corpus search, de-duplication), so these
# counts cannot be derived from the token files; they are recorded here as reported in the manuscript.
UPSTREAM = {"candidates_heritage": 2244, "duplicates_removed_heritage": 444,
            "candidates_xuexue": 1045, "duplicates_removed_xuexue": 170}
n_her = int(kids.child.isin(["Avia", "Luna"]).sum())
n_xx = int((kids.child == "XueXue").sum())
say(f"heritage children: {UPSTREAM['candidates_heritage']} candidates - {UPSTREAM['duplicates_removed_heritage']} duplicates = "
    f"{UPSTREAM['candidates_heritage'] - UPSTREAM['duplicates_removed_heritage']} (token file: {n_her})")
say(f"XueXue: {UPSTREAM['candidates_xuexue']} - {UPSTREAM['duplicates_removed_xuexue']} = "
    f"{UPSTREAM['candidates_xuexue'] - UPSTREAM['duplicates_removed_xuexue']} (token file: {n_xx})")
assert n_her == UPSTREAM["candidates_heritage"] - UPSTREAM["duplicates_removed_heritage"]
assert n_xx == UPSTREAM["candidates_xuexue"] - UPSTREAM["duplicates_removed_xuexue"]
say("final dataset:", len(kids), {c: int((kids.child == c).sum()) for c in CHILDREN})
say("mother tokens:", len(moms), {c: int((moms.child == c).sum()) for c in CHILDREN})

say("\n=== 2.6 Maternal sessions (number, and gaps in months between sampled sessions) ===")
for c in CHILDREN:
    ms = sorted(moms[moms.child == c].groupby("session").month.first())
    say(LAB[(c, "mother")], len(ms), "sessions at", [fmt_age(m) for m in ms], "gaps:", np.diff(ms).tolist())

say("\n=== 3.1 个 share, shared window and overt noun (percent) ===")
for c in CHILDREN:
    s = shared[c]
    say(c, "all tokens", share_plain(s), "| overt noun", share_plain(s[s.overt == 1]), "| noun omitted", share_plain(s[s.overt == 0]))
say("noun omitted vs overt (all months), children and mothers:")
for c, role in SPEAKERS:
    s = D[(D.child == c) & (D.role == role)]
    say(" ", LAB[(c, role)], "overt", share_plain(s[s.overt == 1]), "| omitted", share_plain(s[s.overt == 0]))

say("\n=== 3.2 Constructions ===")
for c in CHILDREN:
    s = kids[kids.child == c]
    say(c, "first month of each construction:", {k: fmt_age(s[s.construction == k].month.min()) for k in CONS[:4]},
        "| months when all four had occurred:", fmt_age(max(s[s.construction == k].month.min() for k in CONS[:4])))
say("个 share within NUM+CL, heritage children pooled:", share_plain(kids[kids.child.isin(["Avia", "Luna"]) & (kids.construction == "NUM+CL")]),
    "| XueXue:", share_plain(kids[(kids.child == "XueXue") & (kids.construction == "NUM+CL")]))
say("specific-classifier tokens by construction (all children, misselections excluded):",
    kids[(kids.spec == 1) & (kids.mis == 0)].construction.value_counts().to_dict())

NUMS, DEMS = set("一二两三四五六七八九十"), set("这那")


def _clean(w):
    return re.sub(r"^[<\[(]+|[>\])]+$", "", w)


def _before(r):
    """words immediately before the classifier in the utterance; a fused form such as 这个 gives its first character"""
    w = [_clean(t) for t in str(r.utterance).split()]
    cl, out = r.classifier, []
    for i, t in enumerate(w):
        if t == cl and i > 0:
            out.append(w[i - 1])
        elif len(t) == len(cl) + 1 and t.endswith(cl):
            out.append(t[0])
    return out


say("\nnumerals before the classifier (NUM+CL tokens) and demonstratives (DEM+CL tokens), all months:")
for c in CHILDREN:
    for cons, S, nm in [("NUM+CL", NUMS, "numerals"), ("DEM+CL", DEMS, "demonstratives")]:
        s = kids[(kids.child == c) & (kids.construction == cons)]
        got = collections.Counter()
        for _, r in s.iterrows():
            f = [p for p in _before(r) if p in S]
            if f:
                got[f[0]] += 1
        say(f"  {c} {cons}: {len(s)} tokens, {len(got)} distinct {nm}: {dict(got.most_common())}; unidentified: {len(s) - sum(got.values())}")

say("\n=== 3.2 Omissions (from the separate search of numeral and demonstrative contexts; not token-file data) ===")
# Classifier omissions are absences, so they cannot appear in a token file. Counts as reported in the manuscript.
OMISSIONS = {"after a numeral": {"Avia": 0, "Luna": 0, "XueXue": 0},
             "after a demonstrative": {"Avia": 0, "Luna": 4, "XueXue": 0},
             "Luna's demonstrative omissions, ages": "2;5 (1) and 4;2–4;11 (3)",
             "mothers (one each, after a demonstrative)": "Avia's mother 1, XueXue's mother 1, Luna's mother 0"}
for k, v in OMISSIONS.items():
    say(" ", k, v)

say("\n=== 3.2 Misselections ===")
mis = D[D.mis == 1]
say("misselections:", len(mis), mis.speaker.value_counts().to_dict())
for _, r in mis.iterrows():
    say(f"  {r.speaker} {fmt_age(r.month)}: {r.classifier}; {r.note}; utterance: {r.utterance}")
her = kids[kids.child.isin(["Avia", "Luna"])]
say("heritage children's specific tokens (incl. misselections):", int(her.spec.sum()), "of which misselected:", int(her.mis.sum()))

say("\n=== 3.2 Redundant classifier phrases ===")
red = kids[kids.note.fillna("").str.contains("redundant classifier phrase")]
red = red.assign(kind=red.note.str.extract(r"(redundant classifier phrase: [^|]*)")[0].str.strip())
u = red.drop_duplicates(["session", "utterance"])
say("heritage utterances with a redundant classifier phrase carried by a token in the file:", len(u), u.kind.value_counts().to_dict(),
    "by child:", u.child.value_counts().to_dict())
# One further utterance (Luna, 4;0, in the 'hesitation or restart' group) contains no classifier token in the file and
# is counted from the transcript, giving the 14 reported in the manuscript (12 hesitations/restarts + 2).
REDUNDANT_UTTERANCE_WITHOUT_TOKEN = 1
n_h = int((u.kind == "redundant classifier phrase: hesitation or restart").sum()) + REDUNDANT_UTTERANCE_WITHOUT_TOKEN
n_c = int((u.kind == "redundant classifier phrase: DEM+CL before NUM+CL").sum())
say(f"plus {REDUNDANT_UTTERANCE_WITHOUT_TOKEN} utterance without a token in the file -> total {len(u) + REDUNDANT_UTTERANCE_WITHOUT_TOKEN}: "
    f"{n_h} hesitations or restarts, {n_c} DEM+CL before NUM+CL (Avia {int((u.child == 'Avia').sum())}, Luna {int((u.child == 'Luna').sum()) + REDUNDANT_UTTERANCE_WITHOUT_TOKEN})")
say("mothers' redundant phrases:", int(moms.note.fillna("").str.contains("redundant").sum()), "; XueXue:",
    int(kids[kids.child == "XueXue"].note.fillna("").str.contains("redundant").sum()))

say("\n=== 3.3 Specific-classifier repertoire ===")
for c in CHILDREN:
    s = kids[kids.child == c]
    sh = shared[c]
    say(c, "types (shared window):", spec_tokens(sh).value_counts().astype(int).to_dict(), "| rarefied to", M_RARE, ":",
        round(expected_types(spec_tokens(sh).value_counts().tolist(), M_RARE), 1))
# The nouns behind the next two analyses are listed in data/nouns_in_paper.csv (the token files have no noun column).
#   use == "alternation": nouns that one child used with both 个 and a correctly selected specific classifier
#   use == "specific_classifier_pairing": the nouns Avia used with 只, 本 and 条
NOUNS = pd.read_csv(os.path.join(DATA, "nouns_in_paper.csv"), encoding="utf-8-sig")
say("Avia: nouns occurring with 只, 本 and 条:")
for k, v in NOUNS[NOUNS.use == "specific_classifier_pairing"].groupby("classifier", sort=False):
    say(f"  {k}: {v.noun.nunique()} nouns {v.noun.tolist()}")
say("\nAlternation with the same noun (misselections not counted as specific):")
alt = NOUNS[NOUNS.use == "alternation"]
for c in CHILDREN:
    t = alt[alt.speaker == c]
    say(f"  {c}: {len(t)} nouns used with both 个 and a specific classifier; {int((t.within_one_session == 'yes').sum())} of them within a single session")
say("self-correction:", kids[kids.note.fillna("").str.contains("self-correction")][["speaker", "month", "utterance", "classifier"]].values.tolist())

say("\n=== 3.4 / 4 Mothers and children (个 share) ===")
for c in CHILDREN:
    k_, m_ = kids[kids.child == c], moms[moms.child == c]
    say(c, f"child {100 * k_.gen.mean():.1f}% vs mother {100 * m_.gen.mean():.1f}% | overt: {100 * k_[k_.overt == 1].gen.mean():.1f}% vs "
        f"{100 * m_[m_.overt == 1].gen.mean():.1f}% | DEM+CL: {100 * k_[k_.construction == 'DEM+CL'].gen.mean():.1f}% vs "
        f"{100 * m_[m_.construction == 'DEM+CL'].gen.mean():.1f}% | NUM+CL: {100 * k_[k_.construction == 'NUM+CL'].gen.mean():.1f}% vs "
        f"{100 * m_[m_.construction == 'NUM+CL'].gen.mean():.1f}%")
for c in ["Avia", "Luna"]:
    k_ = kids[kids.child == c]
    byyear = {fmt_age(y * 12)[:1]: round(float(100 * (k_[k_.year == y].construction == "DEM+CL").mean()), 1) for y in sorted(k_.year.unique())}
    say(c, "DEM+CL share of tokens by year of age:", byyear, "| NUM+CL:",
        {fmt_age(y * 12)[:1]: round(float(100 * (k_[k_.year == y].construction == "NUM+CL").mean()), 1) for y in sorted(k_.year.unique())})

say("\n=== 4.2 Overt-noun 个 share in the shared window and heritage pooled ===")
say("heritage children pooled, overt noun, shared window:", share_plain(pd.concat([shared['Avia'], shared['Luna']]).query("overt == 1")))

open(os.path.join(OUT, "numbers.txt"), "w", encoding="utf-8").write(LOG.getvalue())
