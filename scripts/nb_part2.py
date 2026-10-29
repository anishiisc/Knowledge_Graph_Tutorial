"""Notebook cells, part 2: Labs 5 and 6."""


def cells(M, C):
    # ------------------------------------------------------------------ Lab 5
    M(r'''---
# Lab 5. Entity resolution

**Goal.** Decide which records from different systems describe the same real-world entity.

Before extraction or inference, a knowledge graph needs one node per real entity. Meridian's HR system, Jira,
the wiki and the staff directory all mention the same people with different spellings, and its CRM, contract and
finance systems do the same for clients. `raw_records.csv` holds 374 such records, and each carries a hidden
`gold_entity_id` that we use only for scoring.

The standard pipeline has four stages: **normalise**, **block**, **compare**, **cluster**.
''')
    C(r'''rec = pd.read_csv(DATA / "raw_records.csv").fillna("")
print(rec.shape)
display(rec[rec.gold_entity_id.isin(["per_038", "per_039", "cli_01"])].sort_values(["gold_entity_id", "source_system"]))
display(rec.groupby(["entity_type", "source_system"]).size().rename("records").to_frame())''')

    M(r'''### Stage 1 and 2: normalise and block

Normalisation reduces superficial variation: case, punctuation, the `Last, First` order, company suffixes, city codes.
**Blocking** avoids comparing every record with every other record (about 70,000 pairs here, and billions in a real
company). Records are only compared if they share a cheap key. For people we use the pair *(first initial, last
initial)*, which survives typos in the middle of a name and works for records that carry only initials.''')
    C(r'''CITY = {"bengaluru": "BLR", "bangalore": "BLR", "blr": "BLR", "pune": "PNQ", "pnq": "PNQ",
        "hyderabad": "HYD", "hyd": "HYD", "singapore": "SIN", "sg": "SIN", "london": "LON", "ldn": "LON",
        "toronto": "YYZ", "yyz": "YYZ"}
SUFFIX = {"ltd", "limited", "inc"}


def parse_person(raw):
    s = raw.strip()
    if "," in s:
        last, first = [x.strip() for x in s.split(",", 1)]
    else:
        parts = s.split()
        first, last = parts[0], parts[-1]
    clean = lambda x: re.sub(r"[^a-z]", "", x.lower())
    return clean(first), clean(last)


def parse_company(raw):
    toks = [t for t in re.sub(r"[^a-z0-9 ]", " ", raw.lower()).split() if t not in SUFFIX]
    return " ".join(toks)


def norm_email(e):
    return e.lower().replace("meridian-analytics.example", "meridian.example")


records = []
for r in rec.itertuples():
    d = {"id": r.record_id, "src": r.source_system, "type": r.entity_type, "gold": r.gold_entity_id,
         "email": norm_email(r.email_raw), "city": CITY.get(r.city_raw.lower(), "")}
    if r.entity_type == "Person":
        d["first"], d["last"] = parse_person(r.name_raw)
        d["block"] = ("P", d["first"][:1], d["last"][:1])
    else:
        d["name"] = parse_company(r.name_raw)
        d["block"] = ("C", d["name"][:1])
    records.append(d)

blocks = defaultdict(list)
for i, d in enumerate(records):
    blocks[d["block"]].append(i)
cand = [(a, b) for ids in blocks.values() for k, a in enumerate(ids) for b in ids[k + 1:]]

n = len(records)
gold_groups = Counter(d["gold"] for d in records)
true_pairs_total = sum(c * (c - 1) // 2 for c in gold_groups.values())
cand_true = sum(records[a]["gold"] == records[b]["gold"] for a, b in cand)
all_pairs = n * (n - 1) // 2
print(f"all pairs {all_pairs:,} | candidate pairs {len(cand):,} | reduction ratio {1 - len(cand) / all_pairs:.3f}")
print(f"true pairs {true_pairs_total} | kept by blocking {cand_true} | pair completeness {cand_true / true_pairs_total:.3f}")
RESULTS["er_blocking"] = {"records": n, "all_pairs": all_pairs, "candidate_pairs": len(cand),
                          "reduction_ratio": round(1 - len(cand) / all_pairs, 4),
                          "pair_completeness": round(cand_true / true_pairs_total, 4)}''')

    M(r'''### Stage 3: compare

For each candidate pair we compute a similarity score from several signals. String similarity uses the
**Jaro-Winkler** measure, which rewards a shared prefix and tolerates transpositions; we implement it from scratch.
Two signals are asymmetric in an important way: an e-mail or city that *matches* is evidence for a merge, but one that
is *present and different* is evidence against it, whereas a missing value says nothing.''')
    C(r'''def jaro(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(max(la, lb) // 2 - 1, 0)
    am, bm, m = [False] * la, [False] * lb, 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(lb, i + window + 1)):
            if not bm[j] and b[j] == ca:
                am[i] = bm[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if am[i]:
            while not bm[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    t /= 2
    return (m / la + m / lb + (m - t) / m) / 3


def jaro_winkler(a, b, p=0.1):
    j, l = jaro(a, b), 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        l += 1
    return j + l * p * (1 - j)


def name_sim(a, b):
    """Similarity of two name parts; an initial matches any name that starts with it."""
    if len(a) == 1 or len(b) == 1:
        return 1.0 if a[:1] == b[:1] else 0.0
    return jaro_winkler(a, b)


def signal(x, y):
    if x and y:
        return 1.0 if x == y else -1.0
    return 0.0


def score_pair(a, b):
    ra, rb = records[a], records[b]
    if ra["type"] == "Person":
        s = 0.35 * name_sim(ra["first"], rb["first"]) + 0.35 * name_sim(ra["last"], rb["last"])
        s += 0.20 * signal(ra["email"], rb["email"]) + 0.10 * signal(ra["city"], rb["city"])
        return s
    na, nb = ra["name"], rb["name"]
    contain = 1.0 if (na and nb and (na in nb or nb in na)) else 0.0
    dom = signal(ra["email"], rb["email"])
    return 0.70 * max(jaro_winkler(na, nb), contain) + 0.20 * dom + 0.10 * signal(ra["city"], rb["city"])


pairs = pd.DataFrame({"a": [a for a, _ in cand], "b": [b for _, b in cand]})
pairs["score"] = [score_pair(a, b) for a, b in cand]
pairs["truth"] = [records[a]["gold"] == records[b]["gold"] for a, b in cand]
pairs["kind"] = [records[a]["type"] for a, _ in cand]
print(pairs.groupby("truth").score.describe()[["count", "mean", "min", "50%", "max"]].round(3))''')

    M(r'''The two score distributions overlap only a little. Now we pick a threshold. Precision counts how many accepted
merges are correct; recall counts how many true pairs we found, measured against *all* true pairs, so recall
also reflects what blocking lost.''')
    C(r'''def pair_prf(thr):
    pred = pairs.score >= thr
    tp = int((pred & pairs.truth).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / true_pairs_total
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


grid = np.round(np.arange(0.40, 1.01, 0.02), 2)
curve = pd.DataFrame([(t, *pair_prf(t)) for t in grid], columns=["threshold", "precision", "recall", "f1"])
best = curve.loc[curve.f1.idxmax()]
print(f"best pairwise F1 {best.f1:.3f} at threshold {best.threshold:.2f} "
      f"(precision {best.precision:.3f}, recall {best.recall:.3f})")

fig, ax = plt.subplots(figsize=(6.2, 3.4))
for col, name, c in [("precision", "precision", SERIES[0]), ("recall", "recall", SERIES[1]), ("f1", "F1", SERIES[2])]:
    ax.plot(curve.threshold, curve[col], color=c, lw=2, label=name)
ax.axvline(best.threshold, color="#52514e", ls="--", lw=1)
ax.set_xlabel("match threshold"); ax.set_ylabel("pairwise score"); ax.set_ylim(0, 1.03)
ax.set_title("Pairwise entity resolution versus threshold")
ax.legend(loc="lower center", ncol=3)
plt.show()
savefig(fig, "fig_er_threshold.png")
THR = float(best.threshold)''')

    M(r'''### Stage 4: cluster

Pairwise decisions must become a partition: if A matches B and B matches C, then A, B and C are one entity.
The simplest method takes **connected components** of the match graph. Its weakness is chaining: a single
ambiguous record can fuse two real entities. Our data contains such traps on purpose: a record that carries only an initial, such as the wiki entry `Sneha S.`,
can fit several people (the HR system holds both a Sneha Smith and a Sarah Smith).

A useful domain constraint repairs most of this. A source system holds **at most one record per real person**, so two
clusters that both contain an HR record must not merge. We process pairs from highest to lowest score and refuse any
merge that would put two records of the same system in one cluster (a *cannot-link* constraint).''')
    C(r'''def cluster(thr, cannot_link):
    parent_ = list(range(len(records)))
    members = {i: {records[i]["src"]} for i in range(len(records))}

    def find(x):
        while parent_[x] != x:
            parent_[x] = parent_[parent_[x]]
            x = parent_[x]
        return x

    keep = pairs[pairs.score >= thr].sort_values("score", ascending=False)
    for a, b in zip(keep.a, keep.b):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if cannot_link and members[ra] & members[rb]:
            continue
        parent_[rb] = ra
        members[ra] |= members[rb]
    return [find(i) for i in range(len(records))]


def cluster_prf(labels):
    cg = Counter(zip(labels, [d["gold"] for d in records]))
    cc = Counter(labels)
    tp = sum(c * (c - 1) // 2 for c in cg.values())
    pred = sum(c * (c - 1) // 2 for c in cc.values())
    p, r = tp / pred, tp / true_pairs_total
    return p, r, 2 * p * r / (p + r)


naive = cluster(THR, cannot_link=False)
constr = cluster(THR, cannot_link=True)
tab = pd.DataFrame({"connected components": cluster_prf(naive), "with cannot-link": cluster_prf(constr)},
                   index=["precision", "recall", "F1"]).round(3)
display(tab)

# show a cluster that connected components fused incorrectly
cl_gold = defaultdict(set)
for lab, d in zip(naive, records):
    cl_gold[lab].add(d["gold"])
impure = [lab for lab, gs in cl_gold.items() if len(gs) > 1]
print("clusters mixing several real entities (connected components):", len(impure))
if impure:
    lab = impure[0]
    display(rec.loc[[i for i, l in enumerate(naive) if l == lab], ["record_id", "source_system", "name_raw", "city_raw", "gold_entity_id"]])
RESULTS["er"] = {"threshold": THR, "pair_best": {"precision": round(best.precision, 3), "recall": round(best.recall, 3), "f1": round(best.f1, 3)},
                 "components": [round(x, 3) for x in cluster_prf(naive)],
                 "cannot_link": [round(x, 3) for x in cluster_prf(constr)], "impure_clusters": len(impure)}''')

    M(r'''> **Key idea.** Entity resolution is a precision and recall trade-off with a cost on each side. A false merge
> corrupts a node, since facts about two people now sit on one. A missed merge only leaves a duplicate. Production systems
> usually pick a high-precision threshold and send the uncertain band to human review.

### Exercise 5.1
Find the lowest threshold at which pairwise **precision is exactly 1.0**, so that no false pair is accepted. What recall do
you pay for that? Then explain, using the records in `raw_records.csv`, what kind of true pair is lost first.
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 6
    M(r'''---
# Lab 6. From text to triples

**Goal.** Build a small extractor that turns sentences into triples, and score it honestly.

Most organisational knowledge lives in prose. A text-to-graph pipeline has two core steps:
**entity linking** (find mentions and map them to node identifiers) and **relation extraction** (decide which
relation, if any, links the mentions in a sentence). We use classical, transparent methods so every error can be
traced. Modern pipelines replace the pattern step with a language model, but the evaluation method is identical.

The 739 sentences in `doc_sentences.csv` were generated from true triples, so `doc_gold_triples.csv` gives an
exact answer key. About a tenth of the sentences are traps: negations, mentions without a relation, filler.
''')
    C(r'''sent = pd.read_csv(DATA / "doc_sentences.csv")
gold_t = pd.read_csv(DATA / "doc_gold_triples.csv")
gaz = pd.read_csv(DATA / "gazetteer.csv")
print(len(sent), "sentences |", len(gold_t), "gold triples |", len(gaz), "gazetteer entries")
display(sent.sample(8, random_state=4)[["doc_id", "text"]])''')

    M(r'''### Step 1: mention detection with a gazetteer

A **gazetteer** is a dictionary of known surface forms. We match the *longest* form at each position, so that
`Project Atlas` wins over `Atlas`. Ambiguous surface forms such as `P. Nair` were deliberately left out of the
gazetteer, because mapping them to a single node would be a guess.''')
    C(r'''surface_to_id = dict(zip(gaz.surface, gaz.entity_id))


def build_matcher(mapping):
    forms = sorted(mapping, key=len, reverse=True)
    return re.compile(r"(?<![\w])(" + "|".join(re.escape(s) for s in forms) + r")(?![\w])")


mention_re = build_matcher(surface_to_id)


def find_mentions(text, rx=mention_re, mapping=surface_to_id):
    return [(m.start(), m.end(), m.group(1), mapping[m.group(1)]) for m in rx.finditer(text)]


demo = "Priya Nair leads Atlas, which is delivered for Halcyon."
for m in find_mentions(demo):
    print(m)''')

    M(r'''### Step 2: relation patterns with type constraints

Each pattern pairs a *trigger phrase* with a relation and a direction. For a trigger, the **head** is the nearest
preceding mention whose type fits the relation's domain, and the **tails** are the following mentions that fit its
range. Using the ontology for this filter is the single most effective trick in the extractor: it removes most
spurious candidates without any learning. Some triggers are *reversed* (`X manages Y` means `Y reports_to X`).''')
    C(r'''PATTERNS = [
    (r"\bworks on\b", "works_on", False), (r"\bcontributes to\b", "works_on", False),
    (r"\bis part of the\b", "works_on", False),
    (r"\bleads\b", "leads", False), (r"\bis the lead for\b", "leads", False),
    (r"\buses\b", "uses_tech", False), (r"\bis built with\b", "uses_tech", False),
    (r"\brelies on\b", "uses_tech", False),
    (r"\bis delivered for\b", "delivered_for", False), (r"\bis being built for\b", "delivered_for", False),
    (r"\breports to\b", "reports_to", False), (r"\bis managed by\b", "reports_to", False),
    (r"\bmanages\b", "reports_to", True),
    (r"\bworks in the\b", "works_in", False), (r"\bis a member of the\b", "works_in", False),
    (r"\bis based in\b", "located_in", False), (r"\bis headquartered in\b", "located_in", False),
    (r"\bdepends on\b", "depends_on", False), (r"\bis built on\b", "depends_on", False),
    (r"\bhas strong experience with\b", "has_skill", False), (r"\bis skilled in\b", "has_skill", False),
    (r"\bmentors\b", "mentors", False), (r"\bis mentored by\b", "mentors", True),
]
NEGATION = re.compile(r"\b(no longer|not|never)\s*$")


def extract(text, rx=mention_re, mapping=surface_to_id, negation_guard=False):
    mentions = find_mentions(text, rx, mapping)
    if len(mentions) < 2:
        return []
    trig = sorted((m.start(), m.end(), rel, rev) for pat, rel, rev in PATTERNS for m in re.finditer(pat, text))
    out = []
    for k, (s, e, rel, rev) in enumerate(trig):
        end = trig[k + 1][0] if k + 1 < len(trig) else len(text)
        if negation_guard and NEGATION.search(text[:s]):
            continue
        spec = ontology["properties"][rel]
        before = [m for m in mentions if m[1] <= s]
        after = [m for m in mentions if m[0] >= e and m[1] <= end]
        if not rev:
            heads = [m for m in before if is_a(m[3], spec["domain"])]
            tails = [m for m in after if is_a(m[3], spec["range"])]
            if heads:
                out += [(heads[-1][3], rel, t[3]) for t in tails]
        else:
            tails = [m for m in before if is_a(m[3], spec["range"])]
            heads = [m for m in after if is_a(m[3], spec["domain"])]
            if tails:
                out += [(h[3], rel, tails[-1][3]) for h in heads]
    return out


for text in ["Priya Nair leads Atlas, which is delivered for Halcyon.",
             "Verdant Foods is headquartered in London.",
             "Project Atlas uses Postgres, Kafka and Python."]:
    print(text)
    for h, r, t in extract(text):
        print("    ", verbalize(h, r, t))''')

    M(r'''### Scoring against the answer key

We compare extracted triples with the gold triples **per sentence**. Precision is the share of extracted triples that
are correct; recall is the share of gold triples that were found.''')
    C(r'''def run_extractor(**kw):
    rows = []
    for r in sent.itertuples():
        for h, rel, t in extract(r.text, **kw):
            rows.append((r.doc_id, r.sent_idx, h, rel, t))
    return pd.DataFrame(rows, columns=["doc_id", "sent_idx", "head_id", "relation", "tail_id"])


KEY = ["doc_id", "sent_idx", "head_id", "relation", "tail_id"]
gold_set = set(map(tuple, gold_t[KEY].values))


def score_extraction(pred_df):
    pred_set = set(map(tuple, pred_df[KEY].values))
    tp = len(pred_set & gold_set)
    p, r = tp / max(len(pred_set), 1), tp / len(gold_set)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3),
            "extracted": len(pred_set), "correct": tp}


v1 = run_extractor()
s1 = score_extraction(v1)
print("v1 (patterns + types):", s1)

pred1 = set(map(tuple, v1[KEY].values))
by_rel = []
for rel in sorted(set(gold_t.relation)):
    g_ = {x for x in gold_set if x[3] == rel}
    p_ = {x for x in pred1 if x[3] == rel}
    tp = len(g_ & p_)
    by_rel.append((rel, len(g_), len(p_), round(tp / max(len(p_), 1), 3), round(tp / len(g_), 3)))
display(pd.DataFrame(by_rel, columns=["relation", "gold", "extracted", "precision", "recall"]))''')

    M(r'''### Error analysis

Never stop at a single F1 number. Read the errors.

* **False positives** (extracted but not in the key) show where the patterns are too eager.
* **False negatives** (in the key but missed) split into two causes: a mention was not found (a **linking**
  failure) or both mentions were found but no pattern fired (a **pattern** failure).''')
    C(r'''fp_rows = sorted(pred1 - gold_set)
print("false positives:", len(fp_rows))
text_of = {(r.doc_id, r.sent_idx): r.text for r in sent.itertuples()}
for d, s, h, r, t in fp_rows[:6]:
    print("  ", text_of[(d, s)], "->", verbalize(h, r, t))

fn_rows = sorted(gold_set - pred1)
link_fail = pattern_fail = 0
for d, s, h, r, t in fn_rows:
    ids = {m[3] for m in find_mentions(text_of[(d, s)])}
    if h in ids and t in ids:
        pattern_fail += 1
    else:
        link_fail += 1
print(f"\nfalse negatives: {len(fn_rows)} | linking failures: {link_fail} | pattern failures: {pattern_fail}")''')

    M(r'''The false positives are dominated by one cause: sentences such as *"X no longer works on Y"* match the trigger
`works on`. A negation guard removes them. The false negatives are all linking failures in this data: surface forms that
the gazetteer lacks. That is a data problem, not a pattern problem, and no amount of pattern tuning will fix it.''')
    C(r'''v2 = run_extractor(negation_guard=True)
s2 = score_extraction(v2)
cmp_ = pd.DataFrame({"v1 patterns": s1, "v2 + negation guard": s2}).T
display(cmp_)

# what does the extracted graph look like compared with the true graph?
kg_text = v2[["head_id", "relation", "tail_id"]].drop_duplicates().reset_index(drop=True)
truth_set = set(map(tuple, triples[["head_id", "relation", "tail_id"]].values))
ext_set = set(map(tuple, kg_text.values))
cov = len(ext_set & truth_set) / len(truth_set)
prec_kg = len(ext_set & truth_set) / len(ext_set)
print(f"\nunique triples extracted: {len(ext_set)} | correct: {prec_kg:.3f} | share of the true graph recovered: {cov:.3f}")
RESULTS["extraction"] = {"v1": s1, "v2": s2, "fp_v1": len(fp_rows), "fn_v1": len(fn_rows),
                         "linking_failures": link_fail, "pattern_failures": pattern_fail,
                         "kg_unique": len(ext_set), "kg_precision": round(prec_kg, 3), "kg_coverage": round(cov, 3)}''')

    M(r'''The extracted graph is accurate but **incomplete**: text mentions only part of what the structured sources know. That
distinction returns in Lab 10, where we compare a graph built from text with a curated one.

### Exercise 6.1
Build a *complete* gazetteer from `aliases.csv` (all unambiguous aliases, not only the 80 percent that were curated),
rerun extraction with the negation guard and report how much recall improves. What does the remaining gap consist of?
''')
    C(r'''# YOUR CODE HERE''')
