"""Notebook cells, part 3: Labs 7 to 10, wrap-up and appendix."""


def cells(M, C):
    # ------------------------------------------------------------------ Lab 7
    M(r'''---
# Lab 7. Graph analytics

**Goal.** Use graph structure to answer questions that tables answer badly.

Three families of analysis are worth knowing. **Centrality** measures find important or bridging nodes.
**Community detection** finds densely connected groups. **Impact analysis** follows dependencies to see what a change
touches. We use the *collaboration graph*: two people are linked if they work on a common project, weighted by the
number of shared projects.
''')
    C(r'''works = triples[triples.relation.isin(["works_on", "leads"])][["head_id", "tail_id"]].drop_duplicates()
B = nx.Graph()
B.add_edges_from(works.itertuples(index=False, name=None))           # bipartite: people and projects
people_nodes = set(works.head_id)
# person-person projection, weight = number of shared projects (built in sorted order so results are reproducible)
members_of = defaultdict(set)
for h, t in works.itertuples(index=False, name=None):
    members_of[t].add(h)
Cg = nx.Graph()
Cg.add_nodes_from(sorted(people_nodes))
for proj in sorted(members_of):
    ms = sorted(members_of[proj])
    for i, u in enumerate(ms):
        for v in ms[i + 1:]:
            w = Cg[u][v]["weight"] + 1 if Cg.has_edge(u, v) else 1
            Cg.add_edge(u, v, weight=w)

dept_of = {r.head_id: label[r.tail_id] for r in triples[triples.relation == "works_in"].itertuples()}
print(f"collaboration graph: {Cg.number_of_nodes()} people, {Cg.number_of_edges()} edges, "
      f"{nx.number_connected_components(Cg)} connected component(s), density {nx.density(Cg):.3f}")

bet = nx.betweenness_centrality(Cg)
pr = nx.pagerank(Cg, weight="weight")
nproj = {p: B.degree(p) for p in people_nodes}
top = (pd.DataFrame({"person": [label[p] for p in Cg], "department": [dept_of[p] for p in Cg],
                     "projects": [nproj[p] for p in Cg], "collaborators": [Cg.degree(p) for p in Cg],
                     "betweenness": [bet[p] for p in Cg], "pagerank": [pr[p] for p in Cg]})
       .sort_values("betweenness", ascending=False).head(8).round(4).reset_index(drop=True))
display(top)''')

    M(r'''**Betweenness** counts the shortest paths that pass through a node. People with high betweenness and few projects
are often the cross-team connectors that an org chart does not show. Next, communities: we run Louvain
modularity optimisation and ask how closely the discovered groups match the real departments, using the adjusted Rand
index (ARI, where 0 means chance agreement and 1 means identical partitions) and normalised mutual information (NMI).''')
    C(r'''from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

nodes = sorted(Cg)
# Louvain iterates over sets internally; integer labels hash identically in every run, which makes the result reproducible
to_int = {n: i for i, n in enumerate(nodes)}
to_node = {i: n for n, i in to_int.items()}
raw = nx.community.louvain_communities(nx.relabel_nodes(Cg, to_int), weight="weight", seed=SEED)
comms = [{to_node[i] for i in c} for c in raw]
comm_of = {n: i for i, c in enumerate(sorted(comms, key=lambda c: (-len(c), min(c)))) for n in c}
y_dept = [dept_of[n] for n in nodes]
y_comm = [comm_of[n] for n in nodes]
ari = adjusted_rand_score(y_dept, y_comm)
nmi = normalized_mutual_info_score(y_dept, y_comm)
print(f"{len(comms)} communities | ARI vs departments {ari:.3f} | NMI {nmi:.3f}")

cross = sum(dept_of[u] != dept_of[v] for u, v in Cg.edges()) / Cg.number_of_edges()
print(f"share of collaboration edges that cross departments: {cross:.3f}")

dept_names = sorted({dept_of[n] for n in Cg})      # the Executive Office has no projects, so it is absent
dcol = {d: SERIES[i] for i, d in enumerate(dept_names)}
pos = nx.spring_layout(Cg, seed=SEED, weight="weight", k=0.35, iterations=200)
size = [30 + 900 * bet[n] for n in Cg]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax = axes[0]
nx.draw_networkx_edges(Cg, pos, ax=ax, edge_color="#cfcfca", width=0.5)
for d in dept_names:
    ns = [n for n in Cg if dept_of[n] == d]
    nx.draw_networkx_nodes(Cg, pos, nodelist=ns, node_color=dcol[d], node_size=[30 + 900 * bet[n] for n in ns],
                           ax=ax, label=d, edgecolors="white", linewidths=0.8)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, markerscale=0.7)
ax.set_title("Colour: department  |  size: betweenness"); ax.axis("off")
ax = axes[1]
nx.draw_networkx_edges(Cg, pos, ax=ax, edge_color="#cfcfca", width=0.5)
ncomm = len(comms)
for c in range(ncomm):
    ns = [n for n in Cg if comm_of[n] == c]
    nx.draw_networkx_nodes(Cg, pos, nodelist=ns, node_color=SERIES[c % 8], node_size=45, ax=ax,
                           edgecolors="white", linewidths=0.8, label=f"community {c + 1}")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=8, markerscale=0.7)
ax.set_title(f"Colour: Louvain community (ARI {ari:.2f})"); ax.axis("off")
fig.subplots_adjust(wspace=0.12)
plt.show()
savefig(fig, "fig_collab.png")
RESULTS["analytics"] = {"people": Cg.number_of_nodes(), "edges": Cg.number_of_edges(), "communities": len(comms),
                        "ari": round(ari, 3), "nmi": round(nmi, 3), "cross_dept_share": round(cross, 3),
                        "top_bridge": top.person[0], "top_bridge_dept": top.department[0]}''')

    M(r'''The communities follow the departments only partly, because about a third of the collaboration edges cross department
lines. That is exactly the structure a tabular `GROUP BY department` hides.

### Impact analysis over dependencies

If a vulnerability is announced in one technology, which projects are affected? A project is affected if it uses the
technology directly **or** uses anything that depends on it, directly or indirectly. This is a transitive query that
SQL handles awkwardly and a graph handles naturally.''')
    C(r'''dep = nx.DiGraph(triples[triples.relation == "depends_on"][["head_id", "tail_id"]].itertuples(index=False, name=None))
uses = triples[triples.relation == "uses_tech"]
tech_ids = entities[entities.group == "Technology"].entity_id

rows = []
for t in tech_ids:
    dependants = nx.ancestors(dep, t) if t in dep else set()       # technologies that rest on t
    affected = set(uses[uses.tail_id.isin(dependants | {t})].head_id)
    direct = set(uses[uses.tail_id == t].head_id)
    rows.append({"technology": label[t], "type": etype[t], "direct projects": len(direct),
                 "dependent technologies": len(dependants), "affected projects": len(affected)})
impact = pd.DataFrame(rows).sort_values("affected projects", ascending=False).reset_index(drop=True)
display(impact.head(8))
RESULTS["impact"] = {"top": impact.technology[0], "affected": int(impact["affected projects"][0]),
                     "direct": int(impact["direct projects"][0]), "total_projects": int((entities.group == "Project").sum())}''')

    M(r'''Notice how much larger the affected set is than the direct one for a foundational language. Also notice the
boundary of the analysis: it can only see dependencies someone recorded. A graph answers "what does the data say
depends on what", never "what really depends on what".

### Exercise 7.1
For each department, compute the share of its members' collaboration edges that go to people in *other* departments.
Which department is the most outward-facing, and is that what you expected?
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 8
    M(r'''---
# Lab 8. Mining rules from the graph

**Goal.** Discover logical regularities automatically and quantify how reliable they are.

A **rule** has a body and a head, for example
`works_on(x, z) & uses_tech(z, y)  =>  has_skill(x, y)`:
if a person works on a project that uses a technology, they probably have that skill. For every rule we compute

* **support**: the number of pairs *(x, y)* for which both body and head hold;
* **confidence**: support divided by the number of pairs for which the body holds;
* **head coverage**: support divided by the number of facts with that head relation.

This is the idea behind systems such as AMIE. Matrix arithmetic makes it cheap: the body `r1(x,z) & r2(z,y)` is the
Boolean product of the two adjacency matrices. An exponent `^-1` marks an inverse relation (the same edges read
backwards).''')
    C(r'''from scipy import sparse

ent_index = {e: i for i, e in enumerate(entities.entity_id)}
NN = len(ent_index)
A = {}
for rel, grp in triples.groupby("relation"):
    A[rel] = sparse.csr_matrix((np.ones(len(grp)), (grp.head_id.map(ent_index).values, grp.tail_id.map(ent_index).values)),
                               shape=(NN, NN))
atoms = {}
for rel, Mx in A.items():
    atoms[rel] = Mx
    atoms[rel + "^-1"] = Mx.T.tocsr()


def boolean(Mx):
    Mx = Mx.tocsr().copy()
    Mx.data[:] = 1
    Mx.eliminate_zeros()
    return Mx


def drop_diagonal(Mx):
    Mx = Mx - sparse.diags(Mx.diagonal())
    Mx.eliminate_zeros()
    return Mx.tocsr()


rules = []
for head, H in A.items():
    nH = H.nnz
    cands = [((a,), atoms[a]) for a in atoms]
    cands += [((a, b), boolean(atoms[a] @ atoms[b])) for a in atoms for b in atoms]
    for body, Bm in cands:
        if body == (head,):
            continue
        Bm = drop_diagonal(Bm)
        if Bm.nnz == 0:
            continue
        sup = H.multiply(Bm).nnz
        if sup < 15:
            continue
        text = " & ".join(f"{a}({'x' if k == 0 else 'z'},{'z' if k == 0 else 'y'})" for k, a in enumerate(body)) \
            if len(body) == 2 else f"{body[0]}(x,y)"
        rules.append({"rule": f"{text}  =>  {head}(x,y)", "support": sup, "body": Bm.nnz,
                      "confidence": sup / Bm.nnz, "head_coverage": sup / nH})
rules = pd.DataFrame(rules).sort_values(["confidence", "support"], ascending=False).reset_index(drop=True)
print(len(rules), "rules with support >= 15")
display(rules[rules.confidence >= 0.25].head(14).round(3))''')

    M(r'''Three kinds of rule appear.

1. **Schema rules re-discovered from data.** `leads(x,y) => works_on(x,y)` is the subproperty axiom of Lab 3. Its
   confidence is only about 0.44, because roughly half of the leaders' `works_on` edges were never recorded, so the rule
   is also a *missing-data detector*.
2. **Organisational regularities.** A person who reports to someone usually works in the same department as that
   person.
3. **Soft, probabilistic rules.** Working on a project that uses a technology is a strong but imperfect indicator of
   having the skill. These are the rules that suggest candidate facts for review.

> **Watch out.** A rule's confidence is measured under the *closed-world* assumption: facts not in the graph count as
> false. For a sparse relation like `has_skill` that makes confidence a pessimistic estimate, because some "failures" are
> simply unrecorded skills. AMIE and its successors use a "partial completeness" correction for this reason.

### Exercise 8.1
For the rule `works_on(x,z) & uses_tech(z,y) => has_skill(x,y)`, report support, confidence, and the number of
**new** `has_skill` facts it would propose (body true, head absent). Show three proposals with names.
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 9
    M(r'''---
# Lab 9. Knowledge graph embeddings

**Goal.** Learn vector representations of entities and relations, and use them to predict missing links.

Rules are crisp and interpretable but cannot weigh much evidence at once. **Embedding models** place every entity
and relation in a vector space so that true triples score higher than false ones. We implement two classic models
from scratch.

* **TransE** treats a relation as a translation: for a true triple, `h + r` should land near `t`. The score is the
  negative distance `-|| h + r - t ||`.
* **DistMult** uses a three-way product: `score(h, r, t) = sum_i h_i * r_i * t_i`. It is symmetric in `h` and `t` by
  construction, so it cannot tell direction.

Training uses **negative sampling**: for each true triple we corrupt the head or the tail with a random entity and
teach the model to score the corrupted version lower.
''')
    C(r'''ent_ids = list(entities.entity_id)
eidx = {e: i for i, e in enumerate(ent_ids)}
rels = sorted(triples.relation.unique())
ridx = {r: i for i, r in enumerate(rels)}
N, R = len(ent_ids), len(rels)
T = np.array([[eidx[h], ridx[r], eidx[t]] for h, r, t in zip(triples.head_id, triples.relation, triples.tail_id)])

perm = np.random.default_rng(SEED).permutation(len(T))
n10 = int(0.1 * len(T))
train = T[perm[2 * n10:]]
seen = set(train[:, 0]) | set(train[:, 2])
keep = lambda arr: np.array([x for x in arr if x[0] in seen and x[2] in seen])
test, valid = keep(T[perm[:n10]]), keep(T[perm[n10:2 * n10]])
print(f"train {len(train)} | validation {len(valid)} | test {len(test)}  (entities: {N}, relations: {R})")

# filtering data: all known true tails for (h, r) and heads for (r, t)
tails_of, heads_of = defaultdict(set), defaultdict(set)
for h, r, t in T:
    tails_of[(h, r)].add(t)
    heads_of[(r, t)].add(h)
# candidate masks from the ontology: only entities whose type fits the relation's domain or range
dom_mask, ran_mask = {}, {}
for rel, spec in ontology["properties"].items():
    dom_mask[ridx[rel]] = np.array([is_a(e, spec["domain"]) for e in ent_ids])
    ran_mask[ridx[rel]] = np.array([is_a(e, spec["range"]) for e in ent_ids])''')

    M(r'''### Evaluation protocol

For each test triple we hide the tail and rank **all** entities as candidates for it, then do the same for the head.
Two standard refinements apply:

* **Filtered ranking.** Other candidates that are known true answers are removed, so the model is not punished for
  ranking another correct answer above the held-out one.
* **Ontology-typed candidates** (our addition, and optional): only entities that satisfy the relation's domain or
  range are ranked. This feeds the schema from Lab 4 back into the evaluation.

We report **MRR** (mean reciprocal rank), **Hits@k** (share of test cases with the true entity in the top k) and the
mean rank. We compare against a **popularity baseline** (rank by how often an entity appears with the relation in
training) and a **random** scorer. A model that cannot beat popularity has learned nothing about structure.''')
    C(r'''def evaluate(score_fn, data, typed=False):
    ranks, rel_of = [], []
    for h, r, t in data:
        for side in (1, 0):                                   # 1: predict tail, 0: predict head
            s = np.asarray(score_fn(h, r, t, side), dtype=float).copy()
            true = t if side == 1 else h
            bad = np.zeros(N, bool)
            others = (tails_of[(h, r)] if side == 1 else heads_of[(r, t)]) - {true}
            if others:
                bad[list(others)] = True
            if typed:
                bad |= ~(ran_mask[r] if side == 1 else dom_mask[r])
            s[bad] = -np.inf
            greater, ties = (s > s[true]).sum(), (s == s[true]).sum() - 1
            ranks.append(1 + greater + ties / 2)
            rel_of.append(r)
    return np.array(ranks), np.array(rel_of)


def metrics(ranks):
    return {"MRR": float((1 / ranks).mean()), "Hits@1": float((ranks <= 1).mean()),
            "Hits@3": float((ranks <= 3).mean()), "Hits@10": float((ranks <= 10).mean()), "MR": float(ranks.mean())}


# baselines
pop_t, pop_h = np.zeros((R, N)), np.zeros((R, N))
for h, r, t in train:
    pop_t[r, t] += 1
    pop_h[r, h] += 1
popularity = lambda h, r, t, side: (pop_t[r] if side == 1 else pop_h[r])
rnd = np.random.default_rng(0)
random_scorer = lambda h, r, t, side: rnd.random(N)''')

    M(r'''### The two models

Both are a few dozen lines of NumPy. The gradients are derived by hand, which is instructive: for TransE the gradient
of `||h + r - t||` is the unit vector `(h + r - t) / ||h + r - t||`, and for DistMult with a logistic loss it is the
elementwise product of the other two factors.''')
    C(r'''def train_transe(dim=50, epochs=200, lr=0.05, margin=1.0, bs=128, seed=0):
    rg = np.random.default_rng(seed)
    bound = 6 / np.sqrt(dim)
    E = rg.uniform(-bound, bound, (N, dim))
    Rr = rg.uniform(-bound, bound, (R, dim))
    Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
    history = []
    for ep in range(epochs):
        E /= np.linalg.norm(E, axis=1, keepdims=True)         # keep entity vectors on the unit sphere
        order = rg.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), bs):
            b = train[order[i:i + bs]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            h2, t2 = h.copy(), t.copy()
            flip = rg.random(len(b)) < 0.5                    # corrupt head or tail with equal probability
            h2[flip] = rg.integers(0, N, flip.sum())
            t2[~flip] = rg.integers(0, N, (~flip).sum())
            dp, dn = E[h] + Rr[r] - E[t], E[h2] + Rr[r] - E[t2]
            lp, ln = np.linalg.norm(dp, axis=1), np.linalg.norm(dn, axis=1)
            viol = (margin + lp - ln) > 0                     # margin ranking loss is active only when violated
            total += np.maximum(0, margin + lp - ln).sum()
            if not viol.any():
                continue
            gp = dp[viol] / (lp[viol, None] + 1e-9)
            gn = dn[viol] / (ln[viol, None] + 1e-9)
            np.add.at(E, h[viol], -lr * gp); np.add.at(E, t[viol], lr * gp); np.add.at(Rr, r[viol], -lr * gp)
            np.add.at(E, h2[viol], lr * gn); np.add.at(E, t2[viol], -lr * gn); np.add.at(Rr, r[viol], lr * gn)
        history.append(total / len(train))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E, Rr, history


def train_distmult(dim=50, epochs=100, lr=0.1, neg=5, bs=256, reg=1e-3, seed=0):
    rg = np.random.default_rng(seed)
    E, Rr = rg.normal(0, 0.1, (N, dim)), rg.normal(0, 0.1, (R, dim))
    GE, GR = np.full_like(E, 1e-8), np.full_like(Rr, 1e-8)    # Adagrad accumulators
    for ep in range(epochs):
        order = rg.permutation(len(train))
        for i in range(0, len(order), bs):
            b = train[order[i:i + bs]]
            B = len(b)
            h, r, t = (np.repeat(b[:, k], neg + 1) for k in range(3))
            y = np.full(B * (neg + 1), -1.0)
            y[::neg + 1] = 1.0                                # first copy is the positive, the rest are negatives
            flip = rg.random(len(h)) < 0.5
            rand = rg.integers(0, N, len(h))
            h = np.where((y < 0) & flip, rand, h)
            t = np.where((y < 0) & ~flip, rand, t)
            s = (E[h] * Rr[r] * E[t]).sum(1)
            g = -y / (1 + np.exp(y * s))                      # derivative of the logistic loss with respect to the score
            gE, gR = np.zeros_like(E), np.zeros_like(Rr)
            np.add.at(gE, h, g[:, None] * Rr[r] * E[t] + reg * E[h])
            np.add.at(gE, t, g[:, None] * Rr[r] * E[h] + reg * E[t])
            np.add.at(gR, r, g[:, None] * E[h] * E[t] + reg * Rr[r])
            GE += gE ** 2; GR += gR ** 2
            E -= lr * gE / np.sqrt(GE); Rr -= lr * gR / np.sqrt(GR)
    return E, Rr


def transe_scorer(E, Rr):
    def f(h, r, t, side):
        return -np.linalg.norm(E[h] + Rr[r] - E, axis=1) if side == 1 else -np.linalg.norm(E + Rr[r] - E[t], axis=1)
    return f


def distmult_scorer(E, Rr):
    def f(h, r, t, side):
        return (E[h] * Rr[r]) @ E.T if side == 1 else E @ (Rr[r] * E[t])
    return f


E0, R0, loss_hist = train_transe(seed=0)
print("TransE margin loss, first and last epoch:", round(loss_hist[0], 3), "->", round(loss_hist[-1], 3))''')

    M(r'''Now evaluate. The learned models are trained from three random initialisations and the table reports the mean
(standard deviation in the MRR column), because a single seed can mislead on a graph this small.''')
    C(r'''results = {}


def record(name, ranks_all, ranks_typed):
    results[name] = {"all": ranks_all, "typed": ranks_typed}


record("Random", *[evaluate(random_scorer, test, typed=ty) for ty in (False, True)])
record("Popularity", *[evaluate(popularity, test, typed=ty) for ty in (False, True)])

tr_runs, dm_runs = [], []
for seed in range(3):
    Es, Rs, _ = train_transe(seed=seed)
    f = transe_scorer(Es, Rs)
    tr_runs.append([evaluate(f, test, typed=ty) for ty in (False, True)])
    Ed, Rd = train_distmult(seed=seed)
    f = distmult_scorer(Ed, Rd)
    dm_runs.append([evaluate(f, test, typed=ty) for ty in (False, True)])

rows = []
for name, runs in [("DistMult", dm_runs), ("TransE", tr_runs)]:
    for k, mode in enumerate(("all", "typed")):
        ms = [metrics(run[k][0]) for run in runs]
        rows.append({"model": name, "candidates": "typed" if mode == "typed" else "all entities",
                     **{key: np.mean([m[key] for m in ms]) for key in ms[0]},
                     "MRR_std": np.std([m["MRR"] for m in ms])})
for name in ("Random", "Popularity"):
    for mode in ("all", "typed"):
        rows.append({"model": name, "candidates": "typed" if mode == "typed" else "all entities",
                     **metrics(results[name][mode][0]), "MRR_std": 0.0})
kge = pd.DataFrame(rows)
order_ = {"Random": 0, "Popularity": 1, "DistMult": 2, "TransE": 3}
kge = kge.sort_values(["candidates", "model"], key=lambda s: s.map(order_) if s.name == "model" else s).reset_index(drop=True)
display(kge.round(3))
RESULTS["kge"] = {f"{r.model}|{r.candidates}": {k: round(float(r[k]), 3) for k in ("MRR", "Hits@1", "Hits@3", "Hits@10", "MR", "MRR_std")}
                  for _, r in kge.iterrows()}''')

    C(r'''fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
models = ["Random", "Popularity", "DistMult", "TransE"]
x = np.arange(len(models)); w = 0.36
for k, (mode, col) in enumerate([("all entities", SERIES[0]), ("typed", SERIES[1])]):
    vals = [float(kge[(kge.model == m) & (kge.candidates == mode)].MRR.iloc[0]) for m in models]
    bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.04, color=col, label=mode if mode == "typed" else "all entities")
    for xi, v in zip(x + (k - 0.5) * w, vals):
        ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("MRR (filtered)")
ax.set_title("Link prediction on held-out triples")
ax.legend(title="candidate set", loc="upper left")
ax.grid(axis="x", visible=False)
ax = axes[1]
ax.plot(range(1, len(loss_hist) + 1), loss_hist, color=ACCENT, lw=2)
ax.set_xlabel("epoch"); ax.set_ylabel("mean margin loss")
ax.set_title("TransE training loss")
plt.tight_layout()
plt.show()
savefig(fig, "fig_kge.png")''')

    M(r'''Read the table in three steps. First, both learned models beat the popularity baseline, so they capture something
beyond "common entities are common answers". Second, restricting candidates by type helps the weaker model most:
DistMult wastes much of its ranking on entities of the wrong kind, while TransE has already learned the types. Note that typing alone lifts even the random scorer, from an MRR of 0.02 to about 0.10, so compare models within one
candidate setting, never across settings. Third, TransE leads on this graph. That ordering
is partly an accident of data size and shape (small graph, many one-to-many relations) and should not be read as a
general verdict.

Which relations are easy? A per-relation breakdown for the first TransE run shows where the structure is learnable.''')
    C(r'''ranks, rel_of = evaluate(transe_scorer(E0, R0), test, typed=True)
by_rel = (pd.DataFrame({"relation": [rels[r] for r in rel_of], "rr": 1 / ranks, "hit10": ranks <= 10})
          .groupby("relation").agg(cases=("rr", "size"), MRR=("rr", "mean"), hits10=("hit10", "mean"))
          .sort_values("MRR", ascending=False).round(3))
display(by_rel)
RESULTS["kge_by_relation"] = by_rel.reset_index().to_dict(orient="records")''')

    M(r'''### Looking inside the embedding space

Project the learned entity vectors to two dimensions with PCA. If the model has learned structure, entities of the
same kind should sit together. Differences *within* a type are smaller than differences between types, so for the
department question we fit a second PCA on the people alone, and we back the picture with a number: the share of each
person's five nearest neighbours (by cosine similarity) that belong to the same department.''')
    C(r'''from sklearn.decomposition import PCA

xy = PCA(n_components=2, random_state=SEED).fit_transform(E0)
grp_arr = np.array([group(etype[e]) for e in ent_ids])
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axes[0]
for g_, col in PALETTE.items():
    m = grp_arr == g_
    ax.scatter(xy[m, 0], xy[m, 1], s=22, color=col, label=g_, edgecolors="white", linewidths=0.4)
ax.legend(ncol=2, fontsize=8, loc="best")
ax.set_title("Entities by type"); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax = axes[1]
people_idx = [i for i in range(N) if ent_ids[i] in dept_of and dept_of[ent_ids[i]] in dept_names]
pxy = PCA(n_components=2, random_state=SEED).fit_transform(E0[people_idx])
pdept = np.array([dept_of[ent_ids[i]] for i in people_idx])
for k, d in enumerate(dept_names):
    sel = pdept == d
    ax.scatter(pxy[sel, 0], pxy[sel, 1], s=26, color=SERIES[k], label=d, edgecolors="white", linewidths=0.4)
ax.legend(ncol=2, fontsize=7.5, loc="best")
ax.set_title("People only, by department"); ax.set_xlabel("PC 1 (people)"); ax.set_ylabel("PC 2 (people)")
plt.tight_layout()
plt.show()
savefig(fig, "fig_embeddings.png")


def knn_purity(X, y, k=5):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -9)
    nn = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([(y[nn[i]] == y[i]).mean() for i in range(len(y))]))


purity = knn_purity(E0[people_idx], pdept)
chance = sum((c / len(pdept)) ** 2 for c in Counter(pdept).values())
print(f"nearest-neighbour department purity: {purity:.3f} (chance level {chance:.3f})")
RESULTS["kge_structure"] = {"knn_purity": round(purity, 3), "chance": round(chance, 3)}


def neighbours(name, k=6, E=E0):
    v = E[eidx[label_to_id[name]]]
    sims = E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))
    idx = [i for i in np.argsort(-sims) if ent_ids[i] != label_to_id[name]][:k]
    return [(label[ent_ids[i]], round(float(sims[i]), 3)) for i in idx]


for nm in ["Project Atlas", "PyTorch"]:
    print(nm, "->", neighbours(nm))''')

    M(r'''### Using the model: suggest missing skills

The most useful output of a link predictor is a ranked list of *candidate facts* for people to review. Here we
ask TransE which technologies each person most likely has as a skill but has not yet been recorded as having.''')
    C(r'''f = transe_scorer(E0, R0)
r_skill = ridx["has_skill"]
person = label_to_id["Priya Nair"]
s = f(eidx[person], r_skill, None, 1).copy()
s[~ran_mask[r_skill]] = -np.inf
s[list(tails_of[(eidx[person], r_skill)])] = -np.inf
top5 = np.argsort(-s)[:5]
print("Recorded skills :", sorted(label[ent_ids[i]] for i in tails_of[(eidx[person], r_skill)]))
print("Suggested skills:", [label[ent_ids[i]] for i in top5])
used = {t for pr_ in objects(G, person, "works_on") + objects(G, person, "leads") for t in objects(G, pr_, "uses_tech")}
print("Technologies on their projects:", sorted(label[t] for t in used))''')

    M(r'''> **Key idea.** Embeddings and rules are complementary. Rules explain; embeddings generalise from many weak signals.
> A good workflow proposes candidates with embeddings, filters them with ontology constraints, and sends survivors to
> review.

> **Watch out.** Link prediction scores are not probabilities, and the closed-world evaluation above penalises
> correct new facts that happen to be unrecorded. Treat the metrics as a way of comparing models, not as accuracy.

### Exercise 9.1
Tune TransE on the **validation** set: try dimensions 20 and 50 and margins 0.5 and 1.0. Pick the best by validation MRR
(typed candidates) and report its test MRR. Why must the test set stay untouched until the end?
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 10
    M(r'''---
# Lab 10. Question answering: text retrieval versus graph traversal

**Goal.** Measure what a knowledge graph adds to retrieval, in the way retrieval-augmented generation (RAG) systems
use it.

`questions.csv` holds 60 questions of 1, 2 and 3 hops, with exact gold answers. We compare three systems.

| System | Knowledge source | How it answers |
|---|---|---|
| **A. Text retrieval** | the sentences | TF-IDF retrieves the 5 most similar sentences; answers are the entities of the right type mentioned in them |
| **B. Graph from text** | triples extracted in Lab 6 | parse the question into a path, then traverse |
| **C. Curated graph** | `triples.csv` plus RDFS inference | the same traversal over the full graph |

System A has an unfairly good "reader": it picks out the correct entity type perfectly. In a real RAG pipeline a
language model reads the passages instead, so A's numbers here are generous.

### The path language

A question becomes a **path**: a sequence of steps, each `rel>` (follow forward), `rel<` (follow backward) or `rel+>`
(follow transitively). "Which clients has X worked for?" is `works_on>,delivered_for>`. We write the executor and also
return the **evidence**, the chain of triples that justifies each answer.''')
    C(r'''class TripleIndex:
    def __init__(self, df, infer=True):
        self.fwd = defaultdict(lambda: defaultdict(set))
        self.bwd = defaultdict(lambda: defaultdict(set))
        rows = list(df[["head_id", "relation", "tail_id"]].itertuples(index=False, name=None))
        if infer:                                         # RDFS subproperty rule: leads implies works_on
            sub = {p: s["subPropertyOf"] for p, s in ontology["properties"].items() if s.get("subPropertyOf")}
            rows += [(h, sub[r], t) for h, r, t in rows if r in sub]
        for h, r, t in rows:
            self.fwd[r][h].add(t)
            self.bwd[r][t].add(h)

    def step(self, frontier, tok):
        rel, back, trans = tok.rstrip("<>+"), tok.endswith("<"), tok.endswith("+>")
        nxt = {}
        if trans:
            seen, queue = set(frontier), list(frontier.items())
            while queue:
                e, ev = queue.pop(0)
                for t in sorted(self.fwd[rel].get(e, ())):
                    if t not in seen:
                        seen.add(t)
                        nxt[t] = ev + [(e, rel, t)]
                        queue.append((t, nxt[t]))
            return nxt
        for e, ev in frontier.items():
            for x in sorted(self.bwd[rel].get(e, ()) if back else self.fwd[rel].get(e, ())):
                if x not in nxt:
                    nxt[x] = ev + [(x, rel, e) if back else (e, rel, x)]
        return nxt

    def run(self, start, path):
        frontier = {start: []}
        for tok in path.split(","):
            frontier = self.step(frontier, tok)
        frontier.pop(start, None)
        return frontier                                   # answer entity -> evidence chain


idx_curated = TripleIndex(triples)
idx_noinfer = TripleIndex(triples, infer=False)
idx_text = TripleIndex(kg_text)

ex_start = label_to_id["Priya Nair"]
ans = idx_curated.run(ex_start, "works_on>,delivered_for>")
for a, ev in ans.items():
    print(label[a], "<=", " ; ".join(verbalize(*e) for e in ev))''')

    M(r'''### Parsing questions

The questions follow a fixed set of templates, so a regular-expression parser is enough: it extracts the start entity
(linked by exact label) and picks the path. A production system would use a language model for this step, or
generate a graph query directly, but the principle is the same: map language to a structured query, run it, and keep
the evidence.''')
    C(r'''TEMPLATES = [
    (r"^Who leads (?P<x>.+)\?$", "leads<"),
    (r"^Which client is (?P<x>.+) delivered for\?$", "delivered_for>"),
    (r"^Which technologies does (?P<x>.+) use\?$", "uses_tech>"),
    (r"^Who reports directly to (?P<x>.+)\?$", "reports_to<"),
    (r"^Which technologies are used by the projects (?P<x>.+) works on\?$", "works_on>,uses_tech>"),
    (r"^Which clients has (?P<x>.+) worked for\?$", "works_on>,delivered_for>"),
    (r"^In which cities are the clients of the projects (?P<x>.+) works on based\?$", "works_on>,delivered_for>,located_in>"),
    (r"^Who works in the same department as (?P<x>.+)\?$", "works_in>,works_in<"),
    (r"^Which people have a skill in a technology used by (?P<x>.+)\?$", "uses_tech>,has_skill<"),
    (r"^What does (?P<x>.+) depend on, directly or indirectly\?$", "depends_on+>"),
    (r"^Who are the colleagues of (?P<x>.+) on shared projects\?$", "works_on>,works_on<"),
    (r"^Who is in the management chain above (?P<x>.+)\?$", "reports_to+>"),
]


def parse_question(q):
    for rx, path in TEMPLATES:
        m = re.match(rx, q)
        if m:
            return label_to_id.get(m.group("x")), path
    return None, None


questions = pd.read_csv(DATA / "questions.csv")
questions["gold"] = questions.answers.str.split(";").map(set)
parsed = [parse_question(q) for q in questions.question]
ok = np.mean([(s == a and p == b) for (s, p), a, b in zip(parsed, questions.start_entity_id, questions.path)])
print(f"parser accuracy (start entity and path): {ok:.3f}")
display(questions[["qid", "question", "hops", "answer_type"]].sample(5, random_state=3))''')

    M(r'''### Running the three systems''')
    C(r'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(sent.text)
S = vec.transform(sent.text)


def answer_text(question, answer_type, start, k=5):
    sims = linear_kernel(vec.transform([question]), S).ravel()
    found = set()
    for i in np.argsort(-sims)[:k]:
        for m in find_mentions(sent.text.iloc[i]):
            if is_a(m[3], [answer_type]):
                found.add(m[3])
    found.discard(start)
    return found


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


rows = []
for q, (start, path) in zip(questions.itertuples(), parsed):
    gold_a = q.gold
    a = answer_text(q.question, q.answer_type, start)
    b = set(idx_text.run(start, path))
    c = set(idx_curated.run(start, path))
    c0 = set(idx_noinfer.run(start, path))
    rows.append({"qid": q.qid, "hops": q.hops, "path": path, "n_gold": len(gold_a),
                 "A_text": prf(a, gold_a)[2], "B_graph_from_text": prf(b, gold_a)[2],
                 "C_curated": prf(c, gold_a)[2], "C_no_inference": prf(c0, gold_a)[2],
                 "A_recall": prf(a, gold_a)[1], "B_recall": prf(b, gold_a)[1], "C_recall": prf(c, gold_a)[1]})
qa = pd.DataFrame(rows)
by_hops = qa.groupby("hops")[["A_text", "B_graph_from_text", "C_curated"]].mean()
by_hops.loc["all"] = qa[["A_text", "B_graph_from_text", "C_curated"]].mean()
display(by_hops.round(3))''')

    C(r'''fig, ax = plt.subplots(figsize=(7.4, 3.8))
groups_ = ["1", "2", "3", "all"]
vals = {"A. Text retrieval": by_hops.A_text.values, "B. Graph from text": by_hops.B_graph_from_text.values,
        "C. Curated graph": by_hops.C_curated.values}
x = np.arange(len(groups_)); w = 0.26
for k, (name, v) in enumerate(vals.items()):
    ax.bar(x + (k - 1) * w, v, w - 0.03, color=SERIES[k], label=name)
    for xi, vi in zip(x + (k - 1) * w, v):
        ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{g_} hop" + ("s" if g_ in ("2", "3") else "") if g_ != "all" else "all questions" for g_ in groups_])
ax.set_ylabel("mean answer F1"); ax.set_ylim(0, 1.12)
ax.set_title("Question answering by number of hops")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14))
ax.grid(axis="x", visible=False)
plt.show()
savefig(fig, "fig_qa.png")

abl = qa[qa.path.str.contains("works_on")]
print(f"\nablation on the {len(abl)} questions that use works_on: "
      f"F1 with RDFS inference {abl.C_curated.mean():.3f} | without {abl.C_no_inference.mean():.3f}")
RESULTS["qa"] = {"by_hops": by_hops.round(3).to_dict(orient="index"),
                 "recall_all": {"A": round(qa.A_recall.mean(), 3), "B": round(qa.B_recall.mean(), 3), "C": round(qa.C_recall.mean(), 3)},
                 "ablation": {"questions": len(abl), "with_inference": round(abl.C_curated.mean(), 3),
                              "without": round(abl.C_no_inference.mean(), 3)},
                 "parser_accuracy": round(float(ok), 3)}''')

    M(r'''The pattern is clear. Text retrieval is competitive only when the answer sits in one or two sentences, and it
degrades as questions need several facts to be joined (only five questions have three hops, so that last group is noisy). The graph built from text is limited by the extraction recall
of Lab 6: every missed sentence is a missing edge, and missing edges compound along a path. The curated graph answers
all questions exactly, which is expected, since the gold answers were computed from it; the point of the comparison is
how much each source loses, not the score of C itself.

The ablation shows the value of the schema again: without the subproperty rule, questions whose paths cross `works_on`
lose the project leaders.

### From traversal to a RAG prompt

A graph-based RAG system passes the *evidence* to a language model. Two styles are common. The naive style hands over
the whole neighbourhood of the start entity. The guided style hands over only the triples that lie on the answer paths.
We compare their size for one question.''')
    C(r'''def neighbourhood_facts(index, start, hops=2):
    frontier, seen, facts = {start}, {start}, set()
    for _ in range(hops):
        nxt = set()
        for e in frontier:
            for rel in index.fwd:
                for t in index.fwd[rel].get(e, ()):
                    facts.add((e, rel, t)); nxt.add(t)
            for rel in index.bwd:
                for h in index.bwd[rel].get(e, ()):
                    facts.add((h, rel, e)); nxt.add(h)
        frontier = nxt - seen
        seen |= nxt
    return facts


qtext = "Which clients has Priya Nair worked for?"
start, path = parse_question(qtext)
answers = idx_curated.run(start, path)
guided = {tuple(e) for ev in answers.values() for e in ev}
naive_facts = neighbourhood_facts(idx_curated, start)
print(f"question : {qtext}")
print(f"answer   : {sorted(label[a] for a in answers)}")
print(f"naive 2-hop neighbourhood: {len(naive_facts)} facts | guided evidence: {len(guided)} facts\n")

prompt = "Answer using only the facts below.\n\nFacts:\n" + "\n".join("- " + verbalize(*f) for f in sorted(guided)) + f"\n\nQuestion: {qtext}"
print(prompt)
RESULTS["rag_context"] = {"naive_facts": len(naive_facts), "guided_facts": len(guided)}''')

    M(r'''> **Key idea.** The graph does two jobs in a RAG system. It **finds** the relevant facts by following relations that a
> text embedding cannot see, and it **explains** the answer, because every answer comes with the chain of triples that
> produced it.

> **Watch out.** Graph retrieval fails silently when the graph is incomplete. A missing edge yields a confident but
> partial answer. Report coverage alongside accuracy, and keep a text-retrieval fallback.

### Exercise 10.1
Answer "Which people work on a project delivered for *Halcyon Bank*?" with the path executor. Which two steps are
needed, and in which direction does each go? Check the answer against the triples by hand for one person.
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ wrap-up
    M(r'''---
# Wrap-up

## What the numbers taught

* **Schema pays for itself.** Types filtered extraction candidates (Lab 6), caught bad triples (Lab 4), restricted
  link-prediction candidates (Lab 9) and supplied inference (Labs 3 and 10).
* **Every stage has a recall ceiling.** Blocking, the gazetteer and the extraction patterns each cap what the next stage
  can find, and the losses multiply along multi-hop paths.
* **Provenance is data.** Confidence and source on every edge made repair possible and would make audits possible.
* **No single method wins.** Rules explain, embeddings generalise, traversal answers exactly, text retrieval covers what
  nobody modelled.

## Where to go next

1. Replace the pattern extractor of Lab 6 with a language model prompted to emit triples against the same ontology, and
   score it with the same code.
2. Load `triples.csv` into a graph database and rewrite the Lab 3 queries in Cypher.
3. Replace TransE with a relational graph convolutional network and compare on the same split.
4. Let a language model write the path (or a Cypher query) for questions outside the 12 templates.

Finally, the cell below writes every metric collected in this notebook to `figures/results.json`; the companion
note quotes those figures.''')
    C(r'''def to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


(FIG / "results.json").write_text(json.dumps(RESULTS, indent=2, default=to_py))
print("saved", FIG / "results.json", "|", sorted(RESULTS))
print("figures:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))''')

    # ------------------------------------------------------------------ appendix
    M(r'''---
# Appendix. Worked solutions

Try each exercise before reading its solution.
''')

    M(r'''### Solution 1.1''')
    C(r'''print(triples.groupby("relation").tail_id.nunique().sort_values(ascending=False).head(3).to_string(), "\n")

out_deg = triples.groupby("head_id").size().reindex(entities.entity_id, fill_value=0)
by_type = entities.assign(out=out_deg.values).groupby("type").out.mean().sort_values(ascending=False)
print(by_type.round(2).head(4).to_string(), "\n")

no_out = entities[out_deg.values == 0]
print(len(no_out), "entities without outgoing triples:", dict(no_out.groupby("group").size()))''')

    M(r'''### Solution 2.1''')
    C(r'''neo = label_to_id["Neo4j"]
gaps = []
for proj in subjects(G, neo, "uses_tech"):
    for p in set(subjects(G, proj, "works_on")) | set(subjects(G, proj, "leads")):
        if neo not in objects(G, p, "has_skill"):
            gaps.append((label[p], label[proj]))
print(len(gaps), "training gaps")
display(pd.DataFrame(gaps, columns=["person", "project"]).head(8))''')

    M(r'''### Solution 3.1

Inference adds only the leaders whose own `works_on` edge was left out. In this data the two columns match, which is a
useful finding in itself: inference changed team rosters but not which clients each department reaches.''')
    C(r'''q31 = """
SELECT ?dept (COUNT(DISTINCT ?c) AS ?clients) WHERE {
  ?p ex:works_in ?d . ?d rdfs:label ?dept .
  ?p ex:works_on ?pr . ?pr ex:delivered_for ?c .
}
GROUP BY ?dept ORDER BY DESC(?clients)
"""
g_closed = g
out = {}
for name, graph in [("plain", g_plain), ("with inference", g_closed)]:
    g = graph
    out[name] = sparql(q31).set_index("dept").clients.astype(int)
g = g_closed
display(pd.DataFrame(out))''')

    M(r'''### Solution 4.1''')
    C(r'''dept_map = dict(zip(triples[triples.relation == "works_in"].head_id, triples[triples.relation == "works_in"].tail_id))


def cross_dept_mentoring(df):
    sub = df[df.relation == "mentors"]
    bad = [(h, t) for h, t in zip(sub.head_id, sub.tail_id)
           if h in dept_map and t in dept_map and dept_map[h] != dept_map[t]]
    return len(sub), len(bad)


print("clean graph (rows, violations):", cross_dept_mentoring(triples))
print("dirty graph (rows, violations):", cross_dept_mentoring(dirty))
print("The rule is silent on both graphs. The injected errors are of other kinds (self loops, reversals, wrong types), so "
      "this constraint adds nothing here. Keep a constraint only if it catches violations that occur in practice.")''')

    M(r'''### Solution 5.1''')
    C(r'''fine = pd.DataFrame([(t, *pair_prf(t)) for t in np.round(np.arange(0.5, 1.001, 0.01), 2)],
                    columns=["threshold", "precision", "recall", "f1"])
hit = fine[fine.precision >= 1.0].iloc[0]
print(f"lowest threshold with precision 1.0: {hit.threshold:.2f} -> recall {hit.recall:.3f} (F1 {hit.f1:.3f})")

lost = pairs[pairs.truth & (pairs.score < hit.threshold)].sort_values("score", ascending=False)
view = lost.head(6).copy()
view["record_a"] = [rec.name_raw[a] + " (" + rec.source_system[a] + ")" for a in view.a]
view["record_b"] = [rec.name_raw[b] + " (" + rec.source_system[b] + ")" for b in view.b]
display(view[["record_a", "record_b", "score"]].round(3))''')

    M(r'''### Solution 6.1''')
    C(r'''al = pd.read_csv(DATA / "aliases.csv")
full_map = dict(surface_to_id)
for r in al[~al.ambiguous].itertuples():
    full_map[r.alias] = r.entity_id
rx_full = build_matcher(full_map)
v3 = run_extractor(rx=rx_full, mapping=full_map, negation_guard=True)
s3 = score_extraction(v3)
print("curated gazetteer :", s2)
print("complete gazetteer:", s3)
pred3 = set(map(tuple, v3[KEY].values))
rest = sorted(gold_set - pred3)
amb_forms = set(al[al.ambiguous].alias)
n_amb = sum(any(f in text_of[(d, s_)] for f in amb_forms) for d, s_, h, r, t in rest)
print(f"remaining false negatives: {len(rest)}, of which {n_amb} sit in sentences containing an ambiguous surface form")''')

    M(r'''### Solution 7.1''')
    C(r'''rows = []
for d in dept_names:
    members = [n for n in Cg if dept_of[n] == d]
    tot = sum(Cg.degree(n) for n in members)
    out_ = sum(1 for n in members for m in Cg[n] if dept_of[m] != d)
    rows.append((d, len(members), tot, round(out_ / tot, 3)))
display(pd.DataFrame(rows, columns=["department", "people", "collaboration edges", "share to other departments"])
        .sort_values("share to other departments", ascending=False).reset_index(drop=True))''')

    M(r'''### Solution 8.1''')
    C(r'''body = boolean(atoms["works_on"] @ atoms["uses_tech"])
body = drop_diagonal(body)
H = A["has_skill"]
sup = H.multiply(body).nnz
print(f"support {sup} | confidence {sup / body.nnz:.3f} | head coverage {sup / H.nnz:.3f}")

new = body - body.multiply(H)
new.eliminate_zeros()
rows_, cols_ = new.nonzero()
print("new has_skill proposals:", new.nnz)
for i in range(3):
    print("  ", label[ent_ids[rows_[i]]], "--has_skill-->", label[ent_ids[cols_[i]]])''')

    M(r'''### Solution 9.1

The test set must stay untouched because every choice made on it (here, which configuration to keep) leaks information
about it into the model. The selected configuration would then look better on that test set than it will on new data.''')
    C(r'''grid_rows = []
for dim in (20, 50):
    for margin in (0.5, 1.0):
        Es, Rs, _ = train_transe(dim=dim, margin=margin, seed=0)
        v = metrics(evaluate(transe_scorer(Es, Rs), valid, typed=True)[0])["MRR"]
        grid_rows.append((dim, margin, v, Es, Rs))
best_cfg = max(grid_rows, key=lambda x: x[2])
for dim, margin, v, _, _ in grid_rows:
    print(f"dim {dim:>2} margin {margin}: validation MRR {v:.3f}")
test_mrr = metrics(evaluate(transe_scorer(best_cfg[3], best_cfg[4]), test, typed=True)[0])["MRR"]
print(f"selected dim {best_cfg[0]}, margin {best_cfg[1]} -> test MRR {test_mrr:.3f}")
print("For reference, the untuned default (dim 50, margin 1.0, seed 0) scores",
      round(metrics(evaluate(transe_scorer(E0, R0), test, typed=True)[0])["MRR"], 3), "on the same test set.")
print("Differences this small are within seed noise on 132 validation cases; do not over-read a tuning gain.")''')

    M(r'''### Solution 10.1''')
    C(r'''client = label_to_id["Halcyon Bank"]
res = idx_curated.run(client, "delivered_for<,works_on<")
print(len(res), "people work on a project delivered for Halcyon Bank")
p0 = sorted(res, key=lambda e: label[e])[0]
print("example:", label[p0], "<=", " ; ".join(verbalize(*e) for e in res[p0]))''')
