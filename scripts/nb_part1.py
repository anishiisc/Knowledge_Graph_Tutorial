"""Notebook cells, part 1: front matter, setup, Labs 1 to 4."""


def cells(M, C):
    # ------------------------------------------------------------------ front matter
    M(r'''# Knowledge Graphs: A Hands-On Lab

**Dr. Anish Roychowdhury | Generative AI Architect & Educator**

This notebook is the practical companion to the note *Knowledge Graphs: From Triples to Retrieval*. The note
explains the ideas; the notebook lets you run every one of them on a small, fully synthetic dataset.

You will build a knowledge graph for a fictional consultancy, **Meridian Analytics**, starting from flat tables and
ending with a question answering system that uses the graph. Along the way you will meet the data models, the schema
layer, quality control, entity resolution, text extraction, graph analytics, rule mining and embeddings.

## What you will do

| Lab | Topic | Main skill |
|---|---|---|
| 1 | Tables to triples | Read the data as subject, predicate, object statements |
| 2 | Property graph | Model and query a graph with NetworkX |
| 3 | RDF and SPARQL | Query with SPARQL and add RDFS inference |
| 4 | Validation | Check a graph against an ontology and find injected errors |
| 5 | Entity resolution | Merge duplicate records from several source systems |
| 6 | Text to triples | Extract facts from sentences and score the extractor |
| 7 | Graph analytics | Centrality, communities, impact analysis |
| 8 | Rule mining | Discover logical rules from the data |
| 9 | Embeddings | Train TransE and DistMult and predict missing links |
| 10 | Question answering | Compare text retrieval with graph traversal |

## How to use this notebook

Run the cells from top to bottom. Each lab ends with one or two exercises. The exercise cells are empty on purpose:
try them first, then compare with the worked solutions in the appendix at the end. The whole notebook runs in about
a minute on a laptop and needs only `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`, `matplotlib` and `rdflib`.

> **Key idea.** A knowledge graph is a set of typed facts about entities, plus a schema that says what kinds of
> facts are allowed. Everything in this notebook is a way of building, checking, querying or learning from that set.
''')

    C(r'''import json, re, math, random, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from rdflib import Graph, Namespace, RDF, RDFS, Literal

warnings.filterwarnings("ignore")
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

DATA = Path("data")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
RESULTS = {}          # metrics are collected here and written to figures/results.json at the end

ACCENT = "#1E5A96"
PALETTE = {"Person": "#2a78d6", "Technology": "#eb6834", "Project": "#1baf7a",
           "Client": "#eda100", "Department": "#e87ba4", "City": "#008300"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 20)


def savefig(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")


print("data files:", ", ".join(sorted(p.name for p in DATA.glob("*.csv"))))''')

    # ------------------------------------------------------------------ Lab 1
    M(r'''---
# Lab 1. From tables to triples

**Goal.** Look at the raw data and see that a graph is just a list of statements.

A knowledge graph stores facts as **triples**: *(head, relation, tail)*. For example,
`(Priya Nair, works_on, Project Atlas)`. Entities are the nodes, relations are the labelled edges.
Two tables carry the whole graph: `entities.csv` lists the nodes and `triples.csv` lists the edges.
''')

    C(r'''entities = pd.read_csv(DATA / "entities.csv")
triples = pd.read_csv(DATA / "triples.csv")
ontology = json.loads((DATA / "ontology.json").read_text())

label = dict(zip(entities.entity_id, entities.label))
etype = dict(zip(entities.entity_id, entities.type))
label_to_id = {v: k for k, v in label.items()}
parent = ontology["classes"]            # class -> parent class (or None)


def ancestors(cls):
    """A class followed by all of its superclasses, nearest first."""
    out = [cls]
    while parent.get(out[-1]):
        out.append(parent[out[-1]])
    return out


def group(cls):
    """Top-level class of a type, for example Engineer -> Person."""
    return ancestors(cls)[-1]


def is_a(eid, classes):
    """True if the entity's type is, or specialises, any of the given classes."""
    return bool(set(ancestors(etype[eid])) & set(classes))


entities["group"] = entities["type"].map(group)
display(entities.head(8))
display(entities.groupby(["group", "type"]).size().rename("count").reset_index())''')

    M(r'''Each entity has a stable identifier, a human-readable label and a **type**. Types form a hierarchy
(an `Engineer` is a `Person`; a `Database` is a `Technology`). The helper `is_a` respects that hierarchy, and we
will rely on it repeatedly.

Now the edges. Each row has a head, a relation, a tail and three edge properties: where the fact came from
(`source`), how much we trust it (`confidence`) and the year it became valid (`valid_from`).''')

    C(r'''display(triples.sample(8, random_state=1))

rel_counts = triples.relation.value_counts()
print(f"{len(entities)} entities, {len(triples)} triples, {triples.relation.nunique()} relations")

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.barh(rel_counts.index[::-1], rel_counts.values[::-1], color=ACCENT, height=0.65)
for y, v in enumerate(rel_counts.values[::-1]):
    ax.text(v + 6, y, str(v), va="center", fontsize=8.5)
ax.set_xlabel("number of triples")
ax.set_title("Triples per relation")
ax.grid(axis="y", visible=False)
plt.show()
savefig(fig, "fig_relations.png")
RESULTS["data"] = {"entities": len(entities), "triples": len(triples), "relations": int(triples.relation.nunique())}''')

    M(r'''A graph is easiest to understand by asking what it says about one node. The function below lists every fact
in which an entity takes part, as a head or as a tail.''')

    C(r'''def verbalize(h, r, t):
    return f"{label[h]}  --{r}-->  {label[t]}"


def facts_about(eid, df=triples):
    out = df[df.head_id == eid]
    inn = df[df.tail_id == eid]
    return [verbalize(*x) for x in out[["head_id", "relation", "tail_id"]].itertuples(index=False, name=None)], \
           [verbalize(*x) for x in inn[["head_id", "relation", "tail_id"]].itertuples(index=False, name=None)]


atlas = label_to_id["Project Atlas"]
outgoing, incoming = facts_about(atlas)
print("OUTGOING from Project Atlas")
print("\n".join("  " + s for s in outgoing))
print("\nINCOMING to Project Atlas")
print("\n".join("  " + s for s in incoming))''')

    M(r'''> **Watch out.** Direction matters. `Priya Nair --works_on--> Project Atlas` is not the same statement as
> `Project Atlas --works_on--> Priya Nair`. The schema in Lab 4 will make that explicit through *domain* and *range*.

### Exercise 1.1
1. Which relation has the largest number of *distinct* tail entities?
2. Which entity type has the highest average number of outgoing triples?
3. How many entities have no outgoing triples at all, and what are their types?
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 2
    M(r'''---
# Lab 2. The property graph model

**Goal.** Load the triples into a property graph and run simple traversals.

In a **property graph** both nodes and edges carry key-value properties. NetworkX's `MultiDiGraph` fits well: it
allows several edges between the same pair of nodes (distinguished by a key) and stores attributes on everything.
We use the relation name as the edge key and keep `source`, `confidence` and `valid_from` as edge properties.
''')
    C(r'''G = nx.MultiDiGraph()
for r in entities.itertuples():
    G.add_node(r.entity_id, label=r.label, type=r.type, group=r.group)
for r in triples.itertuples():
    G.add_edge(r.head_id, r.tail_id, key=r.relation, relation=r.relation,
               source=r.source, confidence=r.confidence, valid_from=r.valid_from)

print(G)


def objects(G, s, rel):
    """Tails reached from s along relation rel."""
    return [v for _, v, k in G.out_edges(s, keys=True) if k == rel]


def subjects(G, o, rel):
    """Heads that point to o along relation rel."""
    return [u for u, _, k in G.in_edges(o, keys=True) if k == rel]


team = subjects(G, atlas, "works_on")
lead = subjects(G, atlas, "leads")
print("Lead of Project Atlas  :", [label[x] for x in lead])
print("Works on Project Atlas :", [label[x] for x in team])
print("Technologies used      :", [label[x] for x in objects(G, atlas, "uses_tech")])
print("Client                 :", [label[x] for x in objects(G, atlas, "delivered_for")])''')

    M(r'''Edge properties let you slice the graph by provenance. For instance, how much of the graph rests on the
lower-confidence skills survey?''')
    C(r'''edge_df = pd.DataFrame([{"relation": d["relation"], "source": d["source"], "confidence": d["confidence"]}
                        for _, _, d in G.edges(data=True)])
display(edge_df.groupby("source").agg(edges=("relation", "size"), mean_conf=("confidence", "mean"),
                                      min_conf=("confidence", "min")).round(3))''')

    M(r'''### Degree distribution and hubs

Real knowledge graphs are rarely uniform. A few nodes (popular technologies, large departments) collect many edges
while most nodes have only a handful. Hubs matter for retrieval, because a hub connects to everything and can drown
a query in irrelevant neighbours.''')
    C(r'''deg = dict(G.degree())
hubs = (pd.DataFrame({"entity": [label[n] for n in deg], "group": [etype[n] for n in deg],
                      "degree": list(deg.values())})
        .sort_values("degree", ascending=False).head(10).reset_index(drop=True))
display(hubs)

cnt = Counter(deg.values())
xs, ys = zip(*sorted(cnt.items()))
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.scatter(xs, ys, s=22, color=ACCENT)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("degree (log scale)"); ax.set_ylabel("number of nodes (log scale)")
ax.set_title("Degree distribution")
plt.show()
savefig(fig, "fig_degree.png")
RESULTS["degree"] = {"max": int(max(deg.values())), "median": float(np.median(list(deg.values()))),
                     "top_hub": label[max(deg, key=deg.get)]}''')

    M(r'''### Looking at a neighbourhood

Plotting a whole graph rarely helps, but a two-hop neighbourhood around one node is often readable. We drop the
dense `has_skill`, `reports_to` and `mentors` edges so the picture shows the project, its people, its technologies
and its client.''')
    C(r'''skip = {"has_skill", "reports_to", "mentors"}
U = nx.Graph()
for u, v, k, d in G.edges(keys=True, data=True):
    if d["relation"] not in skip:
        U.add_edge(u, v)
ego = nx.ego_graph(U, atlas, radius=2)
print("nodes in the two-hop neighbourhood:", ego.number_of_nodes())

pos = nx.spring_layout(ego, seed=SEED, k=0.55)
fig, ax = plt.subplots(figsize=(8.2, 6.0))
nx.draw_networkx_edges(ego, pos, ax=ax, edge_color="#b9b9b4", width=0.8)
for grp, col in PALETTE.items():
    nodes = [n for n in ego if group(etype[n]) == grp]
    nx.draw_networkx_nodes(ego, pos, nodelist=nodes, node_color=col, node_size=[420 if n == atlas else 150 for n in nodes],
                           ax=ax, label=grp, edgecolors="white", linewidths=1.2)
nx.draw_networkx_labels(ego, pos, labels={n: label[n].replace("Project ", "") for n in ego}, font_size=6.2, ax=ax)
ax.legend(loc="lower left", fontsize=8, ncol=2, markerscale=0.8)
ax.set_title("Two-hop neighbourhood of Project Atlas")
ax.axis("off")
plt.show()
savefig(fig, "fig_subgraph.png")''')

    M(r'''### Exercise 2.1
Find **training gaps**: people who work on a project that uses Neo4j but who have no recorded skill in Neo4j.
Return their names and the project through which the gap arises.
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 3
    M(r'''---
# Lab 3. RDF, SPARQL and a little reasoning

**Goal.** Express the same graph in RDF, query it with SPARQL, and see why a schema layer produces new facts.

**RDF** is the W3C standard data model for triples. Every node and relation is identified by a URI, which makes
graphs from different sources mergeable. **SPARQL** is its query language: you write a pattern of triples with
variables, and the engine returns every way of matching it. **RDFS** adds a small vocabulary for the schema:
`rdfs:subClassOf`, `rdfs:subPropertyOf`, `rdfs:domain` and `rdfs:range`.
''')
    C(r'''EX = Namespace("http://meridian.example/kg/")
g = Graph()
g.bind("ex", EX)

for e in entities.itertuples():
    g.add((EX[e.entity_id], RDF.type, EX[e.type]))
    g.add((EX[e.entity_id], RDFS.label, Literal(e.label)))
for t in triples.itertuples():
    g.add((EX[t.head_id], EX[t.relation], EX[t.tail_id]))

# schema layer: class hierarchy and sub-properties
for cls, par in ontology["classes"].items():
    if par:
        g.add((EX[cls], RDFS.subClassOf, EX[par]))
for prop, spec in ontology["properties"].items():
    if spec.get("subPropertyOf"):
        g.add((EX[prop], RDFS.subPropertyOf, EX[spec["subPropertyOf"]]))

print("RDF triples:", len(g))


def sparql(q):
    res = g.query(q, initNs={"ex": EX, "rdfs": RDFS, "rdf": RDF})
    return pd.DataFrame([[None if v is None else str(v) for v in row] for row in res],
                        columns=[str(v) for v in res.vars])''')

    M(r'''A first query: which projects does Priya Nair work on, and for which clients? The `OPTIONAL` block keeps projects
that have no client recorded.''')
    C(r'''q1 = """
SELECT ?project ?client WHERE {
  ?p rdfs:label "Priya Nair" .
  ?p ex:works_on ?pr .
  ?pr rdfs:label ?project .
  OPTIONAL { ?pr ex:delivered_for ?c . ?c rdfs:label ?client }
}
"""
display(sparql(q1))''')

    M(r'''SPARQL can aggregate. Which technologies are held as a skill by the most people?''')
    C(r'''q2 = """
SELECT ?tech (COUNT(DISTINCT ?p) AS ?people) WHERE {
  ?p ex:has_skill ?t .
  ?t rdfs:label ?tech .
}
GROUP BY ?tech ORDER BY DESC(?people) LIMIT 8
"""
display(sparql(q2))''')

    M(r'''**Property paths** express traversals of unknown length. The `+` operator means "one or more steps", so
`ex:reports_to+` returns the whole management chain without writing a loop.''')
    C(r'''who = sorted(triples[triples.relation == "reports_to"].head_id)[7]
q3 = f"""
SELECT ?boss WHERE {{
  ex:{who} ex:reports_to+ ?b .
  ?b rdfs:label ?boss .
}}
"""
print("Management chain above", label[who])
display(sparql(q3))''')

    M(r'''### Why a schema layer matters: RDFS inference

Ask for every `ex:Technology`. The data only states that Python is a `Language`, PostgreSQL a `Database`, and so on.
Nothing says directly that they are technologies, so a literal query finds none.''')
    C(r'''q_tech = "SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE { ?x rdf:type ex:Technology }"
q_works = "SELECT (COUNT(*) AS ?n) WHERE { ?p ex:works_on ?pr }"
before = {"Technology instances": int(sparql(q_tech).n[0]), "works_on triples": int(sparql(q_works).n[0])}
print(before)''')

    M(r'''The remedy is to apply two RDFS rules until nothing new appears:

* **subclass rule:** if `x` has type `C` and `C` is a subclass of `D`, then `x` has type `D`;
* **subproperty rule:** if `x p y` holds and `p` is a subproperty of `q`, then `x q y` holds.

This is *forward chaining*, also called materialisation. We write it ourselves, because it is short and shows
exactly what a reasoner does.''')
    C(r'''def rdfs_closure(graph):
    """Return a new graph with subclass and subproperty consequences added (fixed point)."""
    out = Graph()
    for t in graph:
        out.add(t)
    while True:
        size = len(out)
        for c, d in [(s, o) for s, _, o in out.triples((None, RDFS.subClassOf, None))]:
            for x in list(out.subjects(RDF.type, c)):
                out.add((x, RDF.type, d))
        for p, q in [(s, o) for s, _, o in out.triples((None, RDFS.subPropertyOf, None))]:
            for s, _, o in list(out.triples((None, p, None))):
                out.add((s, q, o))
        if len(out) == size:
            return out


g_plain = g
g = rdfs_closure(g_plain)                 # sparql() queries whichever graph is bound to g
after = {"Technology instances": int(sparql(q_tech).n[0]), "works_on triples": int(sparql(q_works).n[0])}
summary = pd.DataFrame({"before": before, "after": after})
summary.loc["all RDF triples"] = [len(g_plain), len(g)]
display(summary)
RESULTS["rdfs"] = {"triples_before": len(g_plain), "triples_after": len(g),
                   "tech_before": before["Technology instances"], "tech_after": after["Technology instances"],
                   "works_on_before": before["works_on triples"], "works_on_after": after["works_on triples"]}''')

    M(r'''The jump in `works_on` triples comes entirely from `leads`: a project lead whose own `works_on` edge was left
out of the source data now appears on the team. Without the schema, a query for "everyone on Project Atlas" would
silently miss the lead.

> **Practical tip.** Materialise inference when queries are frequent and the data changes slowly. Compute it at query
> time (for example with property paths) when the data changes often or the closure would be huge.

> **Watch out.** RDFS `domain` and `range` apply to *every* use of a property, and several domains combine as an
> intersection. `located_in` applies to both departments and clients, which RDFS alone cannot say; OWL (or the
> explicit checks in Lab 4) is needed for unions. That is why the next lab validates against our own ontology file.

### Exercise 3.1
Write a SPARQL query that returns, for each department, the number of *distinct clients* reached through the
projects its members work on. Use `g_plain` or `g` as you prefer and explain whether the answer differs.
''')
    C(r'''# YOUR CODE HERE''')

    # ------------------------------------------------------------------ Lab 4
    M(r'''---
# Lab 4. Validating a graph against its ontology

**Goal.** Detect bad triples automatically, using only the schema.

Real graphs arrive dirty. A schema is more than documentation: it is a set of executable constraints. The file
`triples_dirty.csv` is the clean graph plus 90 injected errors of six kinds. Each injected row is labelled in
`error_type`, so we can measure how well our checks perform.

| Constraint | Meaning | Example violation |
|---|---|---|
| domain | the head must be an instance of the property's domain | a *Technology* that `works_on` a project |
| range | the tail must be an instance of the property's range | a *Project* that `uses_tech` a person |
| functional | at most one tail per head | a person who `works_in` two departments |
| irreflexive | no node may relate to itself | a person who `mentors` themselves |
| acyclic | no cycles in `reports_to` or `depends_on` | A reports to B and B reports to A |
| asymmetric (`mentors`) | mentoring in both directions is implausible | A mentors B and B mentors A |
''')
    C(r'''dirty = pd.read_csv(DATA / "triples_dirty.csv").fillna({"error_type": ""})
print(dirty.shape, "| injected errors:", int((dirty.error_type != "").sum()))
display(dirty[dirty.error_type != ""].groupby("error_type").size().rename("rows").to_frame())


def validate(df, ont):
    """Return {row index: set of rules that flagged it}."""
    flags = defaultdict(set)
    props = ont["properties"]

    for i, h, r, t in zip(df.index, df.head_id, df.relation, df.tail_id):
        spec = props[r]
        if not is_a(h, spec["domain"]):
            flags[i].add("domain")
        if not is_a(t, spec["range"]):
            flags[i].add("range")
        if spec.get("irreflexive") and h == t:
            flags[i].add("self_loop")

    # functional properties: keep the most confident edge per head, flag the others
    for rel, spec in props.items():
        if spec.get("functional"):
            sub = df[df.relation == rel]
            for _, grp in sub.groupby("head_id"):
                if grp.tail_id.nunique() > 1:
                    keep = grp.sort_values("confidence", ascending=False).index[0]
                    for i in grp.index:
                        if i != keep:
                            flags[i].add("functional")

    # cycles: repeatedly remove the weakest edge of a cycle until the relation is acyclic
    for rel, spec in props.items():
        if spec.get("transitive") and spec.get("irreflexive"):
            sub = df[(df.relation == rel) & (df.head_id != df.tail_id)]
            D = nx.DiGraph()
            for i, h, t, c in zip(sub.index, sub.head_id, sub.tail_id, sub.confidence):
                D.add_edge(h, t, row=i, conf=c)
            while True:
                try:
                    cyc = nx.find_cycle(D)
                except nx.NetworkXNoCycle:
                    break
                weakest = min(cyc, key=lambda e: D.edges[e[0], e[1]]["conf"])
                flags[D.edges[weakest[0], weakest[1]]["row"]].add("cycle")
                D.remove_edge(weakest[0], weakest[1])

    # mutual mentoring: flag the weaker of the two directions
    sub = df[df.relation == "mentors"]
    pair = {(h, t): (i, c) for i, h, t, c in zip(sub.index, sub.head_id, sub.tail_id, sub.confidence)}
    for (h, t), (i, c) in pair.items():
        if h != t and (t, h) in pair and c < pair[(t, h)][1]:
            flags[i].add("mutual")
    return flags


flags = validate(dirty, ontology)
gold = set(dirty.index[dirty.error_type != ""])
flagged = set(flags)
tp = len(gold & flagged)
precision, recall = tp / len(flagged), tp / len(gold)
print(f"flagged {len(flagged)} rows | precision {precision:.3f} | recall {recall:.3f}")''')

    M(r'''Which rule caught which kind of error? A cross-tabulation shows both the coverage and the overlap between rules.''')
    C(r'''rows = []
for i in dirty.index[dirty.error_type != ""]:
    rows.append({"error_type": dirty.error_type[i], "caught": i in flags,
                 "rules": ",".join(sorted(flags.get(i, [])))})
res = pd.DataFrame(rows)
per_type = res.groupby("error_type").agg(injected=("caught", "size"), caught=("caught", "sum"))
per_type["recall"] = (per_type.caught / per_type.injected).round(3)
display(per_type)

fp = [i for i in flagged - gold]
print(f"\nclean rows flagged by mistake: {len(fp)}")
if fp:
    display(dirty.loc[fp, ["head_id", "relation", "tail_id", "confidence"]].assign(rules=[",".join(sorted(flags[i])) for i in fp]))
RESULTS["validation"] = {"flagged": len(flagged), "precision": round(precision, 3), "recall": round(recall, 3),
                         "per_type_recall": per_type.recall.to_dict(), "false_positives": len(fp)}''')

    M(r'''Two lessons come out of this table.

1. **Local rules are cheap and precise.** Domain, range and self-loop checks look at one triple at a time and are
   almost always right.
2. **Structural rules localise an error but cannot always tell which edge is wrong.** A cycle or a functional
   violation involves at least two edges. Here we use provenance (the lower confidence edge loses). Where
   confidence is uninformative, such a rule would need a human review queue rather than automatic deletion.

> **Practical tip.** In production, run the same checks with SHACL shapes on every load and route violations to a
> reviewable queue. Do not silently repair; a violation is information about an upstream system.

### Exercise 4.1
Add a new constraint, **mentor and mentee must work in the same department**, and count its violations on the clean
graph and on the dirty graph. Is the constraint informative here?
''')
    C(r'''# YOUR CODE HERE''')
