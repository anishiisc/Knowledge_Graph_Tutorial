#!/usr/bin/env python3
"""Synthetic data generator for the Knowledge Graph tutorial.

Domain: Meridian Analytics, a fictional technology consultancy.
Everything is generated from a fixed seed, so the files are reproducible.

Usage:  python generate_data.py [output_dir]
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 2026
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)

# --------------------------------------------------------------------------
# 1. Static vocabularies
# --------------------------------------------------------------------------
CITIES = [
    ("cty_01", "Bengaluru", "India", ["Bangalore"]),
    ("cty_02", "Pune", "India", []),
    ("cty_03", "Hyderabad", "India", []),
    ("cty_04", "Singapore", "Singapore", []),
    ("cty_05", "London", "United Kingdom", []),
    ("cty_06", "Toronto", "Canada", []),
]
CITY_ID = {c[1]: c[0] for c in CITIES}

TECHS = [
    ("Python", "Language", []), ("Java", "Language", []), ("Go", "Language", ["Golang"]),
    ("Rust", "Language", []), ("SQL", "Language", []), ("TypeScript", "Language", ["TS"]),
    ("Scala", "Language", []),
    ("PostgreSQL", "Database", ["Postgres"]), ("MongoDB", "Database", ["Mongo"]),
    ("Neo4j", "Database", []), ("Redis", "Database", []),
    ("Cassandra", "Database", ["Apache Cassandra"]), ("ClickHouse", "Database", []),
    ("Elasticsearch", "Database", ["Elastic"]),
    ("Django", "Framework", []), ("FastAPI", "Framework", []),
    ("Spring Boot", "Framework", ["Spring"]), ("React", "Framework", ["ReactJS"]),
    ("Spark", "Framework", ["Apache Spark"]), ("Airflow", "Framework", ["Apache Airflow"]),
    ("Kafka", "Framework", ["Apache Kafka"]), ("dbt", "Framework", []),
    ("AWS S3", "CloudService", ["S3"]), ("AWS Lambda", "CloudService", ["Lambda"]),
    ("Azure Synapse", "CloudService", ["Synapse"]), ("BigQuery", "CloudService", ["Google BigQuery"]),
    ("Snowflake", "CloudService", []), ("Kubernetes", "CloudService", ["K8s"]),
    ("Terraform", "CloudService", []), ("Docker", "CloudService", []),
    ("PyTorch", "MLTool", []), ("TensorFlow", "MLTool", ["TF"]),
    ("scikit-learn", "MLTool", ["sklearn"]), ("MLflow", "MLTool", []),
    ("Hugging Face Transformers", "MLTool", ["HF Transformers"]),
    ("LangChain", "MLTool", []), ("XGBoost", "MLTool", []),
]
DEPS = [
    ("Django", "Python"), ("FastAPI", "Python"), ("Spring Boot", "Java"), ("PyTorch", "Python"),
    ("TensorFlow", "Python"), ("scikit-learn", "Python"), ("Spark", "Scala"), ("Scala", "Java"),
    ("Airflow", "Python"), ("dbt", "SQL"), ("LangChain", "Python"),
    ("Hugging Face Transformers", "PyTorch"), ("MLflow", "Python"), ("Kafka", "Java"),
    ("Kubernetes", "Docker"), ("Docker", "Go"), ("Terraform", "Go"), ("React", "TypeScript"),
    ("Elasticsearch", "Java"), ("Cassandra", "Java"), ("Neo4j", "Java"), ("XGBoost", "Python"),
]

# department: (city, size, focus technologies, favoured client industries, analyst share, researcher share)
DEPTS = {
    "Data Platform": ("Bengaluru", 16, ["Python", "SQL", "Spark", "Kafka", "Airflow", "Snowflake", "dbt", "PostgreSQL", "ClickHouse", "AWS S3"], ["Retail", "Logistics"], 0.20, 0.0),
    "Applied AI": ("Bengaluru", 18, ["Python", "PyTorch", "Hugging Face Transformers", "LangChain", "MLflow", "FastAPI", "Neo4j", "scikit-learn", "Elasticsearch"], ["Banking", "Healthcare"], 0.10, 0.0),
    "Cybersecurity": ("Pune", 12, ["Go", "Rust", "Python", "Elasticsearch", "Kafka", "Kubernetes", "Docker", "Terraform"], ["Banking", "Government"], 0.05, 0.0),
    "Cloud Infrastructure": ("Pune", 13, ["Kubernetes", "Docker", "Terraform", "AWS Lambda", "AWS S3", "Go", "Azure Synapse", "Redis"], ["Logistics", "Manufacturing"], 0.05, 0.0),
    "Finance Analytics": ("London", 14, ["Python", "SQL", "XGBoost", "scikit-learn", "PostgreSQL", "BigQuery", "Snowflake", "TensorFlow"], ["Banking", "Insurance"], 0.35, 0.0),
    "Customer Insights": ("Toronto", 14, ["Python", "SQL", "scikit-learn", "React", "TypeScript", "MongoDB", "BigQuery", "dbt"], ["Retail", "Media"], 0.35, 0.0),
    "Product Engineering": ("Hyderabad", 18, ["Java", "Spring Boot", "React", "TypeScript", "PostgreSQL", "Redis", "Docker", "Django", "Cassandra"], ["Insurance", "Healthcare", "Retail"], 0.05, 0.0),
    "Research": ("Singapore", 12, ["Python", "PyTorch", "TensorFlow", "Hugging Face Transformers", "Neo4j", "Scala", "Spark", "MLflow", "Rust"], ["Healthcare", "Government", "Media"], 0.0, 0.70),
}
PROJECTS_PER_DEPT = {"Data Platform": 6, "Applied AI": 7, "Cybersecurity": 5, "Cloud Infrastructure": 5,
                     "Finance Analytics": 6, "Customer Insights": 6, "Product Engineering": 6, "Research": 4}
EXEC_DEPT = "Executive Office"

CLIENTS = {
    "Retail": ["Verdant Foods", "Kestrel Retail Group", "Maple and Mint Stores"],
    "Banking": ["Halcyon Bank", "Northgate Capital", "Sterling Ridge Bank"],
    "Insurance": ["Ardent Assurance", "Pelican Mutual", "Cobalt Life"],
    "Healthcare": ["Lumen Health Network", "Aster Care Systems", "Willow Diagnostics"],
    "Logistics": ["Orbis Logistics", "Tidewater Freight", "Corvid Cargo"],
    "Manufacturing": ["Ferrous Dynamics", "Quill Industrial", "Granite Works"],
    "Government": ["Metro Transit Authority", "State Revenue Office", "Civic Registry Board"],
    "Media": ["Prism Broadcasting", "Echo Street Media", "Lantern Studios"],
}
PROJECT_NAMES = ["Atlas", "Orion", "Vega", "Lyra", "Hydra", "Apollo", "Helios", "Selene", "Nova", "Pulsar",
                 "Quasar", "Zenith", "Aurora", "Cygnus", "Draco", "Elara", "Fornax", "Gemini", "Halley",
                 "Kepler", "Mira", "Nebula", "Oberon", "Pegasus", "Rigel", "Sirius", "Titan", "Umbra",
                 "Vesta", "Altair", "Bellatrix", "Callisto", "Deneb", "Europa", "Ganymede", "Hyperion",
                 "Janus", "Tethys", "Phoebe", "Rhea", "Dione", "Iapetus", "Triton", "Proteus", "Larissa"]

FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "Arjun", "Kavya", "Rahul", "Sneha",
         "Aditya", "Isha", "Karan", "Divya", "Nikhil", "Pooja", "Siddharth", "Neha", "Amit", "Riya",
         "Sanjay", "Tara", "Manish", "Lakshmi", "Deepak", "Anjali", "Imran", "Farah", "Wei", "Mei",
         "Hiroshi", "Yuki", "Daniel", "Sarah", "Michael", "Emily", "James", "Olivia", "Thomas", "Hannah",
         "Lucas", "Sofia", "Mateo", "Elena", "Omar", "Layla", "Chen", "Linh", "Ravi", "Shreya"]
LAST = ["Nair", "Iyer", "Sharma", "Verma", "Gupta", "Reddy", "Menon", "Das", "Banerjee", "Chatterjee",
        "Kulkarni", "Patel", "Singh", "Khan", "Joshi", "Rao", "Bose", "Mehta", "Kapoor", "Malhotra",
        "Tan", "Lim", "Wong", "Nakamura", "Sato", "Smith", "Johnson", "Clarke", "Evans", "Taylor",
        "Walker", "Morgan", "Fischer", "Rossi", "Costa", "Haddad", "Nguyen", "Park", "Kim", "Ito",
        "Bhat", "Pillai", "Shetty", "Ghosh", "Sen", "Roy", "Dutta", "Saha", "Mukherjee", "Basu"]
# Deliberately confusable pairs: same initial and same surname.
FORCED = [("Priya", "Nair"), ("Pooja", "Nair"), ("Rahul", "Verma"), ("Rohan", "Verma"),
          ("Sarah", "Smith"), ("Sneha", "Smith"), ("Amit", "Khan"), ("Aarav", "Khan")]

# --------------------------------------------------------------------------
# 2. Entities
# --------------------------------------------------------------------------
entities: list[dict] = []
aliases: list[dict] = []


def add_entity(eid, label, etype, **attrs):
    row = {"entity_id": eid, "label": label, "type": etype}
    row.update(attrs)
    entities.append(row)


def add_alias(eid, alias):
    aliases.append({"entity_id": eid, "alias": alias})


for cid, name, country, al in CITIES:
    add_entity(cid, name, "City", country=country)
    for a in al:
        add_alias(cid, a)

tech_id = {}
for i, (name, cls, al) in enumerate(TECHS, 1):
    eid = f"tec_{i:02d}"
    tech_id[name] = eid
    add_entity(eid, name, cls)
    for a in al:
        add_alias(eid, a)

dept_id = {}
for i, (name, spec) in enumerate(DEPTS.items(), 1):
    eid = f"dep_{i:02d}"
    dept_id[name] = eid
    add_entity(eid, name, "Department")
    add_alias(eid, f"{name} team")
dept_id[EXEC_DEPT] = "dep_09"
add_entity("dep_09", EXEC_DEPT, "Department")

client_id, client_industry = {}, {}
ci = 0
for industry, names in CLIENTS.items():
    for nm in names:
        ci += 1
        eid = f"cli_{ci:02d}"
        client_id[nm] = eid
        client_industry[eid] = industry
        add_entity(eid, nm, "Client", industry=industry)
first_tokens = [n.split()[0] for n in client_id]
for nm, eid in client_id.items():
    tok = nm.split()[0]
    if first_tokens.count(tok) == 1 and len(tok) > 4:
        add_alias(eid, tok)

# ---- people ----
pool = [(f, l) for f in FIRST for l in LAST if (f, l) not in FORCED]
rng.shuffle(pool)
names = list(FORCED) + pool[: 120 - len(FORCED)]
rng.shuffle(names)

slots = []  # (dept, role)
slots += [(EXEC_DEPT, "Executive")] * 3
for d, (city, size, focus, inds, an, rs) in DEPTS.items():
    n_mgr = 2 if size >= 14 else 1
    n_an = round(size * an)
    n_rs = round(size * rs)
    roles = ["Manager"] * n_mgr + ["Analyst"] * n_an + ["Researcher"] * n_rs
    roles += ["Engineer"] * (size - len(roles))
    slots += [(d, r) for r in roles]
assert len(slots) == 120, len(slots)
rng.shuffle(slots)
# make sure executives come first so the CEO/CTO/COO ids are stable
slots.sort(key=lambda s: 0 if s[1] == "Executive" else 1)

people = []
for i, ((first, last), (dept, role)) in enumerate(zip(names, slots), 1):
    eid = f"per_{i:03d}"
    if role == "Executive":
        seniority, joined = 6, rng.randint(2008, 2014)
    elif role == "Manager":
        seniority, joined = rng.choice([4, 5]), rng.randint(2010, 2019)
    else:
        seniority, joined = rng.choice([1, 2, 2, 3, 3, 4]), rng.randint(2014, 2025)
    label = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@meridian.example"
    add_entity(eid, label, role, email=email, joined_year=joined, seniority=f"L{seniority}")
    people.append({"id": eid, "first": first, "last": last, "label": label, "dept": dept,
                   "role": role, "joined": joined, "email": email})
    add_alias(eid, f"{first[0]}. {last}")

by_dept = defaultdict(list)
for p in people:
    by_dept[p["dept"]].append(p)
execs = by_dept[EXEC_DEPT]
ceo, cto, coo = execs[0], execs[1], execs[2]

# ---- projects ----
projects = []
names_iter = iter(PROJECT_NAMES)
pi = 0
for d, cnt in PROJECTS_PER_DEPT.items():
    for _ in range(cnt):
        pi += 1
        nm = next(names_iter)
        eid = f"prj_{pi:02d}"
        add_entity(eid, f"Project {nm}", "Project",
                   start_year=rng.randint(2019, 2026),
                   status=rng.choices(["Active", "Completed", "Planned", "On Hold"], [5, 4, 1, 1])[0],
                   budget_band=rng.choice(["S", "M", "L"]))
        add_alias(eid, nm)
        add_alias(eid, nm.upper())
        projects.append({"id": eid, "name": nm, "dept": d})

# --------------------------------------------------------------------------
# 3. Triples
# --------------------------------------------------------------------------
triples: list[dict] = []
seen = set()


def add_triple(h, r, t, source, lo=0.9, hi=1.0):
    if (h, r, t) in seen:
        return False
    seen.add((h, r, t))
    triples.append({"head_id": h, "relation": r, "tail_id": t, "source": source,
                    "confidence": round(rng.uniform(lo, hi), 2), "valid_from": rng.randint(2019, 2026)})
    return True


# located_in
for d, (city, *_rest) in DEPTS.items():
    add_triple(dept_id[d], "located_in", CITY_ID[city], "Master_Data", 0.99, 1.0)
add_triple(dept_id[EXEC_DEPT], "located_in", CITY_ID["Bengaluru"], "Master_Data", 0.99, 1.0)
for nm, eid in client_id.items():
    add_triple(eid, "located_in", rng.choice(list(CITY_ID.values())), "Master_Data", 0.95, 1.0)

# depends_on
for a, b in DEPS:
    add_triple(tech_id[a], "depends_on", tech_id[b], "Architecture_Wiki", 0.9, 1.0)

# works_in and reports_to
for p in people:
    add_triple(p["id"], "works_in", dept_id[p["dept"]], "HR_System", 0.98, 1.0)
for d in DEPTS:
    members = by_dept[d]
    mgrs = [p for p in members if p["role"] == "Manager"]
    head = mgrs[0]
    boss = cto if d in ("Data Platform", "Applied AI", "Cybersecurity", "Cloud Infrastructure", "Research") else coo
    add_triple(head["id"], "reports_to", boss["id"], "HR_System", 0.98, 1.0)
    for p in members:
        if p is head:
            continue
        target = head if p["role"] == "Manager" or rng.random() < 0.6 else rng.choice(mgrs)
        add_triple(p["id"], "reports_to", target["id"], "HR_System", 0.98, 1.0)
add_triple(cto["id"], "reports_to", ceo["id"], "HR_System", 0.99, 1.0)
add_triple(coo["id"], "reports_to", ceo["id"], "HR_System", 0.99, 1.0)

# projects: lead, team, technology, client
dept_focus = {d: spec[2] for d, spec in DEPTS.items()}
dept_inds = {d: spec[3] for d, spec in DEPTS.items()}
all_tech_names = [t[0] for t in TECHS]
non_exec = [p for p in people if p["role"] != "Executive"]
project_team, project_techs = {}, {}
for pr in projects:
    d = pr["dept"]
    members = [p for p in by_dept[d]]
    leaders = [p for p in members if p["role"] == "Manager" or (p["joined"] <= 2018)]
    lead = rng.choice(leaders)
    add_triple(lead["id"], "leads", pr["id"], "Project_Registry")
    team = {lead["id"]}
    size = rng.randint(3, 7)
    while len(team) < size:
        if rng.random() < 0.8:
            cand = rng.choice(members)
        else:
            cand = rng.choice(non_exec)
        team.add(cand["id"])
    project_team[pr["id"]] = team
    for pid in sorted(team):
        if pid == lead["id"] and rng.random() < 0.5:
            continue  # leader's works_on is left implicit half of the time (RDFS inference demo)
        add_triple(pid, "works_on", pr["id"], "Project_Registry")
    # technologies
    k = rng.randint(3, 6)
    chosen = set()
    while len(chosen) < k:
        pool_t = dept_focus[d] if rng.random() < 0.75 else all_tech_names
        chosen.add(rng.choice(pool_t))
    project_techs[pr["id"]] = [tech_id[t] for t in sorted(chosen)]
    for t in sorted(chosen):
        add_triple(pr["id"], "uses_tech", tech_id[t], "Project_Registry")
    # client
    ind = rng.choice(dept_inds[d]) if rng.random() < 0.8 else rng.choice(list(CLIENTS))
    cl = rng.choice(CLIENTS[ind])
    add_triple(pr["id"], "delivered_for", client_id[cl], "Project_Registry")

# make sure every non-executive works on at least one project
for p in non_exec:
    if not any((p["id"] in tm) for tm in project_team.values()):
        pr = rng.choice([x for x in projects if x["dept"] == p["dept"]])
        project_team[pr["id"]].add(p["id"])
        add_triple(p["id"], "works_on", pr["id"], "Project_Registry")

# has_skill
person_projects = defaultdict(set)
for pid_, team in project_team.items():
    for m in team:
        person_projects[m].add(pid_)
for p in non_exec:
    pool_own = sorted({t for pr in person_projects[p["id"]] for t in project_techs[pr]})
    k = rng.randint(3, 6)
    got = set()
    tries = 0
    while len(got) < k and tries < 50:
        tries += 1
        u = rng.random()
        if u < 0.60 and pool_own:
            got.add(rng.choice(pool_own))
        elif u < 0.85:
            got.add(tech_id[rng.choice(dept_focus[p["dept"]])])
        else:
            got.add(tech_id[rng.choice(all_tech_names)])
    for t in sorted(got):
        add_triple(p["id"], "has_skill", t, "Skills_Survey", 0.6, 0.95)

# mentors
for d in DEPTS:
    members = by_dept[d]
    juniors = [p for p in members if p["role"] in ("Engineer", "Analyst", "Researcher")]
    rng.shuffle(juniors)
    load = defaultdict(int)
    for jr in juniors[: max(2, len(juniors) // 3)]:
        seniors = [p for p in members if p["joined"] <= jr["joined"] - 5 and load[p["id"]] < 3]
        if seniors:
            m = rng.choice(seniors)
            load[m["id"]] += 1
            add_triple(m["id"], "mentors", jr["id"], "Mentoring_Program", 0.85, 1.0)

triples_df = pd.DataFrame(triples)

# --------------------------------------------------------------------------
# 4. Ontology
# --------------------------------------------------------------------------
ontology = {
    "namespace": "http://meridian.example/kg/",
    "classes": {
        "Person": None, "Executive": "Person", "Manager": "Person", "Engineer": "Person",
        "Analyst": "Person", "Researcher": "Person",
        "Technology": None, "Language": "Technology", "Database": "Technology",
        "Framework": "Technology", "CloudService": "Technology", "MLTool": "Technology",
        "Department": None, "Project": None, "Client": None, "City": None,
    },
    "properties": {
        "works_in": {"domain": ["Person"], "range": ["Department"], "functional": True, "inverse": "has_member"},
        "reports_to": {"domain": ["Person"], "range": ["Person"], "functional": True, "inverse": "manages", "transitive": True, "irreflexive": True},
        "works_on": {"domain": ["Person"], "range": ["Project"], "inverse": "staffed_by"},
        "leads": {"domain": ["Person"], "range": ["Project"], "subPropertyOf": "works_on", "inverse": "led_by"},
        "uses_tech": {"domain": ["Project"], "range": ["Technology"], "inverse": "used_by"},
        "delivered_for": {"domain": ["Project"], "range": ["Client"], "functional": True, "inverse": "has_project"},
        "has_skill": {"domain": ["Person"], "range": ["Technology"], "inverse": "skill_of"},
        "located_in": {"domain": ["Department", "Client"], "range": ["City"], "functional": True, "inverse": "hosts"},
        "depends_on": {"domain": ["Technology"], "range": ["Technology"], "transitive": True, "irreflexive": True, "inverse": "required_by"},
        "mentors": {"domain": ["Person"], "range": ["Person"], "irreflexive": True, "inverse": "mentored_by"},
    },
}

# sanity check: the clean graph must satisfy the ontology
ent_df = pd.DataFrame(entities)
etype = dict(zip(ent_df.entity_id, ent_df.type))


def ancestors(c):
    out = {c}
    while ontology["classes"].get(c):
        c = ontology["classes"][c]
        out.add(c)
    return out


for _, r in triples_df.iterrows():
    prop = ontology["properties"][r.relation]
    assert set(prop["domain"]) & ancestors(etype[r.head_id]), (r.head_id, r.relation)
    assert set(prop["range"]) & ancestors(etype[r.tail_id]), (r.tail_id, r.relation)

# --------------------------------------------------------------------------
# 5. Aliases, gazetteer
# --------------------------------------------------------------------------
alias_df = pd.DataFrame(aliases)
label_of = dict(zip(ent_df.entity_id, ent_df.label))
surface_count = defaultdict(set)
for eid, lab in label_of.items():
    surface_count[lab].add(eid)
for _, a in alias_df.iterrows():
    surface_count[a.alias].add(a.entity_id)
alias_df["ambiguous"] = alias_df.alias.map(lambda s: len(surface_count[s]) > 1)
alias_df["in_gazetteer"] = [(not amb) and rng.random() < 0.8 for amb in alias_df.ambiguous]
alias_df.to_csv(OUT / "aliases.csv", index=False)

gaz_rows = [{"surface": lab, "entity_id": eid, "type": etype[eid]} for eid, lab in label_of.items()
            if len(surface_count[lab]) == 1]
for _, a in alias_df[alias_df.in_gazetteer].iterrows():
    gaz_rows.append({"surface": a.alias, "entity_id": a.entity_id, "type": etype[a.entity_id]})
pd.DataFrame(gaz_rows).to_csv(OUT / "gazetteer.csv", index=False)

# --------------------------------------------------------------------------
# 6. Documents with gold triples
# --------------------------------------------------------------------------
alias_by_entity = defaultdict(list)
for _, a in alias_df.iterrows():
    alias_by_entity[a.entity_id].append(a.alias)


def surface(eid):
    """Pick a surface form for an entity: usually the label, sometimes an alias."""
    cands = alias_by_entity.get(eid, [])
    if etype[eid] in ("Executive", "Manager", "Engineer", "Analyst", "Researcher"):
        initial = [a for a in cands if ". " in a]
        return rng.choice(initial) if initial and rng.random() < 0.30 else label_of[eid]
    # drop the "<Dept> team" alias; it would double up with the word "team" in some templates
    cands = [a for a in cands if not a.endswith(" team")]
    return rng.choice(cands) if cands and rng.random() < 0.30 else label_of[eid]


def join_list(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


out_by_rel = defaultdict(lambda: defaultdict(list))
for _, r in triples_df.iterrows():
    out_by_rel[r.relation][r.head_id].append(r.tail_id)
all_triples = list(zip(triples_df.head_id, triples_df.relation, triples_df.tail_id))
by_rel_triples = defaultdict(list)
for h, r, t in all_triples:
    by_rel_triples[r].append((h, r, t))

DISTRACTORS = [
    "The quarterly review will be held next week.", "Budget approvals are pending for the next phase.",
    "Several action items were carried over from the previous meeting.",
    "A risk register update is due before the end of the month.", "The steering committee asked for a clearer timeline.",
    "Travel plans for the offsite are still being finalised.", "Documentation for the release candidate is incomplete.",
]


def make_sentence():
    """Return (text, [gold triples as (h, r, t, h_surface, t_surface)])."""
    u = rng.random()
    if u < 0.07:
        return rng.choice(DISTRACTORS), []
    if u < 0.10:  # negation: looks like a relation but is not asserted
        h, r, t = rng.choice(by_rel_triples["works_on"])
        return f"{surface(h)} no longer works on {surface(t)}.", []
    if u < 0.14:  # mention without relation
        p = rng.choice([x for x in people if x["role"] != "Executive"])["id"]
        t = rng.choice(list(tech_id.values()))
        return f"{surface(p)} attended the {surface(t)} workshop.", []
    rel = rng.choices(
        ["works_on", "leads", "uses_tech", "delivered_for", "reports_to", "works_in", "located_in",
         "depends_on", "has_skill", "mentors"],
        [14, 8, 14, 7, 10, 7, 6, 8, 14, 6])[0]
    h, _, t = rng.choice(by_rel_triples[rel])
    hs, ts = surface(h), surface(t)
    g = lambda hh, rr, tt, a, b: (hh, rr, tt, a, b)
    if rel == "works_on":
        tpl = rng.choice(["{h} works on {t}.", "{h} contributes to {t}.", "{h} is part of the {t} team."])
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "leads":
        if rng.random() < 0.4 and out_by_rel["delivered_for"].get(t):
            c = out_by_rel["delivered_for"][t][0]
            cs = surface(c)
            return f"{hs} leads {ts}, which is delivered for {cs}.", [g(h, "leads", t, hs, ts), g(t, "delivered_for", c, ts, cs)]
        tpl = rng.choice(["{h} leads {t}.", "{h} is the lead for {t}."])
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "uses_tech":
        techs = out_by_rel["uses_tech"][h]
        pick = rng.sample(techs, min(len(techs), rng.randint(1, 3)))
        ps = [surface(x) for x in pick]
        tpl = rng.choice(["{h} uses {l}.", "{h} is built with {l}.", "{h} relies on {l}."])
        return tpl.format(h=hs, l=join_list(ps)), [g(h, rel, x, hs, s) for x, s in zip(pick, ps)]
    if rel == "delivered_for":
        tpl = rng.choice(["{h} is delivered for {t}.", "{h} is being built for {t}."])
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "reports_to":
        if rng.random() < 0.35:
            return f"{ts} manages {hs}.", [g(h, rel, t, hs, ts)]
        tpl = rng.choice(["{h} reports to {t}.", "{h} is managed by {t}."])
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "works_in":
        tpl = rng.choice(["{h} works in the {t} department.", "{h} is a member of the {t} department."])
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "located_in":
        tpl = "{h} is based in {t}." if etype[h] == "Department" else "{h} is headquartered in {t}."
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "depends_on":
        tpl = rng.choice(["{h} depends on {t}.", "{h} is built on {t}."])
        return tpl.format(h=hs, t=ts), [g(h, rel, t, hs, ts)]
    if rel == "has_skill":
        skills = out_by_rel["has_skill"][h]
        pick = rng.sample(skills, min(len(skills), rng.randint(1, 3)))
        ps = [surface(x) for x in pick]
        tpl = rng.choice(["{h} has strong experience with {l}.", "{h} is skilled in {l}."])
        return tpl.format(h=hs, l=join_list(ps)), [g(h, rel, x, hs, s) for x, s in zip(pick, ps)]
    if rel == "mentors":
        if rng.random() < 0.4:
            return f"{ts} is mentored by {hs}.", [g(h, rel, t, hs, ts)]
        return f"{hs} mentors {ts}.", [g(h, rel, t, hs, ts)]
    raise ValueError(rel)


DOC_TYPES = ["Status Report", "Wiki Page", "Meeting Notes", "Email Digest"]
doc_rows, sent_rows, gold_rows = [], [], []
for i in range(1, 161):
    doc_id = f"doc_{i:03d}"
    dtype = rng.choice(DOC_TYPES)
    author = rng.choice([p for p in people if p["role"] != "Executive"])["id"]
    date = f"20{rng.choice([25, 26])}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    doc_rows.append({"doc_id": doc_id, "title": f"{dtype} {i:03d}", "doc_type": dtype, "date": date, "author_id": author})
    for s in range(rng.randint(3, 6)):
        text, gold = make_sentence()
        sent_rows.append({"doc_id": doc_id, "sent_idx": s, "text": text})
        for (h, r, t, hs, ts) in gold:
            gold_rows.append({"doc_id": doc_id, "sent_idx": s, "head_id": h, "relation": r, "tail_id": t,
                              "head_surface": hs, "tail_surface": ts})
pd.DataFrame(doc_rows).to_csv(OUT / "documents.csv", index=False)
pd.DataFrame(sent_rows).to_csv(OUT / "doc_sentences.csv", index=False)
pd.DataFrame(gold_rows).to_csv(OUT / "doc_gold_triples.csv", index=False)

# --------------------------------------------------------------------------
# 7. Raw records for entity resolution
# --------------------------------------------------------------------------
CITY_VARIANTS = {"Bengaluru": ["Bengaluru", "Bangalore", "BLR"], "Pune": ["Pune", "PNQ"],
                 "Hyderabad": ["Hyderabad", "HYD"], "Singapore": ["Singapore", "SG"],
                 "London": ["London", "LDN"], "Toronto": ["Toronto", "YYZ"]}
dept_city = {d: DEPTS[d][0] for d in DEPTS}
dept_city[EXEC_DEPT] = "Bengaluru"


def typo(s):
    if len(s) < 5 or rng.random() > 0.5:
        return s
    i = rng.randint(1, len(s) - 3)
    kind = rng.choice(["swap", "drop", "dup"])
    if kind == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == "drop":
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


rec_rows = []


def add_rec(source, etype_, name, email, city, gold):
    rec_rows.append({"record_id": f"rec_{len(rec_rows) + 1:04d}", "source_system": source, "entity_type": etype_,
                     "name_raw": name, "email_raw": email, "city_raw": city, "gold_entity_id": gold})


for p in people:
    city = dept_city[p["dept"]]
    f, l = p["first"], p["last"]
    add_rec("HR", "Person", p["label"], p["email"], rng.choice(CITY_VARIANTS[city][:1]), p["id"])
    if rng.random() < 0.70:
        add_rec("Jira", "Person", f"{l}, {f}", p["email"] if rng.random() < 0.8 else "", rng.choice(CITY_VARIANTS[city]), p["id"])
    if rng.random() < 0.50:
        add_rec("Wiki", "Person", rng.choice([f"{f[0]}. {l}", f"{f} {l[0]}.", f"{f} {l}"]), "", rng.choice(CITY_VARIANTS[city]), p["id"])
    if rng.random() < 0.40:
        nm = rng.choice([f"{f} {l}".upper(), f"{f}  {l}", typo(f"{f} {l}")])
        mail = p["email"].replace("meridian.example", "meridian-analytics.example") if rng.random() < 0.6 else ""
        add_rec("Directory", "Person", nm, mail, rng.choice(CITY_VARIANTS[city]), p["id"])

client_city = {}
for _, r in triples_df[triples_df.relation == "located_in"].iterrows():
    if r.head_id.startswith("cli_"):
        client_city[r.head_id] = label_of[r.tail_id]
for nm, eid in client_id.items():
    domain = "".join(ch for ch in nm.lower() if ch.isalnum()) + ".example"
    city = client_city[eid]
    add_rec("CRM", "Client", nm, domain, city, eid)
    suffix = rng.choice([" Ltd", " Limited", " Inc", ""])
    add_rec("Contracts", "Client", (nm + suffix).upper() if rng.random() < 0.4 else nm + suffix, domain, rng.choice(CITY_VARIANTS[city]), eid)
    add_rec("Finance", "Client", typo(nm) if rng.random() < 0.6 else nm.split()[0], "", "", eid)
pd.DataFrame(rec_rows).to_csv(OUT / "raw_records.csv", index=False)

# --------------------------------------------------------------------------
# 8. Dirty triples (constraint violations injected)
# --------------------------------------------------------------------------
dirty = triples_df.copy()
dirty["error_type"] = ""
inj = []


def inject(h, r, t, kind):
    src = rng.choice(["HR_System", "Project_Registry", "Skills_Survey", "Architecture_Wiki"])
    inj.append({"head_id": h, "relation": r, "tail_id": t, "source": src,
                "confidence": round(rng.uniform(0.6, 0.98), 2), "valid_from": rng.randint(2019, 2026),
                "error_type": kind})


persons = [p["id"] for p in people]
others = [e for e in etype if etype[e] in ("Department", "Client", "Technology", "City", "Project")]
techs_ids = list(tech_id.values())
proj_ids = [p["id"] for p in projects]
dept_ids = [dept_id[d] for d in DEPTS]
# (a) domain violations
for _ in range(20):
    h = rng.choice([e for e in others if etype[e] in ("Department", "Client", "Technology")])
    inject(h, rng.choice(["works_on", "has_skill", "mentors"]), rng.choice(proj_ids if rng.random() < .5 else techs_ids), "domain")
# (b) range violations
for _ in range(20):
    inject(rng.choice(proj_ids), "uses_tech", rng.choice(persons), "range")
for _ in range(10):
    inject(rng.choice(persons), "works_in", rng.choice(techs_ids), "range")
# (c) functional property violations: a second works_in for a person
works_in = {r.head_id: r.tail_id for _, r in triples_df[triples_df.relation == "works_in"].iterrows()}
for pid in rng.sample([p for p in persons if p not in (ceo["id"], cto["id"], coo["id"])], 20):
    new = rng.choice([d for d in dept_ids if d != works_in[pid]])
    inject(pid, "works_in", new, "functional")
# (d) cycles in reports_to
rt = triples_df[triples_df.relation == "reports_to"]
for _, r in rt.sample(8, random_state=SEED).iterrows():
    inject(r.tail_id, "reports_to", r.head_id, "cycle")
# (e) self loops
for pid in rng.sample(persons, 6):
    inject(pid, "mentors", pid, "self_loop")
# (f) mutual mentoring
for _, r in triples_df[triples_df.relation == "mentors"].sample(6, random_state=SEED).iterrows():
    inject(r.tail_id, "mentors", r.head_id, "mutual")
dirty = pd.concat([dirty, pd.DataFrame(inj)], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
dirty.to_csv(OUT / "triples_dirty.csv", index=False)

# --------------------------------------------------------------------------
# 9. Question set for multi-hop retrieval
# --------------------------------------------------------------------------
fwd = defaultdict(lambda: defaultdict(set))
bwd = defaultdict(lambda: defaultdict(set))
for h, r, t in all_triples:
    fwd[r][h].add(t)
    bwd[r][t].add(h)
# implicit inference used by the questions: leads implies works_on
for h, t in [(h, t) for h, r, t in all_triples if r == "leads"]:
    fwd["works_on"][h].add(t)
    bwd["works_on"][t].add(h)


def run_path(start, path):
    cur = {start}
    for step in path.split(","):
        trans = step.endswith("+>")
        rel = step.rstrip("<>+")
        nxt = set()
        if step.endswith("<"):
            for e in cur:
                nxt |= bwd[rel][e]
        elif trans:
            frontier = set(cur)
            while frontier:
                new = set()
                for e in frontier:
                    new |= fwd[rel][e] - nxt
                nxt |= new
                frontier = new
        else:
            for e in cur:
                nxt |= fwd[rel][e]
        cur = nxt
    cur.discard(start)
    return cur


QTEMPLATES = [
    ("Who leads {x}?", "leads<", "Project", "Person"),
    ("Which client is {x} delivered for?", "delivered_for>", "Project", "Client"),
    ("Which technologies does {x} use?", "uses_tech>", "Project", "Technology"),
    ("Who reports directly to {x}?", "reports_to<", "Manager", "Person"),
    ("Which technologies are used by the projects {x} works on?", "works_on>,uses_tech>", "Person", "Technology"),
    ("Which clients has {x} worked for?", "works_on>,delivered_for>", "Person", "Client"),
    ("In which cities are the clients of the projects {x} works on based?", "works_on>,delivered_for>,located_in>", "Person", "City"),
    ("Who works in the same department as {x}?", "works_in>,works_in<", "Person", "Person"),
    ("Which people have a skill in a technology used by {x}?", "uses_tech>,has_skill<", "Project", "Person"),
    ("What does {x} depend on, directly or indirectly?", "depends_on+>", "Technology", "Technology"),
    ("Who are the colleagues of {x} on shared projects?", "works_on>,works_on<", "Person", "Person"),
    ("Who is in the management chain above {x}?", "reports_to+>", "Person", "Person"),
]
q_rows = []
qn = 0
for tpl, path, start_type, ans_type in QTEMPLATES:
    pool_s = [e for e in etype if start_type in ancestors(etype[e])]
    rng.shuffle(pool_s)
    made = 0
    for s in pool_s:
        ans = run_path(s, path)
        if not ans:
            continue
        if len(ans) > 40:
            continue
        qn += 1
        made += 1
        q_rows.append({"qid": f"q_{qn:03d}", "question": tpl.format(x=label_of[s]), "start_entity_id": s, "path": path,
                       "hops": len(path.split(",")), "answer_type": ans_type, "answers": ";".join(sorted(ans))})
        if made == 5:
            break
pd.DataFrame(q_rows).to_csv(OUT / "questions.csv", index=False)

# --------------------------------------------------------------------------
# 10. Write core tables
# --------------------------------------------------------------------------
ent_df = pd.DataFrame(entities)
ent_df.to_csv(OUT / "entities.csv", index=False)
triples_df.to_csv(OUT / "triples.csv", index=False)
(OUT / "ontology.json").write_text(json.dumps(ontology, indent=2))

print("entities      :", len(ent_df), dict(ent_df.type.value_counts()))
print("triples       :", len(triples_df), dict(triples_df.relation.value_counts()))
print("aliases       :", len(alias_df), "ambiguous:", int(alias_df.ambiguous.sum()))
print("gazetteer     :", len(gaz_rows))
print("documents     :", len(doc_rows), "sentences:", len(sent_rows), "gold triples:", len(gold_rows))
print("raw records   :", len(rec_rows))
print("dirty triples :", len(dirty), "injected:", len(inj), dict(pd.DataFrame(inj).error_type.value_counts()))
print("questions     :", len(q_rows), dict(pd.DataFrame(q_rows).hops.value_counts()))
