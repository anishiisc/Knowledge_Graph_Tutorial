# Meridian Analytics: synthetic knowledge graph dataset

All files are synthetic. Names of people, clients and projects are invented, and any resemblance to real
individuals or organisations is coincidental. Technology names (Python, Kafka, and so on) are real product
names used only as vocabulary. The files are produced by `scripts/generate_data.py` with a fixed seed (2026),
so regenerating them gives byte-identical output.

## Files

| File | Rows | Purpose |
|---|---|---|
| `entities.csv` | 241 | One row per node: `entity_id`, `label`, `type`, plus type-specific attributes. |
| `triples.csv` | 1,332 | Clean edge list: `head_id`, `relation`, `tail_id`, with edge properties `source`, `confidence`, `valid_from`. |
| `ontology.json` | 1 | Class hierarchy and property constraints (domain, range, functional, transitive, subPropertyOf, inverse). |
| `aliases.csv` | 261 | Alternative surface forms. `ambiguous` marks aliases shared by several entities; `in_gazetteer` marks those curated into the gazetteer. |
| `gazetteer.csv` | 434 | Dictionary of unambiguous surface forms used for mention detection. |
| `documents.csv` | 160 | Document metadata (type, date, author). |
| `doc_sentences.csv` | 739 | One sentence per row, keyed by `doc_id` and `sent_idx`. |
| `doc_gold_triples.csv` | 859 | Gold triples expressed by the sentences, for scoring text-to-triples extraction. |
| `raw_records.csv` | 374 | Noisy records of people and clients from several source systems, with `gold_entity_id` for scoring entity resolution. |
| `triples_dirty.csv` | 1,422 | `triples.csv` plus 90 injected constraint violations; column `error_type` is empty for clean rows. |
| `questions.csv` | 60 | Natural-language questions with a path specification and gold answers, for multi-hop retrieval. |

## Entity types

Person subclasses: `Executive`, `Manager`, `Engineer`, `Analyst`, `Researcher`.
Technology subclasses: `Language`, `Database`, `Framework`, `CloudService`, `MLTool`.
Other classes: `Department`, `Project`, `Client`, `City`.

## Relations

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `works_in` | Person | Department | functional |
| `reports_to` | Person | Person | functional, transitive, acyclic |
| `works_on` | Person | Project | leaders' `works_on` edges are left implicit about half the time |
| `leads` | Person | Project | subproperty of `works_on` |
| `uses_tech` | Project | Technology | |
| `delivered_for` | Project | Client | functional |
| `has_skill` | Person | Technology | survey source, lower confidence |
| `located_in` | Department or Client | City | functional |
| `depends_on` | Technology | Technology | transitive, acyclic |
| `mentors` | Person | Person | irreflexive |

## Structure planted in the data

The generator gives the graph learnable regularities, so that embedding and retrieval experiments are meaningful:
departments favour particular technologies and client industries, people tend to hold skills in technologies used
by their own projects, and about one fifth of project staffing crosses department lines.

## Deliberate difficulty

* Some surface forms are ambiguous (for example `P. Nair` could be Priya Nair or Pooja Nair).
* About one fifth of unambiguous aliases are missing from the gazetteer, which caps recall of dictionary-based extraction.
* About ten percent of sentences contain negation or a mention without a relation, which tempts pattern-based extractors.
* `raw_records.csv` contains typos, reversed names, initials, case changes and city codes.
* `triples_dirty.csv` contains six error types: `domain`, `range`, `functional`, `cycle`, `self_loop`, `mutual`.

## Path syntax in `questions.csv`

Steps are comma separated. `rel>` follows a relation forward, `rel<` follows it backwards, and `rel+>` follows it
transitively. Answers are computed over the full graph with `leads` implying `works_on`, and the start entity is
removed from the answer set.
