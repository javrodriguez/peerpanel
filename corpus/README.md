# Corpus

Two scales, both pinned and honestly sized (actual numbers live in the manifests):

- **`ci/` + `ci.manifest.json`** — the small committed corpus CI exercises: CC BY full texts
  from the PMC Open Access subset, pinned by (PMCID, version, md5) and byte-verified by
  `python -m peerpanel corpus verify`. One document (PMC10729969) is force-included as a
  labeled twin-retrieval showcase — it is the published form of a manuscript the demo reviews,
  and it is **excluded from every metric** by the self-exclusion rule (see DECISIONS.md).
- **`demo.candidates.json`** — the ordered candidate list for the larger demo corpus:
  citation seeds first (the query manuscripts' cited papers — forced, never dropped), then
  one-hop citation neighbours, then topical top-up. The final `demo.manifest.json` is written
  once indexing cost is measured; `make corpus` then fetches into `demo/` (gitignored),
  md5-verified, license-gated at fetch time.

Licensing: the repo's MIT license covers code only. Every corpus document remains under its
own Creative Commons license, held by its authors — per-document attribution in
[CITATIONS.md](../CITATIONS.md). Only CC BY / CC0 documents are ever fetched or committed;
the gate also refuses retracted papers.
