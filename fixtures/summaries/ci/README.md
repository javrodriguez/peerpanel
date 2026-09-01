# ci community-report cache — real recorded fixtures

Every file here is the REAL output of the summarisation LLM (llama3.1:8b via
the provider seam, temperature 0, prompt v1) for one Leiden community,
produced by an actual call. Nothing is hand-written.

Cached by the CONTENT of the community (its member entities and the relations
between them) plus the provider and prompt version — so a community whose
membership changes re-summarises, and one that does not is served from here.
That is also why the file count can exceed the report count of any single run:
a community that changed shape leaves its earlier report behind, keyed to the
membership that produced it.

Regenerate for real: `make summaries` with Ollama running.

These reports are the residual contamination surface LIMITATIONS.md names: they
are generated once over the whole corpus, so a run that excludes a document can
still be influenced by a report describing it. The LLM face of global search
refuses to run when anything is excluded, for exactly that reason.
