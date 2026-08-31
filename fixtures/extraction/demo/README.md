# demo extraction cache — real recorded fixtures

Every file here is the REAL output of the extraction LLM (llama3.1:8b via the
provider seam, temperature 0, prompt v3) over one demo-corpus
chunk, produced by an actual call and cached by (chunk id, provider, prompt
version). Nothing is hand-written. Regenerate for real:
`python -m peerpanel graph build --corpus demo` with Ollama running (a changed prompt version re-extracts
everything).
