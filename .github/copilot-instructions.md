Project: WhatsApp bond-trade extraction (Python, FastAPI, SQLite, LLM via Ollama/langchain).
Rules for every change:
1. Never hardcode sample-chat content (issuer names, phrases, names, numbers) in code. Anything domain-specific goes in config.py or JSON files (lexicon.json, issuer_aliases.json).
2. Deterministic code (parsing, units, dates, validation) must be pure functions with unit tests. The LLM only interprets meaning.
3. Never invent values. Unknown means null, never a default. Never use datetime.now() as a fallback.
4. Never silently drop data. Anything skipped must be counted and reported.
5. Keep existing function names and signatures where possible. Add keys, don't rename them.
6. Do not touch the bond identity / dedupe logic in db.py; it is being reworked separately. Leave a TODO comment instead.
7. Add or update tests for every behaviour change. Do not commit real chat data.
8. Never print, quote or paste the contents of files in data/ or baseline/. Report counts and field names only.