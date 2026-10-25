# AGENTS.md — tp4a-ReAct

Course lab (TP 4.a, part 1): build a ReAct "AI research assistant" notebook with uv + LangChain 1.x + Groq + Tavily + arXiv + Chroma.
Files: `tp4a-partie1-react-uv.html` (assignment spec — authoritative for code, expected outputs, analysis table), `main.ipynb` (the deliverable, in progress), `check.py` (key pre-flight), `pyproject.toml`/`uv.lock`, `.env`. `rag_search tool.py` is an empty stray file — ignore it.
No tests, lint, CI, or git repo (`.gitignore` lists only `.env`).

## Commands (PowerShell, repo root)

```powershell
uv sync                      # 225 pkgs; .venv already populated
uv run check.py              # key check; loads .env itself — expect two ✅ ok
uv run jupyter lab           # opens main.ipynb; only kernel is "python3" = this .venv
uv run --env-file .env jupyter lab      # fallback if keys are missing after a kernel restart
uv add <pkg>                 # never pip install / !pip install inside the notebook
uv lock --check              # pyproject matches uv.lock
```

`uv` 0.12.5, uv-managed CPython 3.12.9, `requires-python >=3.12`. Don't create a venv with another interpreter; don't hand-edit `.venv`.

## Where the work actually is

- `main.ipynb` has **12 cells, 0–11**: 0 keys, 1 retriever (Chroma), 2 `rag_search`, 3 Tavily + arXiv tools, 4 `SYSTEM_PROMPT`, 5 agent, 6 golden set, 7 `run_react_agent` (definition only — running it alone prints nothing), 8 the Step 7 loop over `golden_set`, 9 the Step 8 French analysis table (markdown), 10 the LangChain 1.0 migration recap (`TavilySearch` + `create_agent`), 11 Exercise 3 (`routing_score`). **All cells are executed and their real outputs are stored**; the golden set has been run (latest: Q1 1.72 s, Q2 2.47 s, Q3 26.82 s, Q4 29.95 s, Q5 8.24 s, score 0.8). Keep new cells at the end rather than renaming/reordering existing ones.
- Work step by step: Step N of `tp4a-partie1-react-uv.html` ↔ one notebook cell, copying the spec's code and output shape instead of inventing a design.
- `check.py` is a standalone duplicate of the notebook's key check — optional pre-flight; the notebook must never import it.

## Gotchas

- **The model is `openai/gpt-oss-20b` — keep the hyphens.** Cell 0 once said `openai/gpt-oss20b`, which returns `404 model_not_found`; because `run_react_agent` catches every exception, that surfaced as an `[Erreur : ...]` answer with 0 tools, not a traceback. The spec's `llama-3.3-70b-versatile` also returns 404 on this account (Groq now lists it as enterprise/contact-sales) — do not "restore" it. `check.py` hardcodes its own model name and does not read the notebook; keep the two in sync.
- **This account's free tier for gpt-oss-20b caps at 8 000 TPM**, so the 5-question loop can return `413 rate_limit_exceeded` (observed on Q4). Mitigations already in the notebook: `reasoning_effort="low"` + `max_tokens=1024` in cell 5, and a 60 s pause + single retry in cell 8. Don't remove them; don't run the golden set repeatedly.
- **`run_react_agent` catches every exception** and returns `{"answer": "[Erreur : ...]", "tools_used": [], "steps": 0}`. A bad model/key/network therefore looks like an empty answer, not a traceback — always inspect `tools_used`/`steps` and the printed trace, not just whether an answer is non-empty.
- **The notebook is deliberately migrated off `langchain-community`** (it still emits a sunset `DeprecationWarning`): Chroma comes from `langchain_chroma`, Tavily from `langchain_tavily.TavilySearch`, arXiv from the `arxiv` SDK wrapped with `@tool`, the agent from `langchain.agents.create_agent`, embeddings via `langchain_huggingface`. Reintroducing a `langchain_community` import brings the warning back (the package stays in `pyproject.toml` for the spec — declared ≠ importable). Consequences: `TavilySearch` returns `{"results": [...]}` (not a list), and `create_agent(llm, tools, system_prompt=...)` replaces `prompt=`; `invoke({"messages": [...]}, {"recursion_limit": 12})` still works unchanged.
- **Use `@tool` decorators, not `Tool(func=...)`.** A plain `Tool` wrapping a one-arg function exposes the argument as `__arg1`, so traces read `rag_search({'__arg1': ...})`; `@tool` yields the expected `{'query': ...}` and keeps the name from the function.
- **Three HF warnings are silenced by env vars set at the top of the retriever cell, before any transformers import** (`HF_HUB_VERBOSITY=error`, `HF_HUB_DISABLE_PROGRESS_BARS=1`, `TRANSFORMERS_VERBOSITY=error`). Setting them after the import, or with `logging.getLogger("huggingface_hub").setLevel(...)`, does **not** work — huggingface_hub re-applies its own level at import. Moving that preamble below the imports resurrects the "Loading weights" bar and the unauthenticated-HF warning.
- **Keys only reach code via `.env`.** `uv` does not load it; notebook cell 0 and `check.py` call `load_dotenv()`. New scripts need `load_dotenv()` (it resolves against the *script's* directory, so a script outside the repo root must pass `dotenv_path`) or `uv run --env-file .env <cmd>`. After editing `.env`, restart the kernel — env is read once at import.
- `.env` holds live Groq/Tavily keys. They must stay there: read them with `os.environ.get(...)` and validate with `startswith("gsk_")` / `startswith("tvly-")`. Hardcoding them in cells (has happened — it also silently breaks the prefix check, since the key trivially starts with itself) leaks them into notebook outputs and any shared copy.
- `temperature=0` keeps runs comparable, but tool choices and durations still vary between runs and models. The Step 8 table describes one real run; the spec's numbers are illustrative, not a target.
- `recursion_limit=12` is a required anti-loop guardrail; `GraphRecursionError` means the agent repeated tool calls — inspect the trace, don't raise the limit.
- **Golden set Q1–Q5 with `expected_tools` is the evaluation baseline.** Keep question text stable; Q5 expects no tool call (out-of-domain refusal). Reported scores depend on run-to-run LLM variation even at temperature 0 — the spec's expected outputs explicitly say yours will differ.
- **Cold import ≈ 107 s** for `sentence_transformers` + `chromadb` + `langgraph`; first retriever run also downloads `all-MiniLM-L6-v2` (~90 MB, cached afterwards). Avoid needless kernel restarts and use long timeouts.
- The retriever indexes 4 docs with `k=4`, so every query returns the whole corpus (only order varies) — intended by the lab, not a bug.
- Re-running cell 1 used to append chunks to the same collection (4 → 8 → 12, so the retriever returned duplicates). The cell now deletes collection `tp4a_corpus` before rebuilding — keep that guard.
- Rebuild the agent (`create_agent(...)`) after editing `SYSTEM_PROMPT` or the tool list; a stale agent keeps the old prompt.
- Keep tool names `rag_search` / `tavily_web_search` / `arxiv_search`, collection `tp4a_corpus`, and French for all user-facing text (prompt, tool descriptions, printed analysis) — the spec and its expected outputs use them.
- `.ipynb_checkpoints/main-checkpoint.ipynb` is an empty, stale checkpoint (0 cells). Don't restore from it.

## Verification (no test suite)

1. `uv run check.py` → both keys ✅.
2. Retriever check: an HNSW question must return `vector_databases.md` first (cell 2 already asserts this by printing the top hit).
3. Run the notebook top-to-bottom: cells 0–8 must produce **no stderr** (the acceptance test for keeping it warning-free), and the golden set must report per-question tools used, duration, step count, and routing verdicts (exact/partial/wrong). Cell 9 is the markdown analysis; cells 10–11 rebuild the agent and score the routing.
4. Headless re-run that avoids your own kernel state: build a one-cell notebook and execute it with `nbclient.NotebookClient(..., timeout=600)` from a script; cell 1 needs ~2 min the first time.
