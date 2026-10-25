# AGENTS.md — tp4a-ReAct

Course lab (TP 4.a, part 1): a ReAct "AI research assistant" notebook built with uv + LangChain 1.x + Groq + Tavily + arXiv + Chroma. All user-facing text (system prompt, tool descriptions, prints, analysis) is French.

Files: `main.ipynb` (the deliverable — 12 cells, all executed with stored outputs), `README.md` (French write-up: architecture, golden-set results, exercises — update it when tool names/model/numbers change), `check.py` (key pre-flight), `pyproject.toml` / `uv.lock`, `.env` (untracked, live keys) and `.env.example`. The assignment statement (`tp4a-partie1-react-uv.html`) is deliberately **not versioned** any more — don't re-add it.

Single commit on `main` (plus small fixes). No tests, lint, CI, or task runner. `rag_search tool.py` is an empty untracked stray — ignore it, don't commit it.

## Commands (PowerShell, repo root)

```powershell
uv sync                              # 225 locked pkgs; .venv already populated
uv run check.py                      # loads .env itself -> expect two ✅ ok
uv run --env-file .env jupyter lab   # how the kernel actually gets the keys (see blocker)
uv add <pkg>                         # never pip install / !pip install in a cell
uv lock --check                      # pyproject matches uv.lock
```

`uv` 0.12.5, uv-managed CPython 3.12.9, `requires-python >=3.12`. Don't build a venv with another interpreter; don't hand-edit `.venv`. The only kernel is `python3` (= this `.venv`).

## Keys and `.env`

- Cell 0 calls `load_dotenv()` — **keep it**. `uv` does not inject `.env` into the kernel; when this call was missing, a fresh kernel printed `❌ clé manquante` and `run_react_agent` silently short-circuited to `"[Pas de clé Groq]"` with 0 tools. `uv run --env-file .env jupyter lab` is a working belt-and-braces alternative.
- `.env` holds the live Groq/Tavily keys, is gitignored (`git check-ignore .env` confirms) and untracked; `.env.example` is the committed template (`Copy-Item .env.example .env`).
- **Never paste a key into a cell.** It has happened: an unquoted key line raised `SyntaxError` *and* persisted inside the saved traceback output, i.e. into a committed file. If it happens, clear that cell's outputs (`Edit > Clear Outputs`) and rotate the key.
- Env is read once at import: restart the kernel after editing `.env`.

## Notebook map — Step N of the spec ↔ one cell

| Cell | Contents |
|------|----------|
| 0 | imports, keys from env, `GROQ_MODEL`, prefix/length validation |
| 1 | 4-doc corpus, splitter, `all-MiniLM-L6-v2` embeddings, Chroma `tp4a_corpus`, retriever `k=4` |
| 2 | `rag_search` (`@tool`) + HNSW smoke test |
| 3 | `tavily_web_search` (`TavilySearch`) + `arxiv_search` (`@tool`) + `tools` list |
| 4 | `SYSTEM_PROMPT` (routing rules, out-of-domain refusal, sources, French) |
| 5 | `ChatGroq` + `create_agent(llm, tools, system_prompt=...)` |
| 6 | `golden_set` Q1–Q5 with `expected_tools` |
| 7 | `run_react_agent` — definition only; running that cell alone prints nothing |
| 8 | Step 7 loop over `golden_set` + 60 s pause/retry on TPM limit |
| 9 | Step 8 markdown analysis table (hand-written, French) |
| 10 | LangChain 1.0 migration recap (`TavilySearch` + `create_agent`) |
| 11 | Exercise 3: `routing_score()` |

Add new cells at the end; don't rename or reorder existing ones. Exercises 1, 2 and 4 in the spec are still open (README tracks them as "à faire").

## Gotchas

- **Model name is `openai/gpt-oss-20b` — keep the hyphens.** `openai/gpt-oss20b` returns `404 model_not_found`; since `run_react_agent` swallows exceptions this surfaces as an `[Erreur : ...]` answer with 0 tools, not a traceback. The spec's `llama-3.3-70b-versatile` also 404s on this account (now enterprise-only) — don't "restore" it. `check.py` hardcodes its own copy of the name; keep the two in sync.
- **Free tier for gpt-oss-20b caps at 8 000 TPM** → the 5-question loop can return `413 rate_limit_exceeded` (seen on Q4). Mitigations already in place: `reasoning_effort="low"` + `max_tokens=1024` (cell 5) and a 60 s pause + single retry (cell 8). Don't remove them, and don't re-run the golden set casually — it burns quota.
- **`run_react_agent` catches every exception** and returns `{"answer": "[Erreur : ...]", "tools_used": [], "steps": 0}`. A bad model/key/network therefore looks like an empty answer. Judge success by `tools_used` / `steps` and the printed trace, never by "answer is non-empty".
- **Deliberately migrated off `langchain-community`** (it emits a sunset `DeprecationWarning`): Chroma ← `langchain_chroma`, Tavily ← `langchain_tavily.TavilySearch`, arXiv ← `arxiv` SDK + `@tool`, agent ← `langchain.agents.create_agent`, embeddings ← `langchain_huggingface`. Any `langchain_community` import brings the warning back (the package stays in `pyproject.toml` for the spec — declared ≠ imported). Consequences: `TavilySearch` returns `{"results": [...]}`, and `create_agent(llm, tools, system_prompt=...)` replaces `prompt=`; `invoke({"messages": [...]}, {"recursion_limit": 12})` is unchanged.
- **Use `@tool`, not `Tool(func=...)`, for the exported tools.** A generic `Tool` wrapping a one-arg function exposes the argument as `__arg1`, so traces read `rag_search({'__arg1': ...})`; `@tool` gives `{'query': ...}` and keeps the function name. The only intentional `Tool.from_function` is the no-Tavily-key stub in cell 3.
- **The three HF warnings are silenced by env vars set at the top of cell 1, before any transformers import** (`HF_HUB_VERBOSITY=error`, `HF_HUB_DISABLE_PROGRESS_BARS=1`, `TRANSFORMERS_VERBOSITY=error`). Setting them after the import — or via `logging.getLogger("huggingface_hub").setLevel(...)` — does **not** work: huggingface_hub re-applies its own level at import. Moving that preamble below the imports resurrects the "Loading weights" bar and the unauthenticated-HF warning.
- **`load_dotenv()` resolves against the script's own directory**, not the shell cwd — pass `dotenv_path` for scripts outside the repo root (cell 0 relies on the notebook's directory being the repo root).
- **`recursion_limit=12` is a required anti-loop guardrail.** `GraphRecursionError` means the agent repeated tool calls — inspect the trace, don't raise the limit.
- **Golden set Q1–Q5 with `expected_tools` is the evaluation baseline.** Keep the question text stable; Q5 expects *no* tool call (out-of-domain refusal). Latest stored run (`nbconvert --execute`, 2026-10-25): Q1 1.91 s, Q2 3.61 s, Q3 21.96 s, Q4 16.84 s, Q5 11.78 s, `routing_score` 0.8 (Q1/Q2/Q5 exact, Q3/Q4 partiel). Tools and durations vary run to run even at `temperature=0`. Cell 9's markdown table and the README table are **hand-written and synced to that run only** — a new cell-8 run does not update them, so refresh both (durations, observed tools, verdicts) after any re-run.
- **Cold import ≈ 107 s** for `sentence_transformers` + `chromadb` + `langgraph`; the first retriever run also downloads `all-MiniLM-L6-v2` (~90 MB, cached after). A full restart-and-run-all is ~2 min. Avoid needless kernel restarts and use long timeouts.
- The retriever indexes 4 docs with `k=4`, so every query returns the whole corpus (only order varies) — intended by the lab.
- Re-running cell 1 used to append chunks to the same collection (4 → 8 → 12, duplicated hits). The cell now deletes `tp4a_corpus` before rebuilding — keep that guard.
- Rebuild the agent (`create_agent(...)`) after editing `SYSTEM_PROMPT` or the tool list; a stale agent keeps the old prompt.
- Keep tool names `rag_search` / `tavily_web_search` / `arxiv_search`, the collection name `tp4a_corpus`, and French for all user-facing text — the spec and its expected outputs use them.
- `.ipynb_checkpoints/main-checkpoint.ipynb` is an empty (72-byte, 0-cell) stale checkpoint. Don't restore from it.

## Verification (no test suite)

1. `uv run check.py` → both keys ✅ (this is all `check.py` does; it never imports the notebook).
2. Retriever: an HNSW question must return `vector_databases.md` first (cell 2 prints the top hit) — expected by design.
3. Run cells 0–8 top-to-bottom: **no stderr** is the acceptance test for keeping it warning-free; cell 8 must print per-question tools, duration, step count, and cell 11 the routing verdicts. Cell 7 has no output by design (definition only) and cell 9 is markdown — neither is a bug.
4. Headless full re-run (verified; ~2 min cold, burns 5 Groq calls, rewrites every stored output):

   ```powershell
   uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb --ExecutePreprocessor.timeout=900
   ```

   The kernel-manager `zmq` / "running over TCP without encryption" warnings it prints on stderr come from nbconvert, not from the cells. Afterwards, re-sync cell 9 and the README table to the new numbers.
