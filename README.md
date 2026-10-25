# TP 4.a — Part 1 : Build a ReAct Agent with LangChain

Assistant de recherche en intelligence artificielle construit avec **LangChain 1.x** : un agent **ReAct** qui choisit lui-même, à chaque question, entre un corpus documentaire local (Chroma), une recherche web (Tavily) et arXiv — puis répond en français avec ses sources.

Le notebook trace chaque étape `Action → Observation → Réponse finale`, mesure les outils utilisés, la latence et le nombre d'étapes, et évalue l'agent sur un **golden set de 5 questions**.

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Configuration des clés API](#configuration-des-clés-api)
- [Utilisation](#utilisation)
- [Le notebook, cellule par cellule](#le-notebook-cellule-par-cellule)
- [Résultats du golden set](#résultats-du-golden-set)
- [Enseignements](#enseignements)
- [Choix techniques et pièges](#choix-techniques-et-pièges)
- [Migration vers LangChain 1.0](#migration-vers-langchain-10)
- [Exercices](#exercices)
- [Limites et perspectives](#limites-et-perspectives)

---

## Aperçu

Le pattern **ReAct** (*Reasoning + Acting*) alterne trois phases :

1. le modèle **raisonne** (*Thought*) ;
2. il **agit** en appelant un outil (*Action*) ;
3. il **lit** le résultat de l'outil (*Observation*).

La boucle se répète tant que le modèle estime manquer d'information, puis s'arrête sur une réponse finale. Le notebook utilise les **tool calls structurés** (JSON) exposés nativement par les LLM récents : pas de parsing de texte `Thought:` / `Action:`, mais la même logique.

Un garde-fou `recursion_limit = 12` empêche l'agent de boucler indéfiniment sur des appels d'outils.

---

## Architecture

```
Question utilisateur
        │
        ▼
  Agent ReAct (Groq)
        │
        ├── Thought ──► tool_calls ?
        │                    │ non ──► Réponse finale (+ sources)
        │                    │ oui
        │                    ▼
        │            Action : rag_search | tavily_web_search | arxiv_search
        │                    │
        │                    ▼
        │            Observation (ToolMessage)
        │                    │
        └────────────────────┘  (boucle, max. 12 itérations)
```

| Outil | Source | Quand l'utiliser |
|-------|--------|------------------|
| `rag_search` | Corpus local Chroma (4 documents IA) | Concepts établis : Transformer, RAG, HNSW, LoRA… |
| `tavily_web_search` | Web public (API Tavily) | Informations récentes (2024-2026) ou hors corpus |
| `arxiv_search` | arXiv.org (SDK `arxiv`) | Papier précis, publication académique |

Le corpus local contient 4 documents (`transformers.md`, `rag_systems.md`, `vector_databases.md`, `finetuning.md`), découpés en chunks de 300 tokens (overlap 30) et indexés avec `all-MiniLM-L6-v2`. Le retriever renvoie `k=4`, donc **tout le corpus** à chaque requête : c'est voulu par le sujet, seul l'ordre varie.

---

## Stack technique

| Composant | Version |
|-----------|---------|
| Python | 3.12.9 (géré par uv) |
| uv | 0.12.5 |
| langchain | 1.4.7 |
| langchain-core | 1.7.0 |
| langgraph | 1.2.17 |
| langchain-groq | 1.1.3 |
| langchain-chroma | 1.1.0 |
| langchain-tavily | 0.2.18 |
| langchain-huggingface | 1.2.3 |
| chromadb | 1.5.9 |
| sentence-transformers | 6.2.0 |
| arxiv | 4.0.1 |
| tavily-python | 0.8.5 |
| tiktoken | 0.14.0 |
| JupyterLab / ipykernel | dev dependencies |

Modèle LLM : **`openai/gpt-oss-20b`** via Groq, `temperature=0`, `reasoning_effort="low"`, `max_tokens=1024`.

> `llama-3.3-70b-versatile`, utilisé par le sujet, renvoie `404 model_not_found` sur ce compte (modèle passé en offre entreprise). Le notebook utilise donc `gpt-oss-20b`.

---

## Structure du dépôt

```
tp4a-ReAct/
├── main.ipynb                     # livrable : notebook complet, exécuté
├── check.py                       # pré-vol : vérifie les clés API
├── pyproject.toml                 # dépendances (uv)
├── uv.lock                        # versions verrouillées
├── AGENTS.md                      # notes pour les agents de code
├── .env.example                   # modèle de fichier de clés
└── .gitignore                     # exclut .env, .venv, caches…
```

Le fichier `.env` (clés réelles) **n'est pas versionné**.

---

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), un compte Groq et une clé Tavily.

```powershell
# 1. Dépendances (Python 3.12 géré par uv)
uv sync

# 2. Clés API
Copy-Item .env.example .env
# puis éditer .env

# 3. Pré-vol
uv run check.py
# Groq   : ✅ ok
# Tavily : ✅ ok

# 4. Notebook
uv run jupyter lab
```

Si les clés ne sont pas visibles après un redémarrage du kernel :

```powershell
uv run --env-file .env jupyter lab
```

Le kernel à sélectionner est **`python3`** (celui du `.venv` du projet).

---

## Configuration des clés API

`.env` à la racine du dépôt :

```dotenv
GROQ_API_KEY=gsk_votre_cle_groq
TAVILY_API_KEY=tvly-votre_cle_tavily
```

- `uv` ne charge pas `.env` automatiquement : le notebook (cellule 0) et `check.py` appellent `load_dotenv()`.
- Les clés sont lues avec `os.environ.get(...)` et validées par préfixe (`gsk_`, `tvly-`).
- Ne jamais écrire une clé en dur dans une cellule : elle finirait dans les sorties du notebook.

---

## Utilisation

1. Ouvrir `main.ipynb`.
2. `Kernel > Restart & Run All` (≈ 2 min à froid : import de `sentence_transformers`, `chromadb`, `langgraph` et chargement du modèle d'embedding).
3. Lire les traces de la cellule 8 et l'analyse de la cellule 9.

Les sorties sont déjà enregistrées dans le notebook ; une ré-exécution complète consomme du quota Groq.

---

## Le notebook, cellule par cellule

| # | Contenu |
|---|---------|
| 0 | Imports, `load_dotenv()`, clés API, `GROQ_MODEL`, validation par préfixe |
| 1 | Corpus IA (4 documents), découpage, embeddings `all-MiniLM-L6-v2`, Chroma (`tp4a_corpus`), retriever `k=4` |
| 2 | Outil `rag_search` (`@tool`) + test rapide sur HNSW |
| 3 | Outil `tavily_web_search` (`TavilySearch`) + outil `arxiv_search` (`@tool`), liste `tools` |
| 4 | `SYSTEM_PROMPT` : rôle, règles de routage, refus hors-domaine, sources, français |
| 5 | `ChatGroq` + `create_agent(llm, tools, system_prompt=...)` |
| 6 | Golden set Q1–Q5 avec `expected_tools` |
| 7 | `run_react_agent(question)` : invocation, trace, outils, étapes, durée, gestion d'erreur |
| 8 | Boucle d'exécution sur le golden set + pause/retry si limite TPM |
| 9 | Analyse des résultats (tableau Q / Attendu / Observé / Verdict / Diagnostic) |
| 10 | Récapitulatif de la migration LangChain 1.0 (`TavilySearch` + `create_agent`) |
| 11 | Exercice 3 : `routing_score()` — score de routage automatique |

---

## Résultats du golden set

Exécution de référence (`gpt-oss-20b`, `temperature=0`) :

| Q | Question | Outils attendus | Outils observés | Durée | Étapes | Verdict |
|---|----------|-----------------|-----------------|-------|--------|---------|
| Q1 | Fonctionnement de HNSW | `rag_search` | `rag_search` | 1.91 s | 1 | ✅ Exact |
| Q2 | Nouveaux embeddings open-source 2025 | `tavily_web_search` | `tavily_web_search` | 3.61 s | 1 | ✅ Exact |
| Q3 | Papier original LoRA (2021) | `arxiv_search` | `arxiv_search` + `rag_search` | 21.96 s | 2 | ≈ Partiel |
| Q4 | HNSW vs alternatives en production | `rag_search` + `tavily_web_search` | `tavily_web_search` | 16.84 s | 1 | ≈ Partiel |
| Q5 | Recette de l'auteur de *Attention is all you need* | aucun (refus) | aucun | 11.78 s | 0 | ✅ Exact |

**Score de routage : 0.8 / 1.0** (`routing_score`, cellule 11).

Ces chiffres ne sont pas reproductibles à l'identique : même à `temperature=0`, le choix des outils et les latences varient selon le run et le modèle. Le sujet le précise explicitement.

---

## Enseignements

- **Q1 — le corpus local suffit.** Question simple, un seul outil, une seule étape, réponse fidèle à `vector_databases.md`.
- **Q2 — bon outil ≠ réponse parfaite.** La recherche web est bien choisie et les sources citées, mais la liste annoncée « open-source » mélange les licences : `Llama-Embed-Nemotron-8B` est non commercial et `Gemini-Embedding-2` propriétaire.
- **Q3 — l'auto-correction est accidentelle.** La requête arXiv est trop vague, l'observation hors sujet, et c'est un second outil (`rag_search`, non attendu) qui sauve la réponse. Le titre du papier cité est d'ailleurs erroné.
- **Q4 — les règles du prompt restent des suggestions.** Le prompt demande `rag_search` d'abord pour HNSW ; l'agent va directement sur le web. Aucun mécanisme ne l'oblige au multi-hop attendu.
- **Q5 — le refus hors-domaine est fiable.** Une règle simple et non ambiguë est suivie, sans appel d'outil.

Conclusion : un agent ReAct pur n'a **aucun mécanisme d'auto-évaluation**. Il ne vérifie ni la pertinence des documents récupérés, ni la fidélité de sa réponse, et ne change pas de source de lui-même en cas d'échec. C'est précisément ce que corrige la Partie 3 (LangGraph) avec des nœuds `grade_docs` / `grade_answer`, une réécriture de requête et un fallback web imposés par le développeur.

---

## Choix techniques et pièges

- **`@tool` plutôt que `Tool(func=...)`** : un `Tool` générique expose l'argument en `__arg1`, donc les traces affichent `rag_search({'__arg1': ...})`. Avec `@tool`, on obtient `{'query': ...}` et le nom de la fonction.
- **Migration hors de `langchain-community`** : le package est en fin de vie et émet un `DeprecationWarning`. Chroma vient de `langchain_chroma`, Tavily de `langchain_tavily`, les embeddings de `langchain_huggingface`, l'agent de `langchain.agents.create_agent`.
- **`TavilySearch` renvoie un `dict`** (`{"results": [...]}`) et non une liste.
- **Avertissements Hugging Face** : les variables `HF_HUB_VERBOSITY=error`, `HF_HUB_DISABLE_PROGRESS_BARS=1`, `TRANSFORMERS_VERBOSITY=error` doivent être définies **avant** tout import de `transformers` / `sentence-transformers`.
- **Limite Groq** : le tier gratuit de `gpt-oss-20b` plafonne à **8 000 TPM** ; le golden set peut renvoyer `413 rate_limit_exceeded`. D'où `reasoning_effort="low"`, `max_tokens=1024` et une pause de 60 s avec un nouvel essai dans la boucle.
- **Collection Chroma** : ré-exécuter la cellule 1 sans redémarrer le kernel empilait les chunks (4 → 8 → 12). La cellule supprime désormais `tp4a_corpus` avant de la reconstruire.
- **Agent obsolète** : après modification de `SYSTEM_PROMPT` ou de la liste d'outils, il faut reconstruire l'agent (`create_agent(...)`).
- **Erreurs avalées** : `run_react_agent` intercepte toute exception et renvoie `"[Erreur : ...]"` avec `tools_used = []`. Un échec ressemble donc à une réponse vide : toujours inspecter les outils et les étapes.

---

## Migration vers LangChain 1.0

| Ancienne API (cours) | API utilisée ici |
|----------------------|------------------|
| `langchain_community.vectorstores.Chroma` | `langchain_chroma.Chroma` |
| `langchain_community.tools.tavily_search.TavilySearchResults` | `langchain_tavily.TavilySearch` |
| `langchain_community.tools.arxiv.ArxivQueryRun` | SDK `arxiv` + `@tool` |
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `langchain_community.embeddings.HuggingFaceEmbeddings` | `langchain_huggingface.HuggingFaceEmbeddings` |

Équivalent moderne des deux cellules concernées :

```python
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

tavily_tool = TavilySearch(
    max_results=3,
    name="tavily_web_search",
    description="Recherche sur le web public pour les informations récentes ou hors corpus.",
)

tools = [rag_search, tavily_tool, arxiv_search]
agent_executor = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
```

---

## Exercices

| # | Énoncé | État |
|---|--------|------|
| 1 | Réessayer `run_react_agent` si la réponse finale est vide (2 tentatives max) | à faire |
| 2 | Ajouter une règle « titre exact / auteur » pour `arxiv_search` et reformuler une fois | à faire |
| 3 | `routing_score(golden_set, react_results)` → `exact` / `partiel` / `faux` + score global | ✅ cellule 11 (score 0.8) |
| 4 | Ajouter un quatrième outil (ex. Wikipedia) | à faire |

---

## Limites et perspectives

- Un seul LLM, aucun mécanisme de vérification des sources ni de repli automatique.
- Le corpus local est minuscule (4 documents) et `k=4` renvoie tout le corpus : impossible de mesurer réellement la pertinence du retrieval.
- Les questions multi-hop (Q4) ne sont pas garanties.
- La Partie 3 (Adaptive RAG avec LangGraph) ajoute l'auto-évaluation, la réécriture de requête et le fallback.

---

Projet pédagogique — TP 4.a, Module 4 « AI Agents ».
