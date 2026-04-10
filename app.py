# app.py — Hugging Face Space (Gradio) using a prebuilt Chroma index
# Embeddings: jinaai/jina-embeddings-v3 (HF), trust_remote_code=True, normalize_embeddings=True

import os
import gradio as gr

# Silence Chroma telemetry noise
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import snapshot_download

# -------- Config (can be overridden via Space "Variables") --------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_langchain")     # path to your committed Chroma index
EMB_MODEL   = os.getenv("EMB_MODEL", "jinaai/jina-embeddings-v3")
TOPK_DEF    = int(os.getenv("TOPK", "5"))

if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    print("Downloading Chroma index from HuggingFace...")
    snapshot_download(
        repo_id="cwinkler/bgh-chroma-index",
        repo_type="dataset",
        local_dir=PERSIST_DIR,
    )

# Embedding function for query text — must match the model used to build the index
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

def load_vector_store():
    """
    Load the persisted Chroma collection with the embedding function for query-time encoding.
    Returns (vs, error_message_or_None)
    """
    try:
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=EMBEDDINGS,
            client_settings=Settings(anonymized_telemetry=False),
        )
        # sanity check (forces collection open)
        _ = vs._collection.count()
        return vs, None
    except Exception as e:
        # Helpful diagnostics: list available collections
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
            )
            existing = [c.name for c in client.list_collections()]
        except Exception:
            existing = []
        msg = (
            f"Failed to load Chroma store at '{PERSIST_DIR}'. "
            f"Existing collections: {existing or '—'}. "
            "Check that the index folder is present in the Space and the collection name matches."
        )
        return None, f"{msg}\n\nDetails: {e}"

VS, LOAD_ERR = load_vector_store()

def search(query: str, k: int = TOPK_DEF):
    if LOAD_ERR:
        return f"⚠️ {LOAD_ERR}"
    q = (query or "").strip()
    if not q:
        return "Please enter a query."
    try:
        results = VS.similarity_search_with_score(q, k=int(k))
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results."

    lines = [f"### Top {len(results)} results"]
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "(no source)"
        snippet = (doc.page_content[:800] + "…") if len(doc.page_content) > 800 else doc.page_content
        lines.append(f"**[{i}]**  \nSimilarity: `{score:.4f}`\n\n> {snippet}")
    lines.append("\n> **Disclaimer:** Models can produce incorrect or misleading statements. Verify with sources.")
    return "\n\n".join(lines)

with gr.Blocks(title="Semantische Suchmaschine für BGH Leitsatzentscheidungen v0.1") as demo:
    gr.Markdown(
        """
        ## Semantische Suchmaschine für BGH Leitsatzentscheidungen v0.1
        **Datensatz: 21.603 Leitsatzentscheidungen des BGH (ab dem Jahr 2000) extrahiert aus https://zenodo.org/records/15153244**

        **Modell:** jinaai/jina-embeddings-v3

        **Wie es funktioniert:** Ermöglicht die semantische Suche im Datensatz und gibt die Entscheidungen geordnet nach Ähnlichkeitswerten zurück.

        **Versuche bespielsweise:**
        - `Kann KI Erfinder sein?` → erwartetes Aktenzeichen **X ZB 5/22**

        *Disclaimer:* Models may produce incorrect or misleading statements. Verify with sources.
        """
    )
    with gr.Row():
        q = gr.Textbox(label="Query", placeholder="Kann KI Erfinder sein?")
        k = gr.Slider(1, 20, value=TOPK_DEF, step=1, label="Top-K")
    out = gr.Markdown()
    gr.Button("Search").click(fn=search, inputs=[q, k], outputs=[out])

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
