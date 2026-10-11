import os
import re
import html
import pickle
import time
from datetime import datetime

import faiss
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

# =====================================================================
# CONFIGURATION DE LA PAGE
# =====================================================================
st.set_page_config(
    page_title="ReportAnalyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# DESIGN (CSS)
# =====================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    :root {
        color-scheme: dark;
        --bg: #0d1a1a;
        --field: #1b2e2f;
        --card: rgba(255, 255, 255, 0.035);
        --border: rgba(255, 255, 255, 0.12);
        --text: #f4f8f7;
        --muted: #8aa09e;
        --accent: #ff7a4d;
        --accent-dark: #e8643a;
    }

    html, body, .stApp, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(900px 500px at 78% -8%, rgba(46, 140, 120, 0.28), transparent 60%),
            radial-gradient(700px 400px at 5% 0%, rgba(30, 90, 85, 0.25), transparent 60%),
            var(--bg);
        color: var(--text);
        color-scheme: dark;
    }

    /* Éléments Streamlit inutiles */
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], .stDeployButton { display: none !important; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }

    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px; }

    h1, h2, h3, h4, h5 { color: var(--text) !important; font-weight: 500; letter-spacing: -0.01em; }
    label, [data-testid="stWidgetLabel"] p { color: var(--text) !important; }

    /* Barre du haut */
    .brand {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.1rem; font-weight: 600; color: var(--text); padding-top: 4px;
    }
    .user-chip { text-align: right; color: var(--muted); font-size: 0.88rem; padding-top: 10px; }
    .user-chip b { color: var(--text); font-weight: 500; }

    .section-title { color: var(--text); font-size: 1.02rem; font-weight: 500; margin: 0 0 6px 0; }
    .section-sub { color: var(--muted); font-size: 0.86rem; margin-bottom: 10px; }
    .report-name { color: var(--text); font-weight: 500; font-size: 0.92rem; word-break: break-word; }
    .report-meta { color: var(--muted); font-size: 0.8rem; }
    .empty-state { color: var(--muted); font-size: 0.9rem; padding: 6px 0; }

    /* Cartes (conteneurs avec bordure) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
    }

    /* Boutons */
    .stButton button, .stFormSubmitButton button {
        background-color: var(--accent); color: #1a0f0a; border: none;
        border-radius: 10px; padding: 0.45rem 1rem; font-weight: 500;
        transition: background-color 0.15s;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        background-color: var(--accent-dark); color: #1a0f0a;
    }
    .stButton button p, .stFormSubmitButton button p { color: #1a0f0a !important; }
    .stButton button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }

    /* ---------- Champs de saisie : fond sombre + texte clair visible ---------- */
    div[data-baseweb="input"], div[data-baseweb="base-input"],
    div[data-baseweb="input"] > div {
        background-color: var(--field) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"] { border: 1px solid var(--border) !important; }
    .stTextInput input, div[data-baseweb="input"] input {
        background-color: transparent !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        caret-color: var(--accent) !important;
    }
    .stTextInput input::placeholder, [data-testid="stChatInput"] textarea::placeholder {
        color: #7d9593 !important;
        -webkit-text-fill-color: #7d9593 !important;
        opacity: 1;
    }
    div[data-baseweb="input"] button, div[data-baseweb="input"] svg { color: var(--muted) !important; }

    div[data-baseweb="select"] > div {
        background-color: var(--field) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="select"] input {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] {
        background-color: #16292a !important; color: var(--text) !important;
    }
    div[data-baseweb="popover"] li:hover { background-color: #21393a !important; }
    [data-testid="stCheckbox"] label span, [data-testid="stCheckbox"] label p {
        color: var(--text) !important;
    }

    /* Zone d'import */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.04);
        border: 1px dashed rgba(255,255,255,0.22);
        border-radius: 14px;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--accent) !important; border: none !important; border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"] button * { color: #1a0f0a !important; }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--text); }

    [data-testid="stChatInput"], [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background-color: var(--field) !important;
        border-radius: 999px !important;
    }
    [data-testid="stChatInput"] { border: 1px solid var(--border) !important; }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        caret-color: var(--accent) !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background: var(--accent) !important; border-radius: 999px !important; color: #1a0f0a !important;
    }
    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] { background: transparent !important; }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border); border-radius: 12px;
    }
    [data-testid="stExpander"] summary p { color: var(--text) !important; }

    hr { border-color: var(--border); }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# CONSTANTES
# =====================================================================
DATA_DIR = "report_analyst_data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

TOUS = "Tous les rapports"
NB_PASSAGES = 4
MAX_CONTEXTE = 14000

GEMINI = "Google Gemini"
CLAUDE = "Anthropic (Claude)"
FOURNISSEURS = [GEMINI, CLAUDE]

MODELES = {
    GEMINI: ["gemini-2.5-flash", "gemini-3.8-flash"],
    CLAUDE: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
}
CLES = {GEMINI: "cle_gemini", CLAUDE: "cle_claude"}


def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


# =====================================================================
# PERSISTANCE
# =====================================================================
def load_data():
    chunks, metadata, index = [], {"reports": []}, None
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
        except Exception:
            index = None
    return chunks, metadata, index


def save_data(chunks, metadata, index):
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
    elif os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)


# =====================================================================
# SESSION
# =====================================================================
for key, default in {
    "logged_in": False,
    "username": "",
    "messages": [],
    "flash": [],
    "uploader_key": 0,
    "provider": GEMINI,
    CLES[GEMINI]: get_secret("GEMINI_API_KEY") or "",
    CLES[CLAUDE]: get_secret("ANTHROPIC_API_KEY") or get_secret("CLAUDE_API_KEY") or "",
    "allow_general": False,
    "scope": TOUS,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def cle_active():
    return st.session_state[CLES[st.session_state.provider]].strip()


# =====================================================================
# CONNEXION
# =====================================================================
def login_page():
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:18px;">
                <div class="brand" style="justify-content:center; font-size:1.4rem;">
                    ReportAnalyst
                </div>
                <p style="color:#8aa09e; margin-top:10px; font-size:0.92rem;">
                    Analyse et interrogation de rapports PDF
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            # Le mot de passe est de nouveau masqué (type="password")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
            if submit:
                if username == get_secret("AUTH_USER", "admin") and password == get_secret(
                    "AUTH_PASSWORD", "admin123"
                ):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect.")


if not st.session_state.logged_in:
    login_page()
    st.stop()


# =====================================================================
# MODÈLE D'EMBEDDINGS ET DONNÉES
# =====================================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


embedding_model = load_embedding_model()

if "chunks" not in st.session_state:
    (
        st.session_state.chunks,
        st.session_state.metadata,
        st.session_state.index,
    ) = load_data()


# =====================================================================
# INDEXATION
# =====================================================================
def decouper_texte(texte, taille=1000, chevauchement=200):
    morceaux, debut = [], 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            espace = texte.rfind(" ", debut, fin)
            if espace > debut:
                fin = espace
        morceau = texte[debut:fin].strip()
        if morceau:
            morceaux.append(morceau)
        if fin == len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)
    return morceaux


def construire_index(chunks):
    if not chunks:
        return None
    vecteurs = embedding_model.encode(
        [c["texte"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)
    return index


def indexer_fichiers(fichiers):
    chunks = list(st.session_state.chunks)
    rapports = list(st.session_state.metadata.get("reports", []))
    noms = {r["name"] for r in rapports}
    messages, traite = [], False

    for fichier in fichiers:
        if fichier.name in noms:
            messages.append(("info", f"{fichier.name} est déjà importé."))
            continue
        try:
            reader = PdfReader(fichier)
            nb_pages = len(reader.pages)
            nb_chunks = 0
            for numero, page_obj in enumerate(reader.pages, start=1):
                texte = re.sub(r"\s+", " ", page_obj.extract_text() or "").strip()
                if not texte:
                    continue
                for i, morceau in enumerate(decouper_texte(texte), start=1):
                    chunks.append(
                        {"source": fichier.name, "page": numero, "chunk": i, "texte": morceau}
                    )
                    nb_chunks += 1
            if nb_chunks == 0:
                messages.append(
                    ("warning", f"Aucun texte extractible dans {fichier.name} (PDF scanné ?).")
                )
                continue
            rapports.append(
                {
                    "name": fichier.name,
                    "pages": nb_pages,
                    "chunks": nb_chunks,
                    "date": datetime.now().strftime("%d/%m/%Y"),
                }
            )
            noms.add(fichier.name)
            traite = True
            messages.append(("success", f"{fichier.name} est prêt ({nb_pages} pages)."))
        except Exception as e:
            messages.append(("error", f"Erreur de lecture sur {fichier.name} : {e}"))

    if traite:
        index = construire_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.metadata["reports"] = rapports
        st.session_state.index = index
        save_data(chunks, st.session_state.metadata, index)
    return messages


def supprimer_rapport(nom):
    restants = [c for c in st.session_state.chunks if c.get("source") != nom]
    rapports = [r for r in st.session_state.metadata.get("reports", []) if r["name"] != nom]
    index = construire_index(restants)
    st.session_state.chunks = restants
    st.session_state.metadata["reports"] = rapports
    st.session_state.index = index
    st.session_state.scope = TOUS
    save_data(restants, st.session_state.metadata, index)


# =====================================================================
# RECHERCHE
# =====================================================================
MOTS_RESUME = re.compile(
    r"résum|resum|synth|summar|overview|de quoi (parle|traite)|ملخص|لخص|خلاصة",
    re.IGNORECASE,
)


def rechercher(question, source=None, k=NB_PASSAGES):
    index = st.session_state.index
    if index is None or index.ntotal == 0:
        return []

    vecteur = embedding_model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    n = min(index.ntotal, max(k * 6, 40))
    scores, indices = index.search(vecteur, n)

    retenus, vus = [], set()
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(st.session_state.chunks):
            continue
        item = st.session_state.chunks[int(idx)]
        if source and item["source"] != source:
            continue
        retenus.append({**item, "score": float(score)})
        vus.add(int(idx))
        if len(retenus) >= k:
            break

    if MOTS_RESUME.search(question):
        noms = [source] if source else [r["name"] for r in st.session_state.metadata["reports"]]
        for nom in noms:
            debut = [
                (i, c) for i, c in enumerate(st.session_state.chunks) if c["source"] == nom
            ][:3]
            for i, c in debut:
                if i not in vus:
                    retenus.append({**c, "score": 0.0})
                    vus.add(i)

    retenus.sort(key=lambda c: (c["source"], c["page"], c["chunk"]))
    return retenus


# =====================================================================
# APPEL À L'IA
# =====================================================================
class ModeleIntrouvable(Exception):
    pass


def _appel_gemini(modele, api_key, system, echanges, message):
    client = genai.Client(api_key=api_key)
    
    contents = []
    for m in echanges:
        contents.append({
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}]
        })
    contents.append({"role": "user", "parts": [{"text": message}]})

    config = {
        "temperature": 0.2,
    }
    if system:
        config["system_instruction"] = system

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=modele,
                contents=contents,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "not found" in err_str.lower():
                raise ModeleIntrouvable(err_str)
            if ("503" in err_str or "unavailable" in err_str.lower() or "high demand" in err_str.lower()) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Erreur API Gemini : {err_str}")


def _appel_claude(modele, api_key, system, echanges, message):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": modele,
            "max_tokens": 4096,
            "system": system,
            "messages": echanges + [{"role": "user", "content": message}],
        },
        timeout=120,
    )
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"Erreur {r.status_code} Claude : {detail}")
    blocs = r.json().get("content", [])
    return "".join(b.get("text", "") for b in blocs if b.get("type") == "text").strip()


def appeler_ia(provider, api_key, system, historique, message):
    echanges = [{"role": m["role"], "content": m["content"]} for m in historique[-6:]]
    fonction = _appel_gemini if provider == GEMINI else _appel_claude
    derniere = None
    for modele in MODELES[provider]:
        try:
            return fonction(modele, api_key, system, echanges, message)
        except ModeleIntrouvable as e:
            derniere = e
        except Exception as e:
            raise e
    raise RuntimeError(f"Aucun modèle disponible avec cette clé. {derniere}")


def construire_consignes(autoriser_general):
    regle_absent = (
        "Si l'information n'est pas dans les extraits, dis-le clairement, puis complète "
        "avec tes connaissances générales en précisant que cette partie ne vient pas du rapport."
        if autoriser_general
        else "Si l'information n'est pas dans les extraits, réponds exactement : "
        "« Cette information n'est pas présente dans le rapport. » Tu peux ajouter ce que "
        "les extraits disent de plus proche, sans rien inventer."
    )
    return f"""Tu es ReportAnalyst, un analyste expert chargé de répondre à des questions sur des rapports PDF.

Règles :
1. Appuie-toi uniquement sur les EXTRAITS DU RAPPORT fournis dans le message.
2. Donne une réponse précise, complète et bien structurée (courts paragraphes ou listes quand c'est utile). Reprends fidèlement les chiffres, dates, noms et unités.
3. Cite la source après chaque information importante, sous la forme (Nom du document, page X).
4. {regle_absent}
5. Ne devine jamais une valeur. Si les extraits se contredisent, signale-le.
6. Le texte vient d'un PDF et peut être mal mis en forme (tableaux aplatis, coupures). Reconstitue-le prudemment et signale toute incertitude.
7. Réponds dans la langue de la question de l'utilisateur."""


def repondre(question, historique):
    ss = st.session_state
    if not cle_active():
        return "Renseignez d'abord la clé API dans « Configuration de l'IA ».", []

    source = None if ss.scope == TOUS else ss.scope
    passages = rechercher(question, source)

    if not passages and not ss.allow_general:
        return "Importez d'abord un rapport PDF pour que je puisse répondre.", []

    if passages:
        blocs, total = [], 0
        for p in passages:
            bloc = f"[{p['source']} page {p['page']}]\n{p['texte']}"
            if total + len(bloc) > MAX_CONTEXTE:
                break
            blocs.append(bloc)
            total += len(bloc)
        contexte = "\n\n".join(blocs)
    else:
        contexte = "(aucun extrait disponible)"

    message = f"EXTRAITS DU RAPPORT :\n{contexte}\n\nQUESTION : {question}"
    try:
        texte = appeler_ia(
            ss.provider,
            cle_active(),
            construire_consignes(ss.allow_general),
            historique,
            message,
        )
        if not texte:
            texte = "Le fournisseur n'a renvoyé aucune réponse. Reformulez la question."
        return texte, passages
    except Exception as e:
        return f"La réponse n'a pas pu être générée.\n\n{e}", []


def afficher_sources(sources):
    if not sources:
        return
    with st.expander("Sources et références"):
        for src in sources:
            st.markdown(f"**{src['source']}** — page {src['page']}")
            st.caption(src["texte"][:300] + "…")


# =====================================================================
# BARRE DU HAUT
# =====================================================================
top_l, top_r1, top_r2 = st.columns([6, 2, 1])
with top_l:
    st.markdown(
        '<div class="brand">ReportAnalyst</div>',
        unsafe_allow_html=True,
    )
with top_r1:
    st.markdown(
        f'<div class="user-chip"><b>{html.escape(st.session_state.username)}</b></div>',
        unsafe_allow_html=True,
    )
with top_r2:
    if st.button("Quitter", key="logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

reports = st.session_state.metadata.get("reports", [])
noms_rapports = [r["name"] for r in reports]

# =====================================================================
# PAGE UNIQUE : RAPPORT (gauche) + CHAT (droite)
# =====================================================================
col_left, col_right = st.columns([1, 2], gap="large")

# ---------------------------- GAUCHE ---------------------------------
with col_left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Rapport</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Déposez un ou plusieurs PDF : ils sont analysés automatiquement.</div>',
            unsafe_allow_html=True,
        )

        for kind, text in st.session_state.flash:
            getattr(st, kind)(text)
        st.session_state.flash = []

        fichiers = st.file_uploader(
            "Fichiers PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if fichiers:
            with st.spinner("Analyse du rapport en cours…"):
                st.session_state.flash = indexer_fichiers(fichiers)
            st.session_state.uploader_key += 1
            st.rerun()

        if reports:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            for idx, r in enumerate(reports):
                c1, c2 = st.columns([4, 1.4])
                with c1:
                    st.markdown(
                        f'<div class="report-name">{html.escape(r["name"])}</div>'
                        f'<div class="report-meta">{r["pages"]} pages · '
                        f'{html.escape(r.get("date", "—"))}</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("Retirer", key=f"del_{idx}", use_container_width=True):
                        supprimer_rapport(r["name"])
                        st.rerun()

            options = [TOUS] + noms_rapports
            if st.session_state.scope not in options:
                st.session_state.scope = TOUS
            if len(noms_rapports) > 1:
                st.selectbox("Interroger", options, key="scope")
        else:
            st.markdown(
                '<div class="empty-state">Aucun rapport importé.</div>', unsafe_allow_html=True
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Configuration de l\'IA</div>', unsafe_allow_html=True
        )

        st.selectbox("Fournisseur", FOURNISSEURS, key="provider")
        
        cle_actuelle_nom = CLES[st.session_state.provider]
        if not get_secret(cle_actuelle_nom.upper()) and not get_secret("GEMINI_API_KEY"):
            st.text_input(
                "Clé API",
                key=cle_actuelle_nom,
                placeholder="Collez votre clé ici",
            )
        else:
            st.success("🔒 Clé API chargée automatiquement.")

        st.checkbox(
            "Autoriser les connaissances générales de l'IA",
            key="allow_general",
            help="Désactivé : l'IA répond uniquement à partir du rapport.",
        )

        if st.button("Tester la connexion", use_container_width=True):
            if not cle_active():
                st.warning("Aucune clé API active trouvée.")
            else:
                try:
                    with st.spinner("Test en cours…"):
                        appeler_ia(
                            st.session_state.provider,
                            cle_active(),
                            "Réponds en un mot.",
                            [],
                            "Réponds uniquement : OK",
                        )
                    st.success("Connexion réussie.")
                except Exception as e:
                    st.error(str(e))

# ---------------------------- DROITE ---------------------------------
with col_right:
    st.markdown(
        '<div class="section-title">Questions sur le rapport</div>', unsafe_allow_html=True
    )

    boite = st.container(height=600, border=True)
    prompt = st.chat_input("Écrivez votre question…")

    with boite:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                afficher_sources(message.get("sources"))

        if not st.session_state.messages and not prompt:
            if not reports:
                st.markdown(
                    '<div class="empty-state">Importez un rapport à gauche, puis posez votre question ici.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="empty-state">Votre rapport est prêt. Posez votre question ci-dessous.</div>',
                    unsafe_allow_html=True,
                )

        if prompt:
            historique = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyse du rapport et rédaction de la réponse…"):
                    reponse, sources = repondre(prompt, historique)
                st.markdown(reponse)
                afficher_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": reponse, "sources": sources}
            )

    if st.session_state.messages:
        if st.button("Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()