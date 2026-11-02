import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import time
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Paper Detector | Deep Sentinel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN ANIMATED CSS & GLASSMORPHISM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Animated Gradient Banner */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.2); }
        50% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.6); }
        100% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.2); }
    }

    @keyframes floatElement {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(1000%); }
    }

    .hero-container {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #090d16);
        background-size: 400% 400%;
        animation: gradientShift 12s ease infinite;
        border-radius: 20px;
        padding: 35px 30px;
        color: white;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-ai {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        animation: pulseGlow 2s infinite;
        letter-spacing: 0.05em;
        font-size: 0.95rem;
    }

    .badge-human {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        animation: pulseGlow 2s infinite;
        letter-spacing: 0.05em;
        font-size: 0.95rem;
    }

    .sentence-box-ai {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .sentence-box-ai:hover {
        background: rgba(239, 68, 68, 0.25);
        transform: translateX(4px);
    }

    .sentence-box-human {
        background: rgba(16, 185, 129, 0.15);
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .sentence-box-human:hover {
        background: rgba(16, 185, 129, 0.25);
        transform: translateX(4px);
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED MODEL LOADER ---
@st.cache_resource(show_spinner=False)
def load_detector_artifacts():
    possible_roots = [
        Path.cwd(),
        Path.cwd() / "models",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "models",
    ]
    vec_path = None
    mod_path = None

    for root in possible_roots:
        candidate_vec = root / "tfidf_vectorizer.pkl" if (root / "tfidf_vectorizer.pkl").exists() else root / "models" / "tfidf_vectorizer.pkl"
        candidate_mod = root / "detector_model.pkl" if (root / "detector_model.pkl").exists() else root / "models" / "detector_model.pkl"

        if candidate_vec.exists() and candidate_mod.exists():
            vec_path = candidate_vec
            mod_path = candidate_mod
            break

    if not vec_path or not mod_path:
        raise FileNotFoundError("Model artifacts (detector_model.pkl, tfidf_vectorizer.pkl) not found.")

    vectorizer = joblib.load(vec_path)
    model = joblib.load(mod_path)
    return vectorizer, model


# --- HELPER FUNCTIONS ---
def split_into_sentences(text: str):
    """Splits text into meaningful sentences while preserving structure."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw_sentences if len(s.strip()) > 8]


def compute_stylometrics(text: str):
    """Calculates text burstiness, perplexity proxy, and lexical diversity."""
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = split_into_sentences(text)
    
    total_words = len(words)
    total_sentences = max(len(sentences), 1)
    
    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences] if sentences else [0]
    avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
    std_sentence_len = np.std(sentence_lengths) if sentence_lengths else 0
    burstiness = (std_sentence_len / avg_sentence_len) if avg_sentence_len > 0 else 0
    
    vocab_size = len(set(words))
    lexical_diversity = (vocab_size / total_words) if total_words > 0 else 0
    
    return {
        "word_count": total_words,
        "sentence_count": total_sentences,
        "vocab_size": vocab_size,
        "avg_sentence_len": float(round(avg_sentence_len, 1)),
        "burstiness_index": float(round(burstiness, 2)),
        "lexical_diversity": float(round(lexical_diversity * 100, 1))
    }


def analyze_contributing_ngrams(text: str, vectorizer, model, top_n=10):
    """Extracts top positive (AI) and negative (Human) weighted n-grams present in the text."""
    transformed = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    nonzero_indices = transformed.indices
    if len(nonzero_indices) == 0:
        return pd.DataFrame()

    results = []
    for idx in nonzero_indices:
        tfidf_val = transformed[0, idx]
        weight = coefs[idx]
        contribution = tfidf_val * weight
        results.append({
            "token": feature_names[idx],
            "contribution": contribution,
            "weight": weight,
            "impact": "AI Indicator" if contribution > 0 else "Human Indicator"
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="contribution", key=abs, ascending=False).head(top_n)
    return df


# --- PRESET SAMPLES ---
PRESET_SAMPLES = {
    "🤖 AI Scientific Abstract": (
        "In this paper, we present a comprehensive investigation of deep learning paradigms "
        "applied to natural language processing. Furthermore, our proposed architectural framework "
        "leverages multi-head attention mechanisms to achieve state-of-the-art benchmark results. "
        "The empirical findings demonstrate significant enhancements in both computational throughput "
        "and contextual semantic accuracy across various downstream tasks. In conclusion, this study "
        "paves the way for scalable and robust foundation model deployments."
    ),
    "👤 Human Astrophysics Paper": (
        "We report new measurements of cosmic microwave background anisotropy at 148 GHz and 218 GHz "
        "made with the Atacama Cosmology Telescope during the 2008 southern observation season. "
        "Cross-correlating the map noise with cryogenic receiver temperature drifts revealed a minor "
        "low-frequency offset that was corrected using maximum likelihood baseline reconstruction. "
        "Systematic errors from beam asymmetries were bounded below 1.2 percent."
    ),
    "🤖 AI Analytical Essay": (
        "Furthermore, it is essential to consider the multifaceted socioeconomic implications "
        "when evaluating the broader paradigm shift brought by generative intelligence. "
        "It is widely acknowledged that ethical governance must accompany technological advancement. "
        "In summary, continuous vigilance and cross-sector collaboration remain imperative for "
        "sustainable human-centered outcomes."
    ),
    "👤 Human Clinical Observation": (
        "A 54-year-old female presented to the emergency department with sudden onset right-sided hemiparesis "
        "and expressive aphasia starting 90 minutes prior to arrival. Non-contrast cranial CT demonstrated "
        "an early hyperdense middle cerebral artery sign without frank parenchymal hypodensity. Intravenous "
        "alteplase was initiated at 0.9 mg/kg following rapid bedside glycemic verification."
    )
}

# --- HEADER SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 Deep Sentinel AI Paper Detector</div>
    <div class="hero-subtitle">
        High-precision neural artifact detection for scientific papers, academic preprints, and research articles.
        Powered by n-gram hyperplanes and stylometric distribution forensics.
    </div>
</div>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
try:
    vectorizer, model = load_detector_artifacts()
except Exception as e:
    st.error(f"❌ Error loading models: {e}. Please make sure `models/detector_model.pkl` and `models/tfidf_vectorizer.pkl` exist.")
    st.stop()


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Calibration & Presets")
    st.markdown("Select a sample or calibrate sensitivity.")

    sample_choice = st.selectbox(
        "📚 Load Sample Text:",
        ["-- Custom Input --"] + list(PRESET_SAMPLES.keys())
    )

    st.markdown("---")
    st.markdown("### 🎚️ Detection Threshold")
    threshold = st.slider(
        "Decision Boundary (AI cutoff):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Higher values require stronger evidence before labeling text as AI-Generated."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Architecture")
    st.info("""
    - **Engine:** SGDClassifier (Log-loss Hyperplane)
    - **Vocabulary:** 50,000 sublinear n-grams (1, 2)
    - **Classes:** Human Authored (0) vs. AI Generated (1)
    - **Metric Target:** Academic Papers & Preprints
    """)

    st.markdown("---")
    st.caption("⚡ Built with Streamlit, Scikit-Learn & Plotly")


# --- INPUT LOGIC ---
default_text = ""
if sample_choice != "-- Custom Input --":
    default_text = PRESET_SAMPLES[sample_choice]

col_input_left, col_input_right = st.columns([3, 1])

with col_input_left:
    user_text = st.text_area(
        "📄 Enter or Paste Text / Abstract to Scan:",
        value=default_text,
        height=240,
        placeholder="Paste academic paper abstract, introduction, or text snippet here..."
    )

with col_input_right:
    st.markdown("#### 📁 File Upload")
    uploaded_file = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
    if uploaded_file is not None:
        try:
            user_text = uploaded_file.read().decode("utf-8")
            st.success("File uploaded successfully!")
        except Exception:
            st.error("Failed to decode text file.")

    st.markdown("#### ⚡ Quick Actions")
    scan_clicked = st.button("🚀 Run Deep Scan", use_container_width=True, type="primary")


# --- TEXT STATISTICS SUMMARY ---
if user_text.strip():
    stats = compute_stylometrics(user_text)
    
    st.markdown("### 📈 Live Stylometric Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="glass-card"><div style="color:#94a3b8;font-size:0.85rem;">WORDS</div><div class="metric-value">{stats["word_count"]}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="glass-card"><div style="color:#94a3b8;font-size:0.85rem;">SENTENCES</div><div class="metric-value">{stats["sentence_count"]}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="glass-card"><div style="color:#94a3b8;font-size:0.85rem;">AVG SENT. LENGTH</div><div class="metric-value">{stats["avg_sentence_len"]}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="glass-card"><div style="color:#94a3b8;font-size:0.85rem;">BURSTINESS INDEX</div><div class="metric-value">{stats["burstiness_index"]}</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="glass-card"><div style="color:#94a3b8;font-size:0.85rem;">LEXICAL DIVERSITY</div><div class="metric-value">{stats["lexical_diversity"]}%</div></div>', unsafe_allow_html=True)


# --- SCAN EXECUTION & RESULTS ---
if scan_clicked or (user_text.strip() and sample_choice != "-- Custom Input --"):
    if not user_text.strip():
        st.warning("⚠️ Please provide some text to analyze.")
        st.stop()

    # Scanning Animation
    with st.spinner("🔬 Scanning text across 50,000 linguistic hyperplane dimensions..."):
        time.sleep(0.35)
        # Vectorize and Predict
        features = vectorizer.transform([user_text])
        probabilities = model.predict_proba(features)[0]
        human_prob = probabilities[0]
        ai_prob = probabilities[1]

        is_ai = ai_prob >= threshold
        confidence = (ai_prob if is_ai else human_prob) * 100

    # Confetti / Snow Animation trigger based on verdict
    if is_ai and ai_prob > 0.85:
        st.snow()
    elif not is_ai and human_prob > 0.85:
        st.balloons()

    st.markdown("---")
    st.markdown("## 🎯 Diagnostic Verdict")

    v_col1, v_col2 = st.columns([1.1, 1.3])

    with v_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if is_ai:
            st.markdown('<span class="badge-ai">🤖 AI-GENERATED CONTENT DETECTED</span>', unsafe_allow_html=True)
            st.markdown(f"### High likelihood of synthetic generation")
            st.write(f"The text exhibits token transition patterns, uniform syntax, and characteristic n-grams strongly associated with Large Language Models.")
        else:
            st.markdown('<span class="badge-human">👤 HUMAN-AUTHORED TEXT VERIFIED</span>', unsafe_allow_html=True)
            st.markdown(f"### High likelihood of organic human authorship")
            st.write(f"The text exhibits natural stylometric variance, organic lexical diversity, and domain-specific idiosyncratic phraseology.")

        st.markdown("#### Confidence Breakdown:")
        st.progress(int(confidence))
        st.write(f"**Confidence Score:** `{confidence:.2f}%` | **Decision Threshold:** `{threshold * 100:.0f}%`")
        st.write(f"• **AI Probability:** `{ai_prob * 100:.2f}%`")
        st.write(f"• **Human Probability:** `{human_prob * 100:.2f}%`")
        st.markdown('</div>', unsafe_allow_html=True)

    with v_col2:
        # Interactive Animated Plotly Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ai_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AI Probability Score", 'font': {'size': 20, 'color': "#cbd5e1"}},
            number={'suffix': "%", 'font': {'size': 44, 'family': 'JetBrains Mono', 'color': "#ef4444" if is_ai else "#10b981"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "#ef4444" if is_ai else "#10b981"},
                'bgcolor': "rgba(255, 255, 255, 0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.2)",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [35, 65], 'color': 'rgba(234, 179, 8, 0.2)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#38bdf8", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=25, r=25, t=40, b=25),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- N-GRAM FORENSIC EXPLANATION ---
    st.markdown("### 🔍 Forensic Feature Contributions (Key N-Grams)")
    st.caption("Top n-grams identified in your text pulling the prediction towards AI or Human authorship.")

    ngram_df = analyze_contributing_ngrams(user_text, vectorizer, model, top_n=12)

    if not ngram_df.empty:
        col_chart, col_table = st.columns([1.4, 1])

        with col_chart:
            fig_bar = px.bar(
                ngram_df,
                x="contribution",
                y="token",
                orientation="h",
                color="impact",
                color_discrete_map={
                    "AI Indicator": "#ef4444",
                    "Human Indicator": "#10b981"
                },
                title="TF-IDF Weighted Feature Impact on Hyperplane"
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                xaxis_title="Weighted Contribution (Negative = Human, Positive = AI)",
                yaxis_title="Feature / N-Gram",
                height=350,
                margin=dict(l=10, r=10, t=35, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            st.markdown("#### Feature Influence Table")
            display_df = ngram_df[["token", "impact", "contribution"]].copy()
            display_df["contribution"] = display_df["contribution"].round(4)
            st.dataframe(display_df, use_container_width=True, height=310)
    else:
        st.info("No significant n-grams in the 50,000 vocabulary matched this specific text snippet.")

    # --- SENTENCE-LEVEL HEATMAP ---
    st.markdown("### 🧬 Sentence-by-Sentence Breakdown")
    st.caption("Detailed line scan indicating which specific segments lean towards synthetic generation.")

    sentences = split_into_sentences(user_text)
    if sentences:
        sent_features = vectorizer.transform(sentences)
        sent_probs = model.predict_proba(sent_features)

        for idx, (sent, p) in enumerate(zip(sentences, sent_probs), 1):
            h_p, a_p = p[0], p[1]
            sent_is_ai = a_p >= threshold
            box_class = "sentence-box-ai" if sent_is_ai else "sentence-box-human"
            badge = f'<span style="color:{"#ef4444" if sent_is_ai else "#10b981"}; font-weight:700;">[{"AI" if sent_is_ai else "HUMAN"}: {a_p*100:.1f}% AI]</span>'

            st.markdown(
                f'<div class="{box_class}">'
                f'{badge} <span>{sent}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.write(user_text)

else:
    # Empty State Guidance
    st.info("💡 Paste a research abstract above or choose a preset sample from the left sidebar, then click **'🚀 Run Deep Scan'**.")
