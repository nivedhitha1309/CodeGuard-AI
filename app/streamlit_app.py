from pathlib import Path
import html

import pandas as pd
import streamlit as st


# =============================================================================
# CODEGUARD AI — STREAMLIT PROTOTYPE
# Dashboard-first visual redesign
# =============================================================================

st.set_page_config(
    page_title="CodeGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"

FINAL_RISK_FILE = DATA / "fusion" / "requests_final_risk_prioritization.csv"
REFACTOR_FILE = DATA / "fusion" / "requests_refactoring_priorities.csv"
REPORT_FILE = DATA / "fusion" / "requests_codeguard_report.csv"
SUMMARY_FILE = DATA / "fusion" / "requests_codeguard_summary.txt"
METHODOLOGY_FILE = DATA / "fusion" / "requests_methodology_implementation_freeze.txt"


# =============================================================================
# GLOBAL STYLE
# =============================================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

/* -------------------------------------------------------------------------
   GLOBAL BACKGROUND
   ------------------------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.18), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(236,72,153,0.15), transparent 25%),
        radial-gradient(circle at 80% 75%, rgba(6,182,212,0.12), transparent 30%),
        linear-gradient(135deg, #f5f7ff 0%, #eef2ff 42%, #f8fafc 100%);
    font-family: 'Inter', sans-serif;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1500px;
}

/* -------------------------------------------------------------------------
   SIDEBAR
   ------------------------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 30% 0%, rgba(139,92,246,0.30), transparent 25%),
        linear-gradient(180deg, #0b1025 0%, #111936 48%, #172554 100%);
    border-right: 1px solid rgba(255,255,255,0.10);
}

section[data-testid="stSidebar"] * {
    color: #f8fafc;
}

.sidebar-brand {
    padding: 20px 12px 24px 12px;
    margin-bottom: 12px;
}

.sidebar-logo {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    box-shadow: 0 12px 30px rgba(99,102,241,0.35);
    font-size: 28px;
    margin-bottom: 14px;
}

.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: white;
}

.sidebar-subtitle {
    font-size: 11px;
    color: #c7d2fe;
    line-height: 1.5;
    margin-top: 4px;
}

.sidebar-section {
    color: #94a3b8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 22px 0 8px 8px;
}

.status-box {
    margin-top: 18px;
    padding: 14px;
    border-radius: 14px;
    background: linear-gradient(
        135deg,
        rgba(16,185,129,0.16),
        rgba(59,130,246,0.12)
    );
    border: 1px solid rgba(52,211,153,0.25);
}

.status-title {
    font-size: 10px;
    color: #6ee7b7;
    font-weight: 700;
    letter-spacing: 1px;
}

.status-value {
    font-size: 14px;
    color: white;
    font-weight: 700;
    margin-top: 5px;
}

/* -------------------------------------------------------------------------
   HERO
   ------------------------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 38px;
    border-radius: 28px;
    margin-bottom: 26px;
    color: white;
    background:
        radial-gradient(circle at 82% 15%, rgba(255,255,255,0.18), transparent 22%),
        radial-gradient(circle at 15% 85%, rgba(34,211,238,0.20), transparent 25%),
        linear-gradient(120deg, #312e81 0%, #4f46e5 38%, #7c3aed 68%, #db2777 100%);
    box-shadow:
        0 25px 55px rgba(79,70,229,0.24),
        inset 0 1px rgba(255,255,255,0.20);
}

.hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -70px;
    bottom: -100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
}

.hero-kicker {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c4b5fd;
    margin-bottom: 8px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1.05;
    margin-bottom: 12px;
}

.hero-description {
    max-width: 850px;
    font-size: 14px;
    line-height: 1.7;
    color: #e0e7ff;
}

.hero-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.20);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
}

/* -------------------------------------------------------------------------
   SECTION HEADERS
   ------------------------------------------------------------------------- */

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #111827;
    margin-top: 24px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 16px;
}

/* -------------------------------------------------------------------------
   KPI CARDS
   ------------------------------------------------------------------------- */

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}

.kpi-card {
    position: relative;
    overflow: hidden;
    min-height: 145px;
    padding: 22px;
    border-radius: 20px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 12px 30px rgba(15,23,42,0.07);
    backdrop-filter: blur(10px);
}

.kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
}

.kpi-purple::before {
    background: linear-gradient(90deg, #6366f1, #a855f7);
}

.kpi-blue::before {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
}

.kpi-pink::before {
    background: linear-gradient(90deg, #ec4899, #f43f5e);
}

.kpi-orange::before {
    background: linear-gradient(90deg, #f97316, #f59e0b);
}

.kpi-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 10px;
}

.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    color: #111827;
    font-size: 34px;
    font-weight: 700;
}

.kpi-description {
    margin-top: 7px;
    color: #94a3b8;
    font-size: 10px;
}

/* -------------------------------------------------------------------------
   RISK CARDS
   ------------------------------------------------------------------------- */

.risk-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 14px 0 30px 0;
}

.risk-card {
    padding: 20px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 12px 25px rgba(15,23,42,0.10);
}

.risk-low {
    background: linear-gradient(135deg, #059669, #10b981);
}

.risk-medium {
    background: linear-gradient(135deg, #d97706, #f59e0b);
}

.risk-high {
    background: linear-gradient(135deg, #ea580c, #f97316);
}

.risk-critical {
    background: linear-gradient(135deg, #dc2626, #ef4444);
}

.risk-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
}

.risk-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 4px;
    opacity: 0.92;
}

/* -------------------------------------------------------------------------
   SIGNAL CARDS
   ------------------------------------------------------------------------- */

.signal-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}

.signal-card {
    min-height: 165px;
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 12px 30px rgba(15,23,42,0.06);
}

.signal-icon {
    font-size: 28px;
    margin-bottom: 12px;
}

.signal-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #111827;
}

.signal-description {
    color: #64748b;
    font-size: 11px;
    line-height: 1.5;
    margin-top: 8px;
}

.signal-coverage {
    margin-top: 12px;
    font-size: 10px;
    color: #475569;
    font-weight: 700;
}

/* -------------------------------------------------------------------------
   INTELLIGENCE PANEL
   ------------------------------------------------------------------------- */

.intelligence-panel {
    padding: 24px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 90% 20%, rgba(236,72,153,0.12), transparent 25%),
        linear-gradient(135deg, rgba(255,255,255,0.95), rgba(238,242,255,0.90));
    border: 1px solid rgba(129,140,248,0.20);
    box-shadow: 0 18px 40px rgba(30,41,59,0.07);
    margin-bottom: 28px;
}

.intelligence-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 700;
    color: #312e81;
}

.intelligence-text {
    color: #475569;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 8px;
}

/* -------------------------------------------------------------------------
   FOOTER
   ------------------------------------------------------------------------- */

.footer {
    margin-top: 50px;
    padding: 20px;
    border-top: 1px solid rgba(148,163,184,0.20);
    color: #94a3b8;
    text-align: center;
    font-size: 10px;
}

/* -------------------------------------------------------------------------
   STREAMLIT WIDGETS
   ------------------------------------------------------------------------- */

div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,0.25);
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    risk = pd.read_csv(FINAL_RISK_FILE)
    refactor = pd.read_csv(REFACTOR_FILE)

    return risk, refactor


@st.cache_data
def load_summary():
    if SUMMARY_FILE.exists():
        return SUMMARY_FILE.read_text(encoding="utf-8")
    return ""


@st.cache_data
def load_methodology():
    if METHODOLOGY_FILE.exists():
        return METHODOLOGY_FILE.read_text(encoding="utf-8")
    return ""


try:
    risk_df, refactor_df = load_data()
except Exception as exc:
    st.error(f"Unable to load CodeGuard output files: {exc}")
    st.stop()


summary_text = load_summary()
methodology_text = load_methodology()


# =============================================================================
# NORMALIZE DATA
# =============================================================================

risk_df["unified_risk"] = pd.to_numeric(
    risk_df["unified_risk"], errors="coerce"
)

risk_df["unified_risk_level"] = (
    risk_df["unified_risk_level"].fillna("Unknown").astype(str)
)

risk_df["applicability_category"] = (
    risk_df["applicability_category"].fillna("other").astype(str)
)

risk_df["signal_completeness"] = pd.to_numeric(
    risk_df["signal_completeness"], errors="coerce"
).fillna(0)

risk_df["available_signal_count"] = pd.to_numeric(
    risk_df["available_signal_count"], errors="coerce"
).fillna(0).astype(int)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown(
    """
<div class="sidebar-brand">
    <div class="sidebar-logo">🛡️</div>
    <div class="sidebar-title">CodeGuard AI</div>
    <div class="sidebar-subtitle">
        AI-assisted software risk intelligence
    </div>
</div>

<div class="sidebar-section">Navigation</div>
""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Risk Analysis",
        "Refactoring Priorities",
        "File Explorer",
        "Methodology",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"""
<div class="sidebar-section">Case Study</div>

<div style="
    padding:10px 8px;
    font-size:13px;
    font-weight:700;
    color:#ffffff;
">
    psf/requests
</div>

<div class="sidebar-section">Implementation Status</div>

<div class="status-box">
    <div class="status-title">RESEARCH PROTOTYPE</div>
    <div class="status-value">Validated • Prototype Ready</div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# COMMON CALCULATIONS
# =============================================================================

total_files = len(risk_df)

eligible_df = risk_df[
    risk_df["prioritization_eligible"].astype(bool)
].copy()

eligible_count = len(eligible_df)
traceability_count = total_files - eligible_count

average_risk = risk_df["unified_risk"].mean()
maximum_risk = risk_df["unified_risk"].max()

risk_counts = (
    risk_df["unified_risk_level"]
    .value_counts()
    .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
)


# =============================================================================
# DASHBOARD
# =============================================================================

def show_dashboard():

    st.markdown(
        """
<div class="hero">

    <div class="hero-kicker">
        RESEARCH PROTOTYPE • MULTI-SIGNAL SOFTWARE RISK INTELLIGENCE
    </div>

    <div class="hero-title">
        CodeGuard AI
    </div>

    <div class="hero-description">
        A unified software-risk intelligence system that combines
        knowledge concentration, self-admitted technical debt,
        static code risk, and machine-learning defect-risk indicators
        into an availability-aware risk assessment for software repositories.
    </div>

    <div class="hero-badge">
        🧠 4-SIGNAL INTELLIGENCE ENGINE &nbsp; • &nbsp;
        🔬 RESEARCH VALIDATED &nbsp; • &nbsp;
        🛡️ EXPLAINABLE PRIORITIZATION
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # KPI
    # -------------------------------------------------------------------------

    st.markdown(
        """
<div class="section-title">Repository Intelligence</div>
<div class="section-subtitle">
    Current CodeGuard assessment of the psf/requests case study.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="kpi-grid">

    <div class="kpi-card kpi-purple">
        <div class="kpi-label">REPOSITORY FILES</div>
        <div class="kpi-value">{total_files}</div>
        <div class="kpi-description">Files analysed by the prototype</div>
    </div>

    <div class="kpi-card kpi-blue">
        <div class="kpi-label">RISK-APPLICABLE FILES</div>
        <div class="kpi-value">{eligible_count}</div>
        <div class="kpi-description">
            Files eligible for engineering prioritization
        </div>
    </div>

    <div class="kpi-card kpi-pink">
        <div class="kpi-label">AVERAGE UNIFIED RISK</div>
        <div class="kpi-value">{average_risk:.3f}</div>
        <div class="kpi-description">
            Availability-aware fused risk score
        </div>
    </div>

    <div class="kpi-card kpi-orange">
        <div class="kpi-label">MAXIMUM UNIFIED RISK</div>
        <div class="kpi-value">{maximum_risk:.3f}</div>
        <div class="kpi-description">
            Highest observed repository score
        </div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # RISK LANDSCAPE
    # -------------------------------------------------------------------------

    st.markdown(
        """
<div class="section-title">Risk Landscape</div>
<div class="section-subtitle">
    Current prototype risk bands. These thresholds are methodological
    prototype bands, not calibrated production probabilities.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="risk-grid">

    <div class="risk-card risk-low">
        <div class="risk-number">{risk_counts["Low"]}</div>
        <div class="risk-label">LOW RISK FILES</div>
    </div>

    <div class="risk-card risk-medium">
        <div class="risk-number">{risk_counts["Medium"]}</div>
        <div class="risk-label">MEDIUM RISK FILES</div>
    </div>

    <div class="risk-card risk-high">
        <div class="risk-number">{risk_counts["High"]}</div>
        <div class="risk-label">HIGH RISK FILES</div>
    </div>

    <div class="risk-card risk-critical">
        <div class="risk-number">{risk_counts["Critical"]}</div>
        <div class="risk-label">CRITICAL RISK FILES</div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SIGNAL ENGINE
    # -------------------------------------------------------------------------

    st.markdown(
        """
<div class="section-title">Four-Signal Intelligence Engine</div>
<div class="section-subtitle">
    CodeGuard does not rely on a single software metric. Multiple
    complementary signals are combined when available.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="signal-grid">

    <div class="signal-card">
        <div class="signal-icon">🧑‍💻</div>
        <div class="signal-name">Knowledge Concentration</div>
        <div class="signal-description">
            Bus-factor analysis identifies contributor concentration
            and potential ownership risk.
        </div>
        <div class="signal-coverage">● 284 / 284 files</div>
    </div>

    <div class="signal-card">
        <div class="signal-icon">⚠️</div>
        <div class="signal-name">Technical Debt</div>
        <div class="signal-description">
            SATD detection identifies explicit technical-debt
            indicators such as TODO and XXX comments.
        </div>
        <div class="signal-coverage">● 284 / 284 files</div>
    </div>

    <div class="signal-card">
        <div class="signal-icon">📐</div>
        <div class="signal-name">Static Code Risk</div>
        <div class="signal-description">
            Radon complexity and AST-derived metrics provide a
            heuristic code-risk signal.
        </div>
        <div class="signal-coverage">● 35 / 284 files</div>
    </div>

    <div class="signal-card">
        <div class="signal-icon">🤖</div>
        <div class="signal-name">ML Defect Risk</div>
        <div class="signal-description">
            Temporal machine-learning features estimate historical
            defect-risk patterns.
        </div>
        <div class="signal-coverage">● 124 / 284 files</div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # INTELLIGENCE PANEL
    # -------------------------------------------------------------------------

    st.markdown(
        """
<div class="intelligence-panel">

    <div class="intelligence-title">
        🧠 Why CodeGuard is different
    </div>

    <div class="intelligence-text">
        Traditional software-quality tools often examine one dimension
        at a time. CodeGuard combines repository history, technical debt,
        static complexity, and temporal machine-learning signals into a
        single availability-aware assessment.
    </div>

    <div class="intelligence-text">
        Missing signals are not automatically treated as zero risk.
        Instead, the fusion layer renormalizes across the signals that
        are actually available for each file.
    </div>

    <div class="intelligence-text">
        The final prioritization layer then separates engineering-relevant
        files from documentation, binary, and other traceability-only
        artifacts and generates rule-based review/refactoring guidance.
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # TOP FILES
    # -------------------------------------------------------------------------

    st.markdown(
        """
<div class="section-title">Top Risk-Applicable Files</div>
<div class="section-subtitle">
    Files currently receiving the highest unified risk scores.
</div>
""",
        unsafe_allow_html=True,
    )

    top = eligible_df.sort_values(
        "unified_risk",
        ascending=False
    ).head(10).copy()

    display = top[
        [
            "file_path",
            "applicability_category",
            "unified_risk",
            "unified_risk_level",
            "available_signal_count",
            "signal_completeness",
        ]
    ].copy()

    display["unified_risk"] = display["unified_risk"].map(
        lambda x: f"{x:.3f}"
    )

    display["signal_completeness"] = display[
        "signal_completeness"
    ].map(lambda x: f"{x:.0%}")

    display["available_signal_count"] = display[
        "available_signal_count"
    ].astype(str)

    display = display.astype(str)

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
    )

    st.markdown(
        """
<div class="footer">
    CodeGuard AI • Research-oriented software risk assessment prototype
    <br>
    Multi-signal analysis • Availability-aware fusion • Refactoring prioritization
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# RISK ANALYSIS
# =============================================================================

def show_risk_analysis():

    st.markdown(
        """
<div class="hero">
    <div class="hero-kicker">CODEGUARD INTELLIGENCE</div>
    <div class="hero-title">Risk Analysis</div>
    <div class="hero-description">
        Explore how the unified risk landscape is distributed across
        the repository and how many signals contribute to each file.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Files", total_files)

    with col2:
        st.metric("Eligible", eligible_count)

    with col3:
        st.metric("Traceability Only", traceability_count)

    st.markdown("### Risk Distribution")

    chart_df = risk_counts.rename_axis("Risk Level").reset_index()
    chart_df.columns = ["Risk Level", "Files"]

    st.bar_chart(
        chart_df.set_index("Risk Level"),
        width="stretch",
    )

    st.markdown("### Signal Completeness")

    completeness = (
        risk_df["available_signal_count"]
        .value_counts()
        .sort_index()
    )

    completeness_df = pd.DataFrame(
        {
            "Signals Available": completeness.index.astype(str),
            "Files": completeness.values.astype(int),
        }
    )

    st.dataframe(
        completeness_df.astype(str),
        hide_index=True,
        width="stretch",
    )


# =============================================================================
# REFACTORING PRIORITIES
# =============================================================================

def show_refactoring():

    st.markdown(
        """
<div class="hero">
    <div class="hero-kicker">ENGINEERING ACTION LAYER</div>
    <div class="hero-title">Refactoring Priorities</div>
    <div class="hero-description">
        CodeGuard converts observed risk signals into explainable,
        rule-based engineering review recommendations.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    eligible = refactor_df[
        refactor_df["prioritization_eligible"].astype(bool)
    ].copy()

    candidate_count = (
        eligible["final_action_status"]
        .astype(str)
        .eq("Refactoring/review candidate")
        .sum()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Monitor", 166)

    with c2:
        st.metric("Refactoring Candidates", candidate_count)

    with c3:
        st.metric("Traceability Only", traceability_count)

    st.markdown("### Priority Queue")

    columns = [
        "file_path",
        "applicability_category",
        "unified_risk",
        "unified_risk_level",
        "review_priority",
        "dominant_signal",
        "primary_recommendation",
    ]

    available_columns = [
        c for c in columns if c in refactor_df.columns
    ]

    table = refactor_df[
        available_columns
    ].head(30).copy()

    if "unified_risk" in table.columns:
        table["unified_risk"] = pd.to_numeric(
            table["unified_risk"],
            errors="coerce",
        ).map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
        )

    table = table.fillna("-").astype(str)

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=500,
    )


# =============================================================================
# FILE EXPLORER
# =============================================================================

def show_file_explorer():

    st.markdown(
        """
<div class="hero">
    <div class="hero-kicker">REPOSITORY INTELLIGENCE CONSOLE</div>
    <div class="hero-title">File Explorer</div>
    <div class="hero-description">
        Search the repository and inspect the unified risk signals
        associated with individual files.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "🔎 Search file",
        placeholder="e.g. models.py, auth.py, requests/",
    )

    filtered = risk_df.copy()

    if search.strip():
        mask = filtered["file_path"].str.contains(
            search.strip(),
            case=False,
            na=False,
        )
        filtered = filtered[mask]

    filtered = filtered.sort_values(
        "unified_risk",
        ascending=False,
    )

    display_columns = [
        "file_path",
        "applicability_category",
        "unified_risk",
        "unified_risk_level",
        "available_signal_count",
        "signal_completeness",
        "signals_used",
    ]

    display_columns = [
        c for c in display_columns if c in filtered.columns
    ]

    table = filtered[display_columns].head(100).copy()

    if "unified_risk" in table.columns:
        table["unified_risk"] = pd.to_numeric(
            table["unified_risk"],
            errors="coerce",
        ).map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
        )

    if "signal_completeness" in table.columns:
        table["signal_completeness"] = pd.to_numeric(
            table["signal_completeness"],
            errors="coerce",
        ).map(
            lambda x: f"{x:.0%}" if pd.notna(x) else "-"
        )

    table = table.fillna("-").astype(str)

    st.caption(f"{len(filtered)} matching files")

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=550,
    )


# =============================================================================
# METHODOLOGY
# =============================================================================

def show_methodology():

    st.markdown(
        """
<div class="hero">
    <div class="hero-kicker">REPRODUCIBLE RESEARCH PROTOTYPE</div>
    <div class="hero-title">Methodology</div>
    <div class="hero-description">
        CodeGuard's current implementation methodology,
        data coverage, machine-learning evaluation, and fusion design.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
### Four Core Signals

| Signal | Purpose |
|---|---|
| 🧑‍💻 Bus Factor | Knowledge concentration |
| ⚠️ SATD | Self-admitted technical debt |
| 📐 Code Risk | Static complexity heuristic |
| 🤖 ML Risk | Temporal defect-risk indicator |

### Fusion

The current prototype uses:

- Equal baseline weights of 0.25
- Availability-aware renormalization
- Missing signals are not treated as zero
- No learned fusion weights
- No production calibration

### Machine Learning

The final temporal dataset contains:

- 574 events
- 26 positive events
- 548 negative events
- Chronological train/validation/test split
- Random Forest and Logistic Regression baselines

The final test set contains only 5 positive events, so the reported
test metrics should be interpreted cautiously.

### Important Scope

CodeGuard is currently a research-oriented prototype rather than
a production software engineering platform.

The current case study is:

**psf/requests**
""",
    )

    if methodology_text:

        with st.expander("View frozen implementation methodology"):

            st.text(methodology_text)


# =============================================================================
# ROUTING
# =============================================================================

if page == "Dashboard":
    show_dashboard()

elif page == "Risk Analysis":
    show_risk_analysis()

elif page == "Refactoring Priorities":
    show_refactoring()

elif page == "File Explorer":
    show_file_explorer()

elif page == "Methodology":
    show_methodology()