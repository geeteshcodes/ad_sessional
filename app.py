import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from pyECLAT import ECLAT

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0b0f1a;
    color: #e8e6f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #e8e6f0 !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0f172a 50%, #1a1f35 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 12px 12px 0 0;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #a5b4fc;
    line-height: 1;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #c7d2fe;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* Algorithm cards */
.algo-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.8rem;
}
.badge-apriori  { background: rgba(99,102,241,0.2);  color: #818cf8; border: 1px solid rgba(99,102,241,0.4); }
.badge-fpgrowth { background: rgba(168,85,247,0.2);  color: #c084fc; border: 1px solid rgba(168,85,247,0.4); }
.badge-eclat    { background: rgba(20,184,166,0.2);   color: #2dd4bf; border: 1px solid rgba(20,184,166,0.4); }

/* Stat row in algo panels */
.stat-row {
    display: flex;
    gap: 2rem;
    background: #0b0f1a;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    border: 1px solid #1e293b;
    font-family: 'JetBrains Mono', monospace;
}
.stat-item .s-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.stat-item .s-value { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }

/* Dataframe tweaks */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Comparison table */
.cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    border-radius: 12px;
    overflow: hidden;
}
.cmp-table th {
    background: #1e293b;
    color: #94a3b8;
    padding: 0.8rem 1.2rem;
    text-align: right;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.cmp-table th:first-child { text-align: left; }
.cmp-table td {
    background: #111827;
    padding: 0.75rem 1.2rem;
    text-align: right;
    border-top: 1px solid #1e293b;
    color: #e2e8f0;
}
.cmp-table td:first-child { text-align: left; color: #94a3b8; }
.cmp-table tr:hover td { background: #161f2e; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Slider + widgets */
.stSlider > div > div > div { background: #6366f1 !important; }
.stSelectbox select, .stMultiselect > div { background: #111827 !important; }

/* Info box */
.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    color: #a5b4fc;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare(file_bytes):
    df_raw = pd.read_csv(file_bytes, header=None)
    transactions = []
    for _, row in df_raw.iterrows():
        items = [str(i).strip() for i in row if pd.notna(i) and str(i).strip() != 'nan']
        if items:
            transactions.append(items)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    # Convert column names to plain strings to avoid cache serialization issues
    col_names = [str(c) for c in te.columns_]
    df = pd.DataFrame(te_array, columns=col_names)
    return transactions, df, col_names


def run_apriori(df, min_sup, min_conf):
    t0 = time.time()
    freq = apriori(df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
    elapsed = round(time.time() - t0, 4)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    return freq, rules, elapsed


def run_fpgrowth(df, min_sup, min_conf):
    t0 = time.time()
    freq = fpgrowth(df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
    elapsed = round(time.time() - t0, 4)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    return freq, rules, elapsed


def run_eclat(transactions, min_sup):
    t0 = time.time()
    df_ec = pd.DataFrame(transactions)
    eclat_model = ECLAT(data=df_ec, verbose=False)
    _, supports = eclat_model.fit(min_support=min_sup, min_combination=1, max_combination=2)
    elapsed = round(time.time() - t0, 4)
    eclat_df = pd.DataFrame({
        'itemset': list(supports.keys()),
        'support': list(supports.values())
    }).sort_values('support', ascending=False).reset_index(drop=True)
    return eclat_df, elapsed


def fmt_rules(rules_df):
    display = rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    display['antecedents'] = display['antecedents'].apply(lambda x: ', '.join(list(x)))
    display['consequents'] = display['consequents'].apply(lambda x: ', '.join(list(x)))
    display['support']     = display['support'].map('{:.4f}'.format)
    display['confidence']  = display['confidence'].map('{:.4f}'.format)
    display['lift']        = display['lift'].map('{:.4f}'.format)
    return display


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV (Market Basket)",
        type=["csv"],
        help="Each row = one transaction. No header. Items in columns."
    )

    st.markdown("#### Thresholds")
    min_support    = st.slider("Min Support",    0.01, 0.30, 0.04, 0.01,
                               help="Minimum fraction of transactions an itemset must appear in.")
    min_confidence = st.slider("Min Confidence", 0.05, 0.80, 0.20, 0.05,
                               help="Minimum rule accuracy threshold.")

    st.markdown("#### Algorithms")
    run_ap = st.checkbox("Apriori",   value=True)
    run_fp = st.checkbox("FP-Growth", value=True)
    run_ec = st.checkbox("ECLAT",     value=True)

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.6'>
    <b style='color:#64748b'>Support</b> — itemset frequency<br>
    <b style='color:#64748b'>Confidence</b> — rule reliability<br>
    <b style='color:#64748b'>Lift</b> — genuine vs coincidence
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛒 Market Basket Analysis</h1>
  <p>Discover hidden purchasing patterns using Apriori, FP-Growth & ECLAT algorithms</p>
</div>
""", unsafe_allow_html=True)


# ── No file uploaded ──────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="info-box">
    📂 Upload a <b>Market Basket CSV</b> using the sidebar to get started.<br>
    Each row should be one transaction, items in columns, no header row.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="label">Step 1</div>
          <div style="color:#a5b4fc; font-size:1rem; font-weight:700; margin-top:0.4rem">Upload CSV</div>
          <div style="color:#64748b; font-size:0.82rem; margin-top:0.3rem">Use the sidebar file uploader</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="label">Step 2</div>
          <div style="color:#a5b4fc; font-size:1rem; font-weight:700; margin-top:0.4rem">Set Thresholds</div>
          <div style="color:#64748b; font-size:0.82rem; margin-top:0.3rem">Adjust support & confidence sliders</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="label">Step 3</div>
          <div style="color:#a5b4fc; font-size:1rem; font-weight:700; margin-top:0.4rem">Run Analysis</div>
          <div style="color:#64748b; font-size:0.82rem; margin-top:0.3rem">Click ▶ Run Analysis button</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
transactions, df_encoded, all_items = load_and_prepare(uploaded)
basket_sizes = [len(t) for t in transactions]

st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Transactions</div>
    <div class="value">{len(transactions):,}</div>
  </div>
  <div class="metric-card">
    <div class="label">Unique Items</div>
    <div class="value">{len(all_items):,}</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg Basket Size</div>
    <div class="value">{np.mean(basket_sizes):.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Max Basket Size</div>
    <div class="value">{max(basket_sizes)}</div>
  </div>
  <div class="metric-card">
    <div class="label">Min Support</div>
    <div class="value">{min_support}</div>
  </div>
  <div class="metric-card">
    <div class="label">Min Confidence</div>
    <div class="value">{min_confidence}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Basket size distribution chart
fig_dist = px.histogram(
    x=basket_sizes, nbins=20,
    labels={'x': 'Items per Transaction', 'y': 'Count'},
    title="Basket Size Distribution",
    color_discrete_sequence=['#6366f1']
)
fig_dist.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Syne'),
    title_font=dict(color='#c7d2fe', size=14),
    xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
    yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
    bargap=0.1, margin=dict(t=40, b=20, l=20, r=20),
    height=250,
)
st.plotly_chart(fig_dist, use_container_width=True)


# ── Run Analysis ──────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div class="info-box">
    ✅ Data loaded successfully. Click <b>▶ Run Analysis</b> in the sidebar to mine association rules.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

results = {}

with st.spinner("Mining frequent itemsets..."):
    if run_ap:
        freq_ap, rules_ap, ap_time = run_apriori(df_encoded, min_support, min_confidence)
        results['apriori'] = dict(freq=freq_ap, rules=rules_ap, time=ap_time)

    if run_fp:
        freq_fp, rules_fp, fp_time = run_fpgrowth(df_encoded, min_support, min_confidence)
        results['fpgrowth'] = dict(freq=freq_fp, rules=rules_fp, time=fp_time)

    if run_ec:
        eclat_df, ec_time = run_eclat(transactions, min_support)
        results['eclat'] = dict(eclat_df=eclat_df, time=ec_time)


# ── Algorithm Results ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Algorithm Results</div>', unsafe_allow_html=True)

tabs = []
tab_labels = []
if run_ap: tab_labels.append("⬡ Apriori")
if run_fp: tab_labels.append("⬡ FP-Growth")
if run_ec: tab_labels.append("⬡ ECLAT")
if tab_labels:
    tabs = st.tabs(tab_labels)

tab_idx = 0

if run_ap and 'apriori' in results:
    with tabs[tab_idx]:
        r = results['apriori']
        st.markdown('<span class="algo-badge badge-apriori">Apriori</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-item"><div class="s-label">Time</div><div class="s-value">{r['time']}s</div></div>
          <div class="stat-item"><div class="s-label">Frequent Itemsets</div><div class="s-value">{len(r['freq'])}</div></div>
          <div class="stat-item"><div class="s-label">Rules Generated</div><div class="s-value">{len(r['rules'])}</div></div>
          <div class="stat-item"><div class="s-label">Top Lift</div><div class="s-value">{r['rules']['lift'].max():.4f}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Top 10 Rules by Lift**")
        st.dataframe(fmt_rules(r['rules']).head(10), use_container_width=True, hide_index=True)
    tab_idx += 1

if run_fp and 'fpgrowth' in results:
    with tabs[tab_idx]:
        r = results['fpgrowth']
        st.markdown('<span class="algo-badge badge-fpgrowth">FP-Growth</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-item"><div class="s-label">Time</div><div class="s-value">{r['time']}s</div></div>
          <div class="stat-item"><div class="s-label">Frequent Itemsets</div><div class="s-value">{len(r['freq'])}</div></div>
          <div class="stat-item"><div class="s-label">Rules Generated</div><div class="s-value">{len(r['rules'])}</div></div>
          <div class="stat-item"><div class="s-label">Top Lift</div><div class="s-value">{r['rules']['lift'].max():.4f}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Top 10 Rules by Lift**")
        st.dataframe(fmt_rules(r['rules']).head(10), use_container_width=True, hide_index=True)
    tab_idx += 1

if run_ec and 'eclat' in results:
    with tabs[tab_idx]:
        r = results['eclat']
        st.markdown('<span class="algo-badge badge-eclat">ECLAT</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-item"><div class="s-label">Time</div><div class="s-value">{r['time']}s</div></div>
          <div class="stat-item"><div class="s-label">Frequent Itemsets</div><div class="s-value">{len(r['eclat_df'])}</div></div>
          <div class="stat-item"><div class="s-label">Rules Generated</div><div class="s-value">N/A</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Top 10 Frequent Itemsets**")
        st.dataframe(r['eclat_df'].head(10), use_container_width=True, hide_index=True)
    tab_idx += 1


# ── Comparison Table ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Algorithm Comparison</div>', unsafe_allow_html=True)

ap_r = results.get('apriori', {})
fp_r = results.get('fpgrowth', {})
ec_r = results.get('eclat', {})

def cell(val, fallback="—"):
    return val if val is not None else fallback

rows = [
    ("Execution Time (s)",
     cell(ap_r.get('time')), cell(fp_r.get('time')), cell(ec_r.get('time'))),
    ("Frequent Itemsets",
     cell(len(ap_r['freq']) if ap_r else None),
     cell(len(fp_r['freq']) if fp_r else None),
     cell(len(ec_r['eclat_df']) if ec_r else None)),
    ("Rules Generated",
     cell(len(ap_r['rules']) if ap_r else None),
     cell(len(fp_r['rules']) if fp_r else None),
     "N/A"),
    ("Top Lift",
     f"{ap_r['rules']['lift'].max():.4f}" if ap_r else "—",
     f"{fp_r['rules']['lift'].max():.4f}" if fp_r else "—",
     "N/A"),
]

table_html = """
<table class="cmp-table">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Apriori</th>
      <th>FP-Growth</th>
      <th>ECLAT</th>
    </tr>
  </thead>
  <tbody>
"""
for label, a, b, c in rows:
    table_html += f"<tr><td>{label}</td><td>{a}</td><td>{b}</td><td>{c}</td></tr>"
table_html += "</tbody></table>"
st.markdown(table_html, unsafe_allow_html=True)


# ── Visualisations ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Visualisations</div>', unsafe_allow_html=True)

vcol1, vcol2 = st.columns(2)

# Speed comparison bar chart
with vcol1:
    speed_data = {}
    if ap_r: speed_data['Apriori']   = ap_r['time']
    if fp_r: speed_data['FP-Growth'] = fp_r['time']
    if ec_r: speed_data['ECLAT']     = ec_r['time']

    if speed_data:
        fig_speed = go.Figure(go.Bar(
            x=list(speed_data.keys()),
            y=list(speed_data.values()),
            marker_color=['#6366f1','#a78bfa','#2dd4bf'][:len(speed_data)],
            text=[f"{v}s" for v in speed_data.values()],
            textposition='outside',
            textfont=dict(color='#94a3b8', family='JetBrains Mono'),
        ))
        fig_speed.update_layout(
            title="Execution Time (s)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Syne'),
            title_font=dict(color='#c7d2fe', size=13),
            xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            margin=dict(t=40, b=10, l=10, r=10), height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_speed, use_container_width=True)

# Top 15 items by support
with vcol2:
    item_support = df_encoded.mean().sort_values(ascending=False).head(15)
    fig_items = go.Figure(go.Bar(
        x=item_support.values,
        y=item_support.index,
        orientation='h',
        marker=dict(
            color=item_support.values,
            colorscale=[[0,'#312e81'],[0.5,'#6366f1'],[1,'#a5b4fc']],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in item_support.values],
        textposition='outside',
        textfont=dict(color='#94a3b8', family='JetBrains Mono', size=10),
    ))
    fig_items.update_layout(
        title="Top 15 Items by Support",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Syne'),
        title_font=dict(color='#c7d2fe', size=13),
        xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b', autorange='reversed'),
        margin=dict(t=40, b=10, l=10, r=60), height=340,
    )
    st.plotly_chart(fig_items, use_container_width=True)


# Scatter: Support vs Confidence vs Lift (bubble)
if ap_r and len(ap_r['rules']) > 0:
    rules_viz = ap_r['rules'].copy()
    rules_viz['antecedents_str'] = rules_viz['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_viz['consequents_str'] = rules_viz['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_viz['rule_label'] = rules_viz['antecedents_str'] + ' → ' + rules_viz['consequents_str']

    fig_scatter = px.scatter(
        rules_viz.head(80),
        x='support', y='confidence',
        size='lift', color='lift',
        hover_name='rule_label',
        color_continuous_scale=[[0,'#312e81'],[0.5,'#6366f1'],[1,'#c084fc']],
        size_max=22,
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
        title="Support vs Confidence (bubble size = Lift) — Apriori Rules",
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Syne'),
        title_font=dict(color='#c7d2fe', size=13),
        xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
        coloraxis_colorbar=dict(title='Lift', tickfont=dict(color='#94a3b8')),
        margin=dict(t=50, b=20, l=20, r=20), height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; font-family: JetBrains Mono, monospace; padding: 1rem 0'>
Market Basket Analysis · Apriori · FP-Growth · ECLAT
</div>
""", unsafe_allow_html=True)
