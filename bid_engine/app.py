# app.py  —  BidIQ | AI-Powered Bid Response Engine
import os, json, tempfile
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="BidIQ — Bid Response Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Rajdhani:wght@500;600;700&display=swap');
:root{--lava-core:#FF4500;--lava-hot:#FF6B1A;--lava-mid:#FF8C00;--lava-glow:#FFA500;
--lava-ember:#FFD700;--obsidian:#080808;--crust-dark:#100E0C;--crust-mid:#1A1612;
--crust-light:#241E18;--crust-border:#3D2E1E;--ash-text:#E8D5C0;--ash-muted:#9B8878;
--ash-dim:#5C4A3A;}
html,body,.stApp{background:var(--obsidian)!important;font-family:'Inter',sans-serif!important;color:var(--ash-text)!important;}
.stApp::before{content:'';position:fixed;top:0;left:0;right:0;bottom:0;
background:radial-gradient(ellipse 80% 50% at 20% 80%,rgba(255,69,0,.08) 0%,transparent 60%),
radial-gradient(ellipse 60% 40% at 80% 20%,rgba(255,140,0,.06) 0%,transparent 50%);
pointer-events:none;z-index:0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#120D09 0%,#1A1208 40%,#0F0B07 100%)!important;border-right:1px solid var(--crust-border)!important;}
[data-testid="stSidebar"]::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--lava-core),var(--lava-mid),var(--lava-ember));}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown{color:var(--ash-text)!important;}
.main .block-container{background:transparent!important;padding-top:1.5rem!important;}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;letter-spacing:.04em!important;}
h1{font-size:2.4rem!important;font-weight:700!important;}h2{font-size:1.7rem!important;font-weight:600!important;}h3{font-size:1.3rem!important;font-weight:600!important;}
.fire-title{font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:800;
background:linear-gradient(135deg,#FF4500 0%,#FF6B1A 25%,#FF8C00 50%,#FFD700 75%,#FF8C00 100%);
background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
background-clip:text;animation:fireShimmer 4s linear infinite;letter-spacing:.05em;line-height:1.1;}
@keyframes fireShimmer{0%{background-position:0% center}100%{background-position:200% center}}
.lava-divider{height:2px;background:linear-gradient(90deg,transparent,var(--lava-core) 20%,var(--lava-mid) 50%,var(--lava-glow) 80%,transparent);margin:1rem 0;border-radius:2px;}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(26,18,8,.95) 0%,rgba(36,26,16,.9) 100%)!important;border:1px solid var(--crust-border)!important;border-top:2px solid var(--lava-core)!important;border-radius:10px!important;padding:14px 18px!important;box-shadow:0 4px 24px rgba(255,69,0,.12),inset 0 1px 0 rgba(255,140,0,.1)!important;}
[data-testid="stMetricLabel"]{color:var(--ash-muted)!important;font-size:.75rem!important;font-weight:600!important;letter-spacing:.1em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:var(--lava-glow)!important;font-family:'Rajdhani',sans-serif!important;font-weight:700!important;font-size:1.6rem!important;}
.stTabs [data-baseweb="tab-list"]{background:linear-gradient(90deg,var(--crust-dark),var(--crust-mid))!important;border:1px solid var(--crust-border)!important;border-radius:10px!important;padding:5px!important;gap:4px!important;}
.stTabs [data-baseweb="tab"]{color:var(--ash-muted)!important;border-radius:8px!important;font-weight:500!important;font-size:.83rem!important;letter-spacing:.02em!important;transition:all .2s ease!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(255,69,0,.3),rgba(255,140,0,.2))!important;color:var(--lava-glow)!important;border:1px solid rgba(255,100,0,.4)!important;box-shadow:0 0 12px rgba(255,69,0,.2)!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--lava-core) 0%,var(--lava-hot) 40%,var(--lava-mid) 80%,var(--lava-glow) 100%)!important;background-size:200% auto!important;border:none!important;color:#0A0600!important;font-weight:700!important;font-family:'Rajdhani',sans-serif!important;font-size:1.05rem!important;letter-spacing:.06em!important;border-radius:8px!important;padding:.55rem 1.5rem!important;transition:all .3s ease!important;box-shadow:0 4px 20px rgba(255,69,0,.4)!important;}
.stButton>button[kind="primary"]:hover{background-position:right center!important;box-shadow:0 6px 28px rgba(255,69,0,.6)!important;transform:translateY(-1px)!important;}
.stButton>button:not([kind="primary"]){background:transparent!important;border:1px solid var(--crust-border)!important;color:var(--ash-text)!important;border-radius:8px!important;transition:all .2s!important;}
.stButton>button:not([kind="primary"]):hover{border-color:var(--lava-core)!important;color:var(--lava-glow)!important;background:rgba(255,69,0,.06)!important;}
[data-testid="stFileUploader"]{background:linear-gradient(135deg,rgba(26,18,8,.8),rgba(16,12,7,.9))!important;border:1.5px dashed var(--crust-border)!important;border-radius:10px!important;transition:border-color .3s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--lava-core)!important;}
[data-testid="stExpander"]{background:linear-gradient(135deg,rgba(20,14,8,.9),rgba(28,20,10,.85))!important;border:1px solid var(--crust-border)!important;border-left:3px solid var(--lava-core)!important;border-radius:8px!important;margin:6px 0!important;}
[data-testid="stExpander"] summary{color:var(--ash-text)!important;font-weight:500!important;}
[data-testid="stAlert"]{border-radius:8px!important;border-left-width:3px!important;}
textarea{background:var(--crust-mid)!important;border:1px solid var(--crust-border)!important;color:var(--ash-text)!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;}
textarea:focus{border-color:var(--lava-core)!important;box-shadow:0 0 0 2px rgba(255,69,0,.2)!important;}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,var(--lava-core),var(--lava-mid),var(--lava-ember))!important;}
[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,rgba(255,69,0,.15),rgba(255,140,0,.1))!important;border:1px solid rgba(255,100,0,.4)!important;color:var(--lava-glow)!important;font-weight:600!important;border-radius:8px!important;width:100%!important;transition:all .2s!important;}
[data-testid="stDownloadButton"]>button:hover{background:linear-gradient(135deg,rgba(255,69,0,.3),rgba(255,140,0,.2))!important;box-shadow:0 4px 16px rgba(255,69,0,.3)!important;}
::-webkit-scrollbar{width:6px;height:6px;}::-webkit-scrollbar-track{background:var(--crust-dark);}::-webkit-scrollbar-thumb{background:linear-gradient(var(--lava-core),var(--lava-mid));border-radius:3px;}
.lava-card{background:linear-gradient(135deg,rgba(26,18,8,.95) 0%,rgba(20,14,8,.9) 100%);border:1px solid var(--crust-border);border-radius:12px;padding:20px 24px;margin:10px 0;box-shadow:0 4px 20px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,120,0,.08);}
.lava-card-accent{border-top:2px solid var(--lava-core);}
.workspace-card{background:linear-gradient(135deg,rgba(26,18,8,.95),rgba(20,14,8,.9));border:1px solid var(--crust-border);border-radius:10px;padding:12px 16px;margin:6px 0;transition:all .2s;}
.workspace-card-active{border-color:var(--lava-core)!important;background:linear-gradient(135deg,rgba(255,69,0,.12),rgba(255,140,0,.06))!important;box-shadow:0 0 16px rgba(255,69,0,.2)!important;}
.go-banner{background:linear-gradient(135deg,#051A0A 0%,#0A2E12 50%,#051A0A 100%);border:1px solid #1A6B30;border-left:4px solid #2ECC71;border-radius:12px;padding:22px 30px;text-align:center;font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:#2ECC71;margin-bottom:20px;box-shadow:0 0 40px rgba(46,204,113,.15);}
.conditional-banner{background:linear-gradient(135deg,#1A1200 0%,#2E2000 50%,#1A1200 100%);border:1px solid #6B5000;border-left:4px solid #F1C40F;border-radius:12px;padding:22px 30px;text-align:center;font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:#F1C40F;margin-bottom:20px;}
.nogo-banner{background:linear-gradient(135deg,#1A0505 0%,#2E0808 50%,#1A0505 100%);border:1px solid var(--lava-core);border-left:4px solid #E74C3C;border-radius:12px;padding:22px 30px;text-align:center;font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:#E74C3C;margin-bottom:20px;}
.req-pass{background:linear-gradient(135deg,rgba(5,20,10,.9),rgba(10,30,15,.85));border-left:3px solid #2ECC71;border-radius:8px;padding:12px 16px;margin:6px 0;color:#C8E6D0;}
.req-weak{background:linear-gradient(135deg,rgba(20,15,0,.9),rgba(30,22,0,.85));border-left:3px solid #F1C40F;border-radius:8px;padding:12px 16px;margin:6px 0;color:#F5E6C0;}
.req-gap{background:linear-gradient(135deg,rgba(20,5,5,.9),rgba(35,8,8,.85));border-left:3px solid var(--lava-core);border-radius:8px;padding:12px 16px;margin:6px 0;color:#E8C4C0;}
.remediation-box{background:rgba(255,69,0,.07);border:1px dashed rgba(255,100,0,.3);border-radius:6px;padding:8px 12px;margin-top:6px;font-size:.78rem;color:#FFA500;}
.cap-chip{background:linear-gradient(135deg,rgba(36,26,16,.95),rgba(26,18,8,.9));border:1px solid var(--crust-border);border-radius:8px;padding:10px 14px;font-size:.82rem;text-align:center;}
footer{visibility:hidden;}#MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "workspaces" not in st.session_state:
    st.session_state.workspaces = {}
if "active_workspace" not in st.session_state:
    st.session_state.active_workspace = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0          # 0-indexed: 0=RFP Overview, 5=Draft Proposal
if "generating_proposal" not in st.session_state:
    st.session_state.generating_proposal = False


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 6px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:800;
        background:linear-gradient(135deg,#FF4500,#FF8C00,#FFD700);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;letter-spacing:.06em;">🔥 BidIQ</div>
      <div style="color:#9B8878;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;margin-top:2px;">
        AI Bid Response Engine</div>
    </div>
    <div class="lava-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#9B8878;font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Upload New RFP / Tender</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf","docx"],
        help="Upload any RFP, RFQ, or Tender document (PDF or DOCX)",
        label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(255,69,0,.1),rgba(255,140,0,.06));
            border:1px solid rgba(255,100,0,.3);border-radius:8px;padding:10px 14px;
            margin:8px 0;font-size:.85rem;color:#FFA500;">
            ✅ &nbsp;<b>{uploaded_file.name}</b><br>
            <span style="color:#9B8878;font-size:.75rem;">{round(uploaded_file.size/1024,1)} KB</span>
        </div>""", unsafe_allow_html=True)
        analyze_btn = st.button("🔥 Ignite Analysis", type="primary", use_container_width=True)
    else:
        st.markdown("""
        <div style="background:rgba(20,14,8,.6);border:1px dashed #3D2E1E;border-radius:8px;
            padding:14px;text-align:center;color:#5C4A3A;font-size:.83rem;margin:8px 0;">
            Drop a PDF or DOCX to begin</div>""", unsafe_allow_html=True)
        analyze_btn = False

    st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

    # ── Workspace list ────────────────────────────────────────────────────────
    if st.session_state.workspaces:
        st.markdown('<p style="color:#9B8878;font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Open Workspaces</p>', unsafe_allow_html=True)

        for ws_name, ws_data in st.session_state.workspaces.items():
            is_active  = (ws_name == st.session_state.active_workspace)
            card_cls   = "workspace-card workspace-card-active" if is_active else "workspace-card"
            _rfp       = ws_data.get("rfp_data", {})
            _comp      = ws_data.get("compliance", {})
            _score     = ws_data.get("win_data", {}).get("final_score", "—")
            _dec       = ws_data.get("win_data", {}).get("decision", "")
            _dc        = "#2ECC71" if "GO" in _dec and "NO" not in _dec and "COND" not in _dec \
                         else "#F1C40F" if "CONDITIONAL" in _dec else "#E74C3C"
            st.markdown(f"""
            <div class="{card_cls}" style="margin-bottom:4px;">
              <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.9rem;
                color:{'#FFA500' if is_active else '#E8D5C0'};margin-bottom:2px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{ws_name}">
                {'▶ ' if is_active else ''}{ws_name[:28]}{'…' if len(ws_name)>28 else ''}
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#9B8878;font-size:.7rem;">{_rfp.get('sector','—')} · {_comp.get('compliance_rate','—')}% comply</span>
                <span style="color:{_dc};font-size:.7rem;font-weight:700;font-family:'Rajdhani',sans-serif;">{_score}/100</span>
              </div>
            </div>""", unsafe_allow_html=True)
            if not is_active:
                if st.button("Switch →", key=f"sw_{ws_name}", use_container_width=True):
                    st.session_state.active_workspace = ws_name
                    st.session_state.active_tab = 0   # reset to overview on workspace switch
                    st.rerun()

        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        with st.expander("🗑️ Remove a Workspace"):
            to_del = st.selectbox("", list(st.session_state.workspaces.keys()), label_visibility="collapsed")
            if st.button("Remove", use_container_width=True):
                del st.session_state.workspaces[to_del]
                if st.session_state.active_workspace == to_del:
                    keys = list(st.session_state.workspaces.keys())
                    st.session_state.active_workspace = keys[-1] if keys else None
                st.rerun()
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

    # ── Historical KPIs ───────────────────────────────────────────────────────
    st.markdown('<p style="color:#9B8878;font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">Historical Database</p>', unsafe_allow_html=True)
    try:
        from utils.data_loader import load_bid_history, load_capability_library
        _db   = load_bid_history()
        _dc2  = load_capability_library()
        _wins = (_db["Outcome"] == "Win").sum()
        _tot  = len(_db)
        _c1, _c2 = st.columns(2)
        _c1.metric("Bids",     _tot)
        _c2.metric("Win Rate", f"{round(_wins/_tot*100,1)}%")
        st.metric("Capabilities", len(_dc2))
    except Exception:
        st.warning("Could not load historical data.")

    st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#5C4A3A;font-size:.72rem;text-align:center;letter-spacing:.05em;padding-top:4px;">TekRowe Hackathon · Problem Statement 1</div>', unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 4px;">
  <div class="fire-title">BidIQ</div>
  <div style="color:#9B8878;font-size:.9rem;letter-spacing:.08em;text-transform:uppercase;margin-top:4px;margin-bottom:4px;">
    AI-Powered Bid &amp; Proposal Response Engine
  </div>
</div>
<div class="lava-divider"></div>
""", unsafe_allow_html=True)


# ── ANALYSIS TRIGGER ──────────────────────────────────────────────────────────
if analyze_btn and uploaded_file:
    from utils.rfp_parser import parse_rfp
    from utils.scorer import check_compliance, calculate_win_probability
    from utils.rag_engine import search_capabilities
    from utils.effort_metrics import estimate_effort_savings

    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("📄 Parsing RFP with AI…"):
            rfp_data = parse_rfp(tmp_path)
        with st.spinner("🔍 Running compliance check…"):
            compliance = check_compliance(rfp_data.get("mandatory_requirements", []))
        effort_metrics = estimate_effort_savings(rfp_data, compliance)
        with st.spinner("📈 Scoring win probability…"):
            win_data = calculate_win_probability(rfp_data, compliance)
        with st.spinner("🔎 Matching capabilities (RAG)…"):
            cap_matches = {
                req: search_capabilities(req, top_k=3)
                for req in rfp_data.get("mandatory_requirements", [])
            }

        ws_name = uploaded_file.name.replace(".pdf","").replace(".docx","")
        base, n = ws_name, 1
        while ws_name in st.session_state.workspaces:
            ws_name = f"{base} ({n})"; n += 1

        st.session_state.workspaces[ws_name] = {
            "rfp_data":      rfp_data,
            "compliance":    compliance,
            "win_data":      win_data,
            "cap_matches":   cap_matches,
            "effort_metrics": effort_metrics,
            "draft_sections": None,
        }
        st.session_state.active_workspace = ws_name
        st.success(f"✅ Workspace **{ws_name}** created.")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)


# ── ACTIVE WORKSPACE ──────────────────────────────────────────────────────────
active = st.session_state.active_workspace
ws     = st.session_state.workspaces.get(active) if active else None

PL_BG   = "rgba(0,0,0,0)"
PL_FONT = dict(color="#E8D5C0", family="Inter")
PL_GRID = "#2A1E12"

if ws:
    rfp  = ws["rfp_data"]
    comp = ws["compliance"]
    win  = ws["win_data"]
    caps = ws["cap_matches"]
    effort = ws.get("effort_metrics", {})

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(255,69,0,.1),rgba(255,140,0,.05));
        border:1px solid rgba(255,100,0,.3);border-radius:8px;padding:8px 16px;
        margin-bottom:12px;display:inline-block;">
      <span style="color:#9B8878;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;">Active Workspace&nbsp;</span>
      <span style="color:#FFA500;font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.95rem;">{active}</span>
      &nbsp;<span style="color:#5C4A3A;font-size:.72rem;">· {len(st.session_state.workspaces)} workspace(s) open</span>
    </div>""", unsafe_allow_html=True)

    # ── If generating flag is set, show a top banner prompting the user to open tab 6 ──
    if st.session_state.generating_proposal and \
       not st.session_state.workspaces[active].get("draft_sections"):
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(255,140,0,.15),rgba(255,69,0,.1));
            border:1px solid rgba(255,140,0,.4);border-radius:10px;padding:14px 20px;
            color:#FFD700;font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
            margin-bottom:12px;text-align:center;">
            🤖 Proposal generation is ready — click the <b>✍️ Draft Proposal</b> tab below to run it
        </div>""", unsafe_allow_html=True)


    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📋 RFP Overview",
        "✅ Compliance",
        "📈 Win Probability",
        "🔎 Capabilities",
        "❓ Q&A Sections",
        "✍️ Draft Proposal",
        "📊 Eval Criteria",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — RFP OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## 📋 RFP Overview")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

        # ── Quick action strip ────────────────────────────────────────────────
        _qa1, _qa2, _qa3 = st.columns([2,2,3])
        with _qa1:
            if st.button("✍️ Generate Proposal Draft", key="quick_gen", use_container_width=True, type="primary"):
                if not ws.get("draft_sections"):
                    st.session_state.generating_proposal = True
                else:
                    st.session_state.active_tab = 5
                st.rerun()
        with _qa2:
            _dec_label = win.get("decision","—") if ws.get("win_data") else "—"
            st.markdown(f"""
            <div style="background:rgba(20,14,8,.7);border:1px solid #3D2E1E;border-radius:8px;
                padding:8px 14px;font-family:'Rajdhani',sans-serif;font-size:.85rem;
                color:#FFA500;text-align:center;">
              Decision: <b>{_dec_label}</b>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Sector",   rfp.get("sector","—"))
        c2.metric("Budget",   rfp.get("budget") or "—")
        c3.metric("Deadline", rfp.get("deadline") or "—")
        c4.metric("Doc Size", f"{rfp.get('char_count',0):,} chars")

        e1,e2,e3,e4 = st.columns(4)
        e1.metric("Manual Baseline", f"{effort.get('manual_baseline_minutes', 0)} min")
        e2.metric("BidIQ Estimate", f"{effort.get('bidiq_minutes', 0)} min")
        e3.metric("Time Saved", f"{effort.get('saved_minutes', 0)} min")
        e4.metric("Effort Reduction", f"{effort.get('reduction_percent', 0)}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([3,2])

        with col_l:
            st.markdown("### 🗒️ Project Summary")
            st.markdown(f"""
            <div class="lava-card lava-card-accent">
              <p style="color:#E8D5C0;line-height:1.75;margin:0;font-size:.95rem;">
                {rfp.get('summary','No summary available.')}
              </p>
            </div>""", unsafe_allow_html=True)

            st.markdown("### ❓ Key Sections to Answer")
            for i,q in enumerate(rfp.get("key_questions",[]),1):
                st.markdown(f"""
                <div style="background:rgba(255,69,0,.05);border-left:2px solid #FF6B1A;
                    border-radius:0 6px 6px 0;padding:9px 14px;margin:5px 0;
                    color:#E8D5C0;font-size:.88rem;">
                  <span style="color:#FF8C00;font-weight:700;font-family:'Rajdhani',sans-serif;margin-right:8px;">Q{i}</span>{q}
                </div>""", unsafe_allow_html=True)

            if rfp.get("submission_requirements"):
                st.markdown("### 📦 Submission Requirements")
                for sr in rfp["submission_requirements"]:
                    st.markdown(f"""
                    <div style="background:rgba(255,140,0,.05);border-left:2px solid #FF8C00;
                        border-radius:0 6px 6px 0;padding:8px 14px;margin:4px 0;
                        color:#E8D5C0;font-size:.86rem;">📌 {sr}</div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown("### 📌 Details")
            st.markdown(f"""
            <div class="lava-card">
              <div style="margin-bottom:14px;">
                <div style="color:#9B8878;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px;">Project Title</div>
                <div style="color:#FFA500;font-weight:600;font-size:.9rem;font-family:'Rajdhani',sans-serif;">{rfp.get('title','—')}</div>
              </div>
              <div style="margin-bottom:14px;">
                <div style="color:#9B8878;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px;">Client</div>
                <div style="color:#E8D5C0;font-size:.9rem;">{rfp.get('client','—')}</div>
              </div>
              <div>
                <div style="color:#9B8878;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px;">Source File</div>
                <div style="color:#E8D5C0;font-size:.85rem;">{rfp.get('source_file','—')}</div>
              </div>
              <div style="margin-top:14px;">
                <div style="color:#9B8878;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px;">Extraction Chunks</div>
                <div style="color:#E8D5C0;font-size:.85rem;">{rfp.get('extraction_chunks',1)}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            ner = rfp.get("ner_entities", {})
            st.markdown("### Named Entity Checks")
            st.markdown(f"""
            <div class="lava-card">
              <div style="color:#9B8878;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Regex NER Signals</div>
              <div style="color:#E8D5C0;font-size:.85rem;line-height:1.7;">
                <b style="color:#FFA500;">Deadlines:</b> {", ".join(ner.get("deadlines", [])[:4]) or "None detected"}<br>
                <b style="color:#FFA500;">Budgets:</b> {", ".join(ner.get("budgets", [])[:4]) or "None detected"}<br>
                <b style="color:#FFA500;">Certifications:</b> {", ".join(ner.get("certifications", [])[:6]) or "None detected"}<br>
                <b style="color:#FFA500;">Organizations:</b> {", ".join(ner.get("organizations", [])[:4]) or "None detected"}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("### Effort Reduction Evidence")
            st.markdown(f"""
            <div class="lava-card">
              <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:8px;">
                <span style="color:#9B8878;font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;">Baseline</span>
                <span style="color:#FFA500;font-weight:700;font-family:'Rajdhani',sans-serif;">{effort.get('reduction_percent',0)}% less manual effort</span>
              </div>
              <div style="color:#E8D5C0;font-size:.85rem;line-height:1.7;">
                Manual exercise: {effort.get('manual_baseline_minutes',0)} minutes<br>
                BidIQ workflow: {effort.get('bidiq_minutes',0)} minutes<br>
                Saved: {effort.get('saved_minutes',0)} minutes
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("### ⚖️ Evaluation Criteria")
            for ec in rfp.get("evaluation_criteria",[]):
                crit = ec.get("criterion","")
                wt   = ec.get("weight","")
                try:    pct = int(str(wt).replace("%","").strip())
                except: pct = 30
                st.markdown(f"""
                <div style="margin:6px 0;">
                  <div style="display:flex;justify-content:space-between;font-size:.82rem;color:#E8D5C0;margin-bottom:3px;">
                    <span>{crit}</span><span style="color:#FFA500;font-weight:600;">{wt}</span>
                  </div>
                  <div style="background:#1A1612;border-radius:3px;height:5px;">
                    <div style="background:linear-gradient(90deg,#FF4500,#FFD700);width:{pct}%;height:5px;border-radius:3px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — COMPLIANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## ✅ Compliance Checklist")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

        weak_cnt = comp["total"] - comp["passed"] - comp.get("critical_gaps",0)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Requirements", comp["total"])
        c2.metric("✅ Passed",           comp["passed"])
        c3.metric("⚠️ Weak Matches",     weak_cnt)
        c4.metric("❌ Critical Gaps",     comp.get("critical_gaps",0))

        col_chart, col_list = st.columns([1,2])

        with col_chart:
            fig_d = go.Figure(go.Pie(
                labels=["Compliant","Weak Match","Critical Gap"],
                values=[comp["passed"], weak_cnt, comp.get("critical_gaps",0)],
                hole=0.68,
                marker=dict(colors=["#FF6B1A","#F1C40F","#2A1E12"],
                            line=dict(color="#080808",width=2)),
                textinfo="none",
            ))
            fig_d.add_annotation(text=f"<b>{comp['compliance_rate']}%</b>",x=.5,y=.55,
                font=dict(size=28,color="#FFD700",family="Rajdhani"),showarrow=False)
            fig_d.add_annotation(text="compliant",x=.5,y=.38,
                font=dict(size=12,color="#9B8878",family="Inter"),showarrow=False)
            fig_d.update_layout(paper_bgcolor=PL_BG,plot_bgcolor=PL_BG,showlegend=True,
                legend=dict(font=dict(color="#9B8878",size=10),bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10,b=10,l=10,r=10),height=240)
            st.plotly_chart(fig_d, use_container_width=True)

        with col_list:
            st.markdown("### Requirement Breakdown")
            for item in comp["items"]:
                if "Pass" in item["status"]:
                    css,icon,cc = "req-pass","✅","#2ECC71"
                elif "Weak" in item["status"]:
                    css,icon,cc = "req-weak","⚠️","#F1C40F"
                else:
                    css,icon,cc = "req-gap","❌","#FF4500"
                rem_html = ""
                if item.get("remediation"):
                    rem_html = f'<div class="remediation-box">🔧 <b>Action Required:</b> {item["remediation"]}</div>'
                st.markdown(f"""
                <div class="{css}">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;font-size:.88rem;">{icon} {item['requirement'][:80]}</span>
                    <span style="color:{cc};font-size:.75rem;font-weight:700;font-family:'Rajdhani',sans-serif;">{item['confidence']}</span>
                  </div>
                  <div style="color:#9B8878;font-size:.78rem;margin-top:4px;">→ {item['best_match'][:100]}</div>
                  {rem_html}
                </div>""", unsafe_allow_html=True)

        if comp.get("critical_gaps",0) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,69,0,.1),rgba(255,100,0,.06));
                border:1px solid rgba(255,69,0,.3);border-radius:8px;padding:14px 18px;color:#FFA500;font-size:.9rem;">
                ⚠️ &nbsp;<b>{comp['critical_gaps']} critical gap(s) detected.</b>
                Address the flagged actions above before submission to avoid disqualification.
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — WIN PROBABILITY
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("## 📈 Win Probability Dashboard")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

        with st.expander("⚙️ Adjust Competitor Presence (optional override)", expanded=False):
            comp_override = st.selectbox(
                "Override competitor presence:",
                ["Auto (sector heuristic)","Low","Medium","High"],
                key="comp_override"
            )
            if comp_override != "Auto (sector heuristic)":
                from utils.scorer import calculate_win_probability
                win = calculate_win_probability(rfp, comp, competitor_override=comp_override)
                st.session_state.workspaces[active]["win_data"] = win
                st.success(f"Competitor level set to **{comp_override}**. Scores updated.")

        decision = win["decision"]
        bcls = "nogo-banner" if "NO-GO" in decision else "conditional-banner" if "CONDITIONAL" in decision else "go-banner"
        st.markdown(
            f'<div class="{bcls}">{decision}&nbsp;&nbsp;|&nbsp;&nbsp;Score: {win["final_score"]} / 100</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 📝 Decision Rationale")
        for reason in win.get("reasons",[]):
            color = "#2ECC71" if reason.startswith("✅") else "#F1C40F" if reason.startswith("⚠️") else "#FF4500"
            st.markdown(f"""
            <div style="background:rgba(20,14,8,.7);border-left:3px solid {color};
                border-radius:0 8px 8px 0;padding:10px 16px;margin:5px 0;
                color:#E8D5C0;font-size:.88rem;">{reason}</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_radar, col_cards = st.columns([1,1])

        with col_radar:
            st.markdown("### 🕸️ Score Radar")
            breakdown = win["breakdown"]
            factors   = list(breakdown.keys())
            scores    = [breakdown[f]["score"] for f in factors]
            fc = factors + [factors[0]]
            sc = scores  + [scores[0]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=sc, theta=fc, fill="toself",
                fillcolor="rgba(255,100,0,.15)",
                line=dict(color="#FF6B1A",width=2),
                marker=dict(color="#FFD700",size=6), name="Score"))
            fig_r.add_trace(go.Scatterpolar(r=[65]*len(fc), theta=fc,
                line=dict(color="#FFD700",width=1,dash="dot"),
                fill=None, name="GO Threshold (65)", marker=dict(size=0)))
            fig_r.update_layout(paper_bgcolor=PL_BG, plot_bgcolor=PL_BG,
                polar=dict(bgcolor="rgba(16,12,8,.6)",
                    radialaxis=dict(visible=True,range=[0,100],gridcolor="#2A1E12",
                        tickfont=dict(color="#9B8878",size=9),tickcolor="#3D2E1E",linecolor="#3D2E1E"),
                    angularaxis=dict(gridcolor="#2A1E12",tickfont=dict(color="#E8D5C0",size=10),linecolor="#3D2E1E")),
                legend=dict(font=dict(color="#9B8878",size=10),bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20,b=20,l=20,r=20), height=340, font=PL_FONT)
            st.plotly_chart(fig_r, use_container_width=True)

        with col_cards:
            st.markdown("### Factor Scores")
            for factor, data in breakdown.items():
                s    = data["score"]
                cc   = "#2ECC71" if s>=65 else "#FF8C00" if s>=45 else "#FF4500"
                xtra = ""
                if "sample" in data: xtra += f" · n={data['sample']}"
                if "level"  in data: xtra += f" · {data['level']} competition"
                st.markdown(f"""
                <div class="lava-card" style="padding:12px 16px;margin:5px 0;">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <span style="font-size:.82rem;color:#9B8878;">{factor}{xtra}</span>
                    <span style="color:{cc};font-size:1.3rem;font-weight:700;font-family:'Rajdhani',sans-serif;">{s:.1f}</span>
                  </div>
                  <div style="background:#1A1612;border-radius:2px;height:4px;margin-top:6px;">
                    <div style="background:{cc};width:{int(s)}%;height:4px;border-radius:2px;"></div>
                  </div>
                  <div style="color:#5C4A3A;font-size:.7rem;margin-top:4px;">weight {data['weight']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🌋 Historical Win Rate by Sector")
        try:
            from utils.data_loader import load_bid_history
            _df = load_bid_history()
            sg  = (_df.groupby("Sector")
                   .apply(lambda x: round((x["Outcome"]=="Win").sum()/len(x)*100,1))
                   .reset_index())
            sg.columns = ["Sector","Win Rate (%)"]
            sg = sg.sort_values("Win Rate (%)", ascending=True)
            fig_s = go.Figure(go.Bar(
                x=sg["Win Rate (%)"], y=sg["Sector"], orientation="h",
                marker=dict(color=sg["Win Rate (%)"],
                    colorscale=[[0,"#FF4500"],[.5,"#FF8C00"],[1,"#FFD700"]],
                    line=dict(color="#080808",width=.5)),
                text=[f"{v}%" for v in sg["Win Rate (%)"]],
                textposition="outside", textfont=dict(color="#9B8878",size=11)))
            fig_s.add_vline(x=65,line_dash="dot",line_color="#FFD700",line_width=1.5,
                annotation_text="GO threshold",annotation_position="top",
                annotation_font=dict(color="#FFD700",size=10))
            fig_s.update_layout(paper_bgcolor=PL_BG,plot_bgcolor=PL_BG,
                xaxis=dict(range=[0,110],color="#9B8878",gridcolor=PL_GRID,gridwidth=.5,zeroline=False),
                yaxis=dict(color="#E8D5C0"),
                height=340,margin=dict(t=10,b=10,l=10,r=60),font=PL_FONT)
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception:
            st.info("Could not load sector chart.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — CAPABILITIES
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("## 🔎 Capability Matches (RAG)")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#9B8878;font-size:.85rem;margin-bottom:1rem;">Top 3 most relevant past projects retrieved from the Capability Library via FAISS semantic search for each mandatory requirement.</div>', unsafe_allow_html=True)

        for req, matches in caps.items():
            with st.expander(f"🔒 {req[:90]}{'...' if len(req)>90 else ''}"):
                if not matches:
                    st.warning("No matches found in Capability Library.")
                    continue
                cols = st.columns(len(matches))
                for col, m in zip(cols, matches):
                    sc   = m["score"]
                    cc   = "#2ECC71" if sc>=.25 else "#FF8C00" if sc>=.15 else "#FF4500"
                    cert = m["certification"] if str(m["certification"]) != "nan" else "None"
                    col.markdown(f"""
                    <div class="lava-card" style="padding:12px 14px;">
                      <div style="color:{cc};font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1rem;margin-bottom:4px;">[{m['cap_id']}]</div>
                      <div style="color:#FFA500;font-weight:600;font-size:.85rem;margin-bottom:6px;">{m['domain']}</div>
                      <div style="color:#E8D5C0;font-size:.78rem;line-height:1.5;margin-bottom:8px;">{m['summary'][:100]}...</div>
                      <div style="color:{cc};font-size:.75rem;font-weight:700;font-family:'Rajdhani',sans-serif;margin-bottom:6px;">Match: {sc:.3f}</div>
                      <div style="border-top:1px solid #3D2E1E;padding-top:6px;color:#9B8878;font-size:.7rem;line-height:1.6;">
                        📜 {cert}<br>👤 {m['client_type']}<br>💰 {m['contract_value']}
                      </div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📥 Export Full Analysis")
        report = {
            "rfp_overview":    {k:v for k,v in rfp.items() if k!="raw_text"},
            "compliance":      comp,
            "win_probability": win,
            "effort_reduction": effort,
        }
        st.download_button("⬇️ Download JSON Analysis",
            data=json.dumps(report,indent=2),
            file_name=f"bid_analysis_{rfp.get('source_file','report')}.json",
            mime="application/json",
            use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Q&A SECTIONS  (Requirement #5)
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("## ❓ Q&A / Clarification Sections")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

        qa_sections = rfp.get("qa_sections", [])

        if not qa_sections:
            st.markdown("""
            <div style="background:rgba(255,140,0,.06);border:1px solid rgba(255,140,0,.2);
                border-radius:8px;padding:18px 22px;color:#9B8878;font-size:.9rem;">
                ℹ️ No dedicated Q&amp;A or Clarification section was detected in this RFP.<br>
                <span style="font-size:.82rem;">Key questions to address are listed in the
                <b style="color:#FFA500;">RFP Overview → Key Sections</b> tab.</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,69,0,.08),rgba(255,140,0,.04));
                border:1px solid rgba(255,100,0,.25);border-radius:8px;padding:14px 18px;
                color:#E8D5C0;font-size:.9rem;margin-bottom:1rem;">
                📋 &nbsp;<b style="color:#FFD700;">{len(qa_sections)} Q&amp;A block(s)</b>
                extracted from the document's clarification or questionnaire sections.
            </div>""", unsafe_allow_html=True)

            for i, qa in enumerate(qa_sections, 1):
                q = qa.get("question","")
                c = qa.get("context","")
                st.markdown(f"""
                <div class="lava-card" style="margin:8px 0;">
                  <div style="display:flex;align-items:flex-start;gap:12px;">
                    <div style="background:linear-gradient(135deg,rgba(255,69,0,.2),rgba(255,140,0,.1));
                        border-radius:6px;padding:6px 10px;font-family:'Rajdhani',sans-serif;
                        font-weight:700;font-size:1rem;color:#FFD700;flex-shrink:0;">Q{i}</div>
                    <div style="flex:1;">
                      <div style="color:#E8D5C0;font-weight:600;font-size:.9rem;margin-bottom:6px;">{q}</div>
                      {f'<div style="color:#9B8878;font-size:.8rem;font-style:italic;">{c}</div>' if c else ''}
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 All Key Questions (from RFP Overview)")
        kqs = rfp.get("key_questions", [])
        if kqs:
            for i,q in enumerate(kqs,1):
                st.markdown(f"""
                <div style="background:rgba(255,69,0,.04);border-left:2px solid #FF6B1A;
                    border-radius:0 6px 6px 0;padding:9px 14px;margin:5px 0;
                    color:#E8D5C0;font-size:.88rem;">
                  <span style="color:#FF8C00;font-weight:700;font-family:'Rajdhani',sans-serif;margin-right:8px;">Q{i}</span>{q}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No key questions extracted.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — DRAFT PROPOSAL
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("## ✍️ AI Proposal Draft Generator")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)

        questions = rfp.get("key_questions",[])
        if not questions:
            st.warning("No key questions extracted. Re-upload a more detailed RFP document.")
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,69,0,.08),rgba(255,140,0,.04));
                border:1px solid rgba(255,100,0,.25);border-radius:8px;padding:14px 18px;
                color:#E8D5C0;font-size:.9rem;margin-bottom:1rem;">
                🔥 &nbsp;Ready to draft <b style="color:#FFD700;">{len(questions)} proposal sections</b>
                — one per extracted key question, grounded in the Capability Library via RAG.
                Each section is AI-written and fully editable before export.
            </div>""", unsafe_allow_html=True)

            # ── Generate button: set a flag then rerun so Streamlit stays in the tab ──
            if st.button("🤖 Generate Full Proposal Draft", type="primary", key="gen_btn"):
                st.session_state.generating_proposal = True
                st.session_state.active_tab = 5   # Draft Proposal is index 5
                st.rerun()

            # ── Generation loop runs on the NEXT rerun (after flag is set) ──────────
            if st.session_state.generating_proposal and \
               not st.session_state.workspaces[active].get("draft_sections"):
                from utils.proposal_generator import generate_section
                all_sections = []
                prog = st.progress(0, text="Igniting draft generation…")
                for i, q in enumerate(questions):
                    prog.progress(
                        int(i / len(questions) * 100),
                        text=f"✍️ Drafting section {i+1}/{len(questions)}…"
                    )
                    all_sections.append(generate_section(q, rfp))
                prog.progress(100, text="✅ All sections drafted!")
                st.session_state.workspaces[active]["draft_sections"] = all_sections
                st.session_state.generating_proposal = False
                st.rerun()   # rerun once more to render the edit boxes cleanly

            draft_sections = st.session_state.workspaces[active].get("draft_sections")

            if draft_sections:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:linear-gradient(90deg,rgba(46,204,113,.1),transparent);
                    border-left:3px solid #2ECC71;border-radius:0 8px 8px 0;
                    padding:12px 18px;color:#2ECC71;font-weight:600;
                    margin-bottom:1rem;font-family:'Rajdhani',sans-serif;font-size:1.1rem;">
                    ✅ {len(draft_sections)} sections generated — review and edit below
                </div>""", unsafe_allow_html=True)

                for i,sec in enumerate(draft_sections,1):
                    with st.expander(f"📄 Section {i} — {sec['question'][:70]}{'...' if len(sec['question'])>70 else ''}", expanded=(i==1)):
                        if sec.get("evidence_used"):
                            st.markdown('<div style="color:#9B8878;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">Supporting Evidence (RAG)</div>', unsafe_allow_html=True)
                            ecols = st.columns(len(sec["evidence_used"]))
                            for ecol,m in zip(ecols,sec["evidence_used"]):
                                sc = m["score"]
                                cc = "#2ECC71" if sc>=.25 else "#FF8C00" if sc>=.15 else "#FF4500"
                                ecol.markdown(f"""
                                <div class="cap-chip">
                                  <div style="color:{cc};font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.9rem;">[{m['cap_id']}]</div>
                                  <div style="color:#FFA500;font-size:.75rem;margin:2px 0;">{m['domain']}</div>
                                  <div style="color:#9B8878;font-size:.7rem;">match {sc:.2f}</div>
                                </div>""", unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                        edited = st.text_area("Edit this section:",
                            value=sec["drafted_text"], height=230,
                            key=f"edit_{active}_{i}")
                        st.session_state.workspaces[active]["draft_sections"][i-1]["drafted_text"] = edited

                st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📥 Export Final Proposal")
                safe = rfp.get("source_file","proposal").replace(".pdf","").replace(".docx","")
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    from utils.docx_exporter import export_proposal_to_docx
                    docx_b = export_proposal_to_docx(rfp, draft_sections, comp)
                    st.download_button("⬇️ Download Word Document (.docx)",
                        data=docx_b,
                        file_name=f"proposal_{safe}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True)

                with col_d2:
                    json_out = {
                        "rfp":      {k:v for k,v in rfp.items() if k!="raw_text"},
                        "sections": [{"question":s["question"],"draft":s["drafted_text"]} for s in draft_sections]
                    }
                    st.download_button("⬇️ Download JSON Export",
                        data=json.dumps(json_out,indent=2),
                        file_name=f"proposal_{safe}.json",
                        mime="application/json",
                        use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — EVALUATION CRITERIA TAXONOMY
    # ══════════════════════════════════════════════════════════════════════════
    with tab7:
        st.markdown("## 📊 Evaluation Criteria Taxonomy")
        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#9B8878;font-size:.85rem;margin-bottom:1rem;">16 standard RFP evaluation criteria with sector mapping and typical weights. Green border = detected in this RFP.</div>', unsafe_allow_html=True)

        from utils.data_loader import load_eval_criteria
        taxonomy = load_eval_criteria()
        rfp_crit_names = [ec.get("criterion","").lower() for ec in rfp.get("evaluation_criteria",[])]

        rows = [taxonomy[i:i+2] for i in range(0,len(taxonomy),2)]
        for row in rows:
            cols = st.columns(2)
            for col,item in zip(cols,row):
                present = any(item["criterion"].lower()[:15] in rc for rc in rfp_crit_names) \
                       or any(rc[:15] in item["criterion"].lower() for rc in rfp_crit_names)
                bc   = "#2ECC71" if present else "#3D2E1E"
                bdg  = '<span style="color:#2ECC71;font-size:.7rem;">✅ In this RFP</span>' if present \
                       else '<span style="color:#5C4A3A;font-size:.7rem;">Not detected</span>'
                sc   = "#FF8C00" if item["sector"]=="All" else "#9B8878"
                col.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(26,18,8,.95),rgba(20,14,8,.9));
                    border:1px solid {bc};border-radius:10px;padding:14px 16px;margin:6px 0;
                    box-shadow:0 2px 12px rgba(0,0,0,.3);">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                    <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.95rem;color:#FFA500;flex:1;padding-right:8px;">{item['criterion']}</div>
                    <div style="text-align:right;flex-shrink:0;">
                      <div style="color:{sc};font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;">{item['sector']}</div>
                      <div style="color:#FFD700;font-family:'Rajdhani',sans-serif;font-weight:700;font-size:.9rem;">{item['typical_weight']}</div>
                    </div>
                  </div>
                  <div style="color:#9B8878;font-size:.78rem;line-height:1.5;margin-bottom:8px;">{item['description']}</div>
                  {bdg}
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="lava-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 Quick Reference Table")
        df_tax = pd.DataFrame(taxonomy)[["criterion","sector","typical_weight","description"]]
        df_tax.columns = ["Criterion","Sector","Typical Weight","Description"]
        st.dataframe(df_tax[["Criterion","Sector","Typical Weight"]], use_container_width=True, hide_index=True)

# ── EMPTY STATE ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;">
      <div style="font-size:4rem;margin-bottom:16px;">🌋</div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;color:#5C4A3A;margin-bottom:10px;">No RFP Loaded</div>
      <div style="color:#3D2E1E;font-size:.95rem;max-width:380px;margin:0 auto;">
        Upload a PDF or DOCX tender document in the sidebar, then click
        <span style="color:#FF6B1A;font-weight:600;">🔥 Ignite Analysis</span> to create a workspace and begin.
      </div>
    </div>""", unsafe_allow_html=True)
