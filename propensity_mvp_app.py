
# (Same Streamlit app code as previously provided — CRM-ready with HubSpot/Salesforce importers)
# Keeping the full content inline for a self-contained package.
import io, json, math, re
from datetime import datetime, timedelta
import numpy as np, pandas as pd, streamlit as st

REQUIREMENTS = """
# save as requirements.txt
streamlit>=1.36
pandas>=2.1
numpy>=1.26
requests>=2.32
"""

DATE_FMT = "%Y-%m-%d"

@st.cache_data
def _today():
    return datetime.utcnow().date()

def _parse_date(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

SENIORITY_PATTERNS = [
    (r"\b(ceo|coo|cfo|cmo|cto|cio|chief|founder|owner|president)\b", "C-Suite"),
    (r"\b(vp|vice president|svp|evp|partner|principal)\b", "VP"),
    (r"\b(head|director)\b", "Director"),
    (r"\b(manager|lead|supervisor)\b", "Manager"),
    (r"\b(senior|sr\.)\b", "Senior IC"),
]

def infer_seniority_from_title(title):
    if not isinstance(title, str):
        return ""
    t = title.lower()
    for pat, lab in SENIORITY_PATTERNS:
        if re.search(pat, t):
            return lab
    return "IC"

def map_hubspot_companies(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Company ID": "account_id",
        "Name": "account_name",
        "Domain": "domain",
        "Industry": "industry",
        "Number of Employees": "employee_count",
        "Annual revenue": "annual_revenue",
        "Country/Region": "country",
        "State/Region": "state",
    }
    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]
    if "employee_count" in out: out["employee_count"] = pd.to_numeric(out["employee_count"], errors="coerce")
    if "annual_revenue" in out: out["annual_revenue"] = pd.to_numeric(out["annual_revenue"], errors="coerce")
    return out

def map_hubspot_contacts(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Contact ID": "contact_id",
        "Company ID": "account_id",
        "Email": "email",
        "First Name": "first_name",
        "Last Name": "last_name",
        "Job Title": "title",
    }
    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]
    if "seniority" not in out.columns:
        out["seniority"] = [infer_seniority_from_title(t) for t in out.get("title", [])]
    return out

def map_salesforce_accounts(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Account ID": "account_id",
        "Account Name": "account_name",
        "Website": "domain",
        "Industry": "industry",
        "Employees": "employee_count",
        "Annual Revenue": "annual_revenue",
        "Billing Country": "country",
        "Billing State/Province": "state",
        "Billing State": "state",
    }
    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns and dst not in out:
            out[dst] = df[src]
    if "employee_count" in out: out["employee_count"] = pd.to_numeric(out["employee_count"], errors="coerce")
    if "annual_revenue" in out: out["annual_revenue"] = pd.to_numeric(out["annual_revenue"], errors="coerce")
    return out

def map_salesforce_contacts(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Contact ID": "contact_id",
        "Account ID": "account_id",
        "Email": "email",
        "First Name": "first_name",
        "Last Name": "last_name",
        "Title": "title",
    }
    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]
    out["seniority"] = [infer_seniority_from_title(t) for t in out.get("title", [])]
    return out

def icp_fit_score(row, icp):
    score = 0.0
    if icp["industries"]:
        score += 20.0 if str(row.get("industry", "")).lower() in icp["industries"] else 0.0
    emp = row.get("employee_count", np.nan)
    if not pd.isna(emp):
        for (lo, hi, w) in icp["employee_buckets"]:
            if (lo is None or emp >= lo) and (hi is None or emp <= hi):
                score += w
                break
    rev = row.get("annual_revenue", np.nan)
    if not pd.isna(rev):
        for (lo, hi, w) in icp["revenue_buckets"]:
            if (lo is None or rev >= lo) and (hi is None or rev <= hi):
                score += w
                break
    if icp["geo_states"]:
        score += 10.0 if str(row.get("state", "")).upper() in icp["geo_states"] else 0.0
    return min(score, 100.0)

def intent_score(intent_df, account_id, recency_half_life_days=21):
    if intent_df is None or intent_df.empty:
        return 0.0
    now = _today()
    df = intent_df[intent_df["account_id"] == account_id]
    if df.empty:
        return 0.0
    score = 0.0
    for _, r in df.iterrows():
        d = _parse_date(r.get("signal_date"))
        if d is None:
            continue
        age_days = max(0, (now - d).days)
        decay = 0.5 ** (age_days / recency_half_life_days)
        w = float(r.get("signal_weight", 0.5))
        score += 100.0 * w * decay
    return min(score, 100.0)

def engagement_score(eng_df, account_id):
    if eng_df is None or eng_df.empty:
        return 0.0
    df = eng_df[eng_df["account_id"] == account_id]
    if df.empty:
        return 0.0
    r = df.iloc[-1]
    pv = float(r.get("pageviews_30d", 0) or 0)
    forms = float(r.get("form_fills_90d", 0) or 0)
    pv_part = 60.0 * (1 - math.exp(-pv / 10.0))
    form_part = 40.0 * (1 - math.exp(-forms / 2.0))
    return min(pv_part + form_part, 100.0)

def account_total_score(icp_s, intent_s, eng_s, weights):
    w_icp, w_intent, w_eng = weights
    total = (w_icp * icp_s + w_intent * intent_s + w_eng * eng_s) / (w_icp + w_intent + w_eng)
    return round(total, 2)

def demo_accounts():
    return pd.DataFrame([
        {"account_id": "A-100", "account_name": "Acme Robotics", "domain": "acme.ai", "industry": "software", "employee_count": 220, "annual_revenue": 35_000_000, "country": "US", "state": "CA"},
        {"account_id": "A-101", "account_name": "Northwind Steel", "domain": "northwindsteel.com", "industry": "manufacturing", "employee_count": 1200, "annual_revenue": 420_000_000, "country": "US", "state": "TX"},
        {"account_id": "A-102", "account_name": "Globex Health", "domain": "globexhealth.com", "industry": "healthcare", "employee_count": 800, "annual_revenue": 150_000_000, "country": "US", "state": "CA"},
    ])

def demo_contacts():
    return pd.DataFrame([
        {"contact_id": "C-1", "account_id": "A-100", "email": "maria@acme.ai", "first_name": "Maria", "last_name": "Lopez", "title": "VP Marketing", "seniority": "VP"},
        {"contact_id": "C-2", "account_id": "A-100", "email": "cto@acme.ai", "first_name": "Dev", "last_name": "Singh", "title": "CTO", "seniority": "C-Suite"},
        {"contact_id": "C-3", "account_id": "A-101", "email": "itdirector@northwindsteel.com", "first_name": "Ben", "last_name": "Hsu", "title": "Director IT", "seniority": "Director"},
    ])

def demo_intent():
    base = _today()
    return pd.DataFrame([
        {"account_id": "A-100", "topic": "abm software", "signal_date": (base - timedelta(days=3)).strftime(DATE_FMT), "signal_weight": 0.8},
        {"account_id": "A-100", "topic": "contact-level attribution", "signal_date": (base - timedelta(days=9)).strftime(DATE_FMT), "signal_weight": 0.6},
        {"account_id": "A-101", "topic": "intent data", "signal_date": (base - timedelta(days=21)).strftime(DATE_FMT), "signal_weight": 0.4},
        {"account_id": "A-102", "topic": "omnichannel abm", "signal_date": (base - timedelta(days=1)).strftime(DATE_FMT), "signal_weight": 0.9},
    ])

def demo_engagement():
    base = _today()
    return pd.DataFrame([
        {"account_id": "A-100", "last_visit_at": (base - timedelta(days=2)).strftime(DATE_FMT), "pageviews_30d": 18, "form_fills_90d": 1},
        {"account_id": "A-101", "last_visit_at": (base - timedelta(days=25)).strftime(DATE_FMT), "pageviews_30d": 3, "form_fills_90d": 0},
        {"account_id": "A-102", "last_visit_at": (base - timedelta(days=0)).strftime(DATE_FMT), "pageviews_30d": 32, "form_fills_90d": 3},
    ])

st.set_page_config(page_title="ABM MVP — Propensity-like", layout="wide")
st.title("ABM MVP — Propensity-like")
st.caption("Upload your data (native or CRM export), tune scoring, export audiences.")

with st.expander("📦 Install requirements.txt"):
    st.code(REQUIREMENTS, language="bash")

st.sidebar.header("Import format")
import_format = st.sidebar.radio("Choose CSV export format:", ["Native (this app)", "HubSpot export", "Salesforce export"])

st.header("1) Upload Data")
col1, col2, col3, col4 = st.columns(4)

def normalize_accounts(df):
    if import_format == "Native (this app)": return df
    if import_format == "HubSpot export": return map_hubspot_companies(df)
    if import_format == "Salesforce export": return map_salesforce_accounts(df)

def normalize_contacts(df):
    if import_format == "Native (this app)": return df
    if import_format == "HubSpot export": return map_hubspot_contacts(df)
    if import_format == "Salesforce export": return map_salesforce_contacts(df)

with col1:
    acc_file = st.file_uploader("Accounts/Companies CSV", type=["csv"])
    if st.button("Use demo accounts"):
        accounts_df = demo_accounts()
    elif acc_file:
        raw = pd.read_csv(acc_file)
        accounts_df = normalize_accounts(raw)
    else:
        accounts_df = pd.DataFrame()

with col2:
    c_file = st.file_uploader("Contacts CSV (optional)", type=["csv"])
    if st.button("Use demo contacts"):
        contacts_df = demo_contacts()
    elif c_file:
        raw = pd.read_csv(c_file)
        contacts_df = normalize_contacts(raw)
    else:
        contacts_df = pd.DataFrame()

with col3:
    i_file = st.file_uploader("Intent CSV (optional)", type=["csv"])
    if st.button("Use demo intent"):
        intent_df = demo_intent()
    elif i_file:
        intent_df = pd.read_csv(i_file)
    else:
        intent_df = pd.DataFrame()

with col4:
    e_file = st.file_uploader("Engagement CSV (optional)", type=["csv"])
    if st.button("Use demo engagement"):
        engagement_df = demo_engagement()
    elif e_file:
        engagement_df = pd.read_csv(e_file)
    else:
        engagement_df = pd.DataFrame()

required_cols = ["account_id", "account_name", "domain", "industry", "employee_count", "annual_revenue", "country", "state"]
missing = [c for c in required_cols if c not in accounts_df.columns]
if accounts_df.empty or missing:
    st.info("Upload at least an Accounts CSV (or click demo). Missing columns: " + ", ".join(missing) if missing else "Upload at least an Accounts CSV.")
    st.stop()

st.dataframe(accounts_df.head())

st.header("2) Define Your ICP & Weights")
industries = st.multiselect("Target industries (lowercase)", options=sorted(accounts_df.get("industry", pd.Series([], dtype=str)).str.lower().dropna().unique().tolist()))
geo_states = st.multiselect("Target US states", options=sorted(accounts_df.get("state", pd.Series([], dtype=str)).dropna().unique().tolist()))

colw1, colw2, colw3 = st.columns(3)
with colw1: w_icp = st.slider("Weight: ICP", 0, 100, 40)
with colw2: w_intent = st.slider("Weight: Intent", 0, 100, 40)
with colw3: w_eng = st.slider("Weight: Engagement", 0, 100, 20)

st.subheader("Employee & Revenue buckets")
colb1, colb2 = st.columns(2)
with colb1:
    emp_buckets = st.text_area("Employee buckets [(lo,hi,weight) per line]", value="(0,50,10)\n(51,200,15)\n(201,1000,20)\n(1001,None,10)", height=140)
with colb2:
    rev_buckets = st.text_area("Revenue buckets in USD [(lo,hi,weight) per line]", value="(0,10000000,10)\n(10000001,50000000,15)\n(50000001,250000000,20)\n(250000001,None,10)", height=140)

def parse_bucket_lines(s):
    res = []
    for line in s.splitlines():
        line = line.strip()
        if not line: continue
        lo, hi, w = eval(line, {"__builtins__": {}}, {})
        res.append((lo, hi, float(w)))
    return res

icp_cfg = {
    "industries": [x.lower() for x in industries],
    "geo_states": [x.upper() for x in geo_states],
    "employee_buckets": parse_bucket_lines(emp_buckets),
    "revenue_buckets": parse_bucket_lines(rev_buckets),
}

st.header("3) Scoring & MQAs")
if intent_df.empty: intent_df = None
if engagement_df.empty: engagement_df = None

weights = (w_icp, w_intent, w_eng)
rows = []
for _, r in accounts_df.iterrows():
    a_id = r.get("account_id")
    icp_s = icp_fit_score(r, icp_cfg)
    intent_s = intent_score(intent_df, a_id) if intent_df is not None else 0.0
    eng_s = engagement_score(engagement_df, a_id) if engagement_df is not None else 0.0
    total = account_total_score(icp_s, intent_s, eng_s, weights)
    rows.append({"account_id": a_id,"account_name": r.get("account_name"),"ICP": round(icp_s,2),"Intent": round(intent_s,2),"Engagement": round(eng_s,2),"AccountScore": total})

import pandas as pd
score_df = pd.DataFrame(rows).sort_values("AccountScore", ascending=False).reset_index(drop=True)

colk1, colk2, colk3 = st.columns(3)
with colk1: mqa_threshold = st.slider("MQA threshold (AccountScore)", 0, 100, 65)
with colk2: top_n = st.number_input("Show top N accounts", min_value=5, max_value=1000, value=50, step=5)
with colk3: half_life = st.slider("Intent recency half-life (days)", 7, 60, 21)

if intent_df is not None:
    rows2 = []
    for _, r in accounts_df.iterrows():
        a_id = r.get("account_id")
        icp_s = icp_fit_score(r, icp_cfg)
        intent_s = intent_score(intent_df, a_id, recency_half_life_days=half_life)
        eng_s = engagement_score(engagement_df, a_id) if engagement_df is not None else 0.0
        total = account_total_score(icp_s, intent_s, eng_s, weights)
        rows2.append({"account_id": a_id,"account_name": r.get("account_name"),"ICP": round(icp_s,2),"Intent": round(intent_s,2),"Engagement": round(eng_s,2),"AccountScore": total})
    score_df = pd.DataFrame(rows2).sort_values("AccountScore", ascending=False).reset_index(drop=True)

score_df["MQA"] = (score_df["AccountScore"] >= mqa_threshold)
st.dataframe(score_df.head(int(top_n)))
st.metric("# MQAs", int(score_df["MQA"].sum()))

st.header("4) Build Audiences & Export")
selected_audience = st.multiselect("Audience logic (accounts must match ALL selected):", options=["MQA only", "ICP >= 60", "Intent >= 50", "Engagement >= 30"], default=["MQA only"])

aud = score_df.copy()
for rule in selected_audience:
    if rule == "MQA only": aud = aud[aud["MQA"]]
    elif rule == "ICP >= 60": aud = aud[aud["ICP"] >= 60]
    elif rule == "Intent >= 50": aud = aud[aud["Intent"] >= 50]
    elif rule == "Engagement >= 30": aud = aud[aud["Engagement"] >= 30]

st.subheader("Audience Accounts")
st.dataframe(aud)

if 'contacts_df' in locals() and not contacts_df.empty:
    aud_contacts = contacts_df.merge(aud[["account_id"]], on="account_id", how="inner")
    st.subheader("Audience Contacts")
    st.dataframe(aud_contacts)
else:
    aud_contacts = pd.DataFrame()

colx1, colx2 = st.columns(2)
with colx1: st.download_button("Export Audience Accounts (CSV)", aud.to_csv(index=False), file_name="audience_accounts.csv")
with colx2:
    if not aud_contacts.empty:
        st.download_button("Export Audience Contacts (CSV)", aud_contacts.to_csv(index=False), file_name="audience_contacts.csv")

st.header("5) Pipeline Projection (toy)")
base_cvr = st.slider("Assumed opp conversion from MQAs (%)", 0, 50, 12)
avg_deal = st.number_input("Avg deal size ($)", value=25000, step=1000)
expected_opps = int(score_df["MQA"].sum() * (base_cvr / 100))
expected_pipe = expected_opps * avg_deal
colp1, colp2 = st.columns(2)
with colp1: st.metric("Expected Opps", expected_opps)
with colp2: st.metric("Projected Pipeline ($)", f"{expected_pipe:,.0f}")

st.header("6) Connectors (stubs)")
with st.expander("HubSpot CRM (stub)"):
    st.text_input("Private App Token")
    st.caption("Replace with real API calls to create/update contacts and companies, and write MQA activities.")
with st.expander("Google Ads / Customer Match (stub)"):
    st.text_input("Google OAuth JSON path")
    st.caption("Replace with real uploads of hashed emails / domains to Customer Match.")
with st.expander("LinkedIn Matched Audiences (stub)"):
    st.text_input("LinkedIn OAuth token")
    st.caption("Replace with real audience creation using LinkedIn Marketing API.")

st.header("7) Tracking Snippet & Webhook (stub)")
js_snippet = """
<script>
  window.abm = window.abm || {};
  function abmTrack(event, props) {
    fetch('https://your-domain.example/track', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({event, props, ts: new Date().toISOString()})
    });
  }
  // Example: abmTrack('page_view', {url: location.href});
</script>
"""
st.code(js_snippet, language="html")

with st.expander("About & Editing Tips"):
    st.markdown("- Import HubSpot or Salesforce exports via the sidebar.\n- Titles auto-infer seniority when missing.\n- Replace stubs with real connectors; add enrichment or reverse-IP for discovery.\n")
