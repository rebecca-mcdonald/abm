# Propensity-like ABM MVP (CRM-ready, v2)

This package contains a Streamlit app for ICP/Intent/Engagement scoring with HubSpot/Salesforce CSV importers, audience exports, and demo data.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run propensity_mvp_app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repo and push these files.
2. Go to https://share.streamlit.io → **New app** → select repo/branch.
3. Set **Main file path** to `propensity_mvp_app.py` → **Deploy**.
4. (Optional) Add API keys under **Settings → Secrets**.

## Data Formats
- Accounts: account_id, account_name, domain, industry, employee_count, annual_revenue, country, state
- Contacts: contact_id, account_id, email, first_name, last_name, title, seniority
- Intent: account_id, topic, signal_date(YYYY-MM-DD), signal_weight(0–1)
- Engagement: account_id, last_visit_at(YYYY-MM-DD), pageviews_30d, form_fills_90d

Packaged on 2025-08-14.
