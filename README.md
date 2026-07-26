# MovieIQ

A Streamlit dashboard for exploring movie data and predicting movie
success (revenue > budget) using a Random Forest classifier.

## Structure

```
your-repo/
├── app-1.py               # Main Streamlit app
├── requirements.txt        # Python dependencies
├── packages.txt             # System-level dependencies (Streamlit Cloud)
├── movies_cleaned.csv       # Dataset (add your own file here)
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app-1.py
```

## Deploying

Deploy directly to Streamlit Community Cloud by pointing it at this
repo and `app-1.py` as the entry point. `requirements.txt` and
`packages.txt` will be picked up automatically.

## Contents

- **Overview** — dataset summary metrics
- **EDA** — genre distribution, success rate by genre, budget vs.
  revenue, feature distributions
- **Statistical Tests** — t-tests and chi-square test for feature
  significance
- **Predict** — live Random Forest prediction tool
