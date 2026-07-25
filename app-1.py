"""
MovieIQ — Stage 6: Streamlit Dashboard
Interactive app covering EDA, statistical test results, and a live
Random Forest prediction tool for movie success (revenue > budget).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import sys
import pickle

# Try importing joblib with fallback
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    st.warning("⚠️ Joblib not available. Using pickle for model serialization.")

st.set_page_config(page_title="MovieIQ", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/movie.png", width=80)
    st.title("🎬 MovieIQ")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Sudev Patra**")
    st.markdown("---")
    st.markdown("### 📊 App Info")
    st.markdown("""
    - **Version:** 1.0.0
    - **Type:** Movie Success Predictor
    - **Model:** Random Forest
    - **Data:** 2000+ Movies
    """)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("""
    - [GitHub](https://github.com)
    - [LinkedIn](https://linkedin.com)
    - [Portfolio](https://portfolio.com)
    """)
    st.markdown("---")
    st.caption("© 2026 Sudev Patra")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movies_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'movies_cleaned.csv' not found!")
        return None

def save_model_joblib(model, encoder, model_path, encoder_path):
    """Save models using joblib if available, otherwise use pickle"""
    try:
        if HAS_JOBLIB:
            joblib.dump(model, model_path)
            joblib.dump(encoder, encoder_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)
        return True
    except Exception as e:
        st.warning(f"Could not save models: {str(e)}")
        return False

def load_model_joblib(model_path, encoder_path):
    """Load models using joblib if available, otherwise use pickle"""
    try:
        if HAS_JOBLIB:
            model = joblib.load(model_path)
            encoder = joblib.load(encoder_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        return None, None

@st.cache_resource
def load_or_train_model():
    # Check if model files exist
    model_path = "movieiq_rf_model.pkl"
    encoder_path = "genre_encoder.pkl"
    
    # Try to load existing models
    model, encoder = load_model_joblib(model_path, encoder_path)
    
    if model is not None and encoder is not None:
        return model, encoder
    
    # If models don't exist, train from scratch
    df = load_data()
    if df is None:
        return None, None
    
    try:
        # Prepare features
        X = df[['budget', 'popularity', 'runtime', 'vote_average', 'genre']].copy()
        y = df['success']
        
        # Encode genre
        encoder = LabelEncoder()
        X['genre_encoded'] = encoder.fit_transform(X['genre'])
        X = X.drop('genre', axis=1)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X, y)
        
        # Save models
        save_model_joblib(model, encoder, model_path, encoder_path)
        
        return model, encoder
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None

# Load data and models
df = load_data()
if df is not None:
    model, encoder = load_or_train_model()
else:
    model = None
    encoder = None
    st.stop()

st.title("🎬 MovieIQ — Movie Success Predictor")
st.caption("Classification project: predicting whether a movie's revenue exceeds its budget.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "EDA", "Statistical Tests", "Predict"])

# ---------------- Overview ----------------
with tab1:
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies", len(df))
    col2.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
    col3.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    col4.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------- EDA ----------------
with tab2:
    st.header("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Genre distribution")
        fig, ax = plt.subplots()
        df["genre"].value_counts().plot(kind="bar", ax=ax, color="#3498db")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with c2:
        st.subheader("Success rate by genre")
        fig, ax = plt.subplots()
        rate = df.groupby("genre")["success"].mean().sort_values()
        rate.plot(kind="barh", ax=ax, color="#2ecc71")
        ax.set_xlabel("Success rate")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Budget vs. Revenue")
        fig, ax = plt.subplots()
        colors = df["success"].map({1: "#2ecc71", 0: "#e74c3c"})
        ax.scatter(df["budget"], df["revenue"], c=colors, alpha=0.5, s=15)
        ax.set_xlabel("Budget")
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

    with c4:
        st.subheader("Feature distributions")
        feature = st.selectbox("Choose a feature", ["budget", "popularity", "runtime", "vote_average"])
        fig, ax = plt.subplots()
        df[df["success"] == 1][feature].hist(ax=ax, alpha=0.6, label="Success", color="#2ecc71", bins=25)
        df[df["success"] == 0][feature].hist(ax=ax, alpha=0.6, label="Failure", color="#e74c3c", bins=25)
        ax.legend()
        st.pyplot(fig)

# ---------------- Statistical Tests ----------------
with tab3:
    st.header("Statistical Testing")

    st.subheader("T-tests: numeric features vs. success")
    rows = []
    for col in ["budget", "popularity", "runtime", "vote_average"]:
        s = df[df["success"] == 1][col]
        f = df[df["success"] == 0][col]
        t_stat, p_val = stats.ttest_ind(s, f, equal_var=False)
        rows.append({
            "feature": col, 
            "mean_success": s.mean(), 
            "mean_failure": f.mean(),
            "t_statistic": t_stat, 
            "p_value": p_val, 
            "significant (α=0.05)": p_val < 0.05
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Chi-square: genre vs. success")
    contingency = pd.crosstab(df["genre"], df["success"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Chi² statistic", f"{chi2:.3f}")
    cc2.metric("p-value", f"{p_val:.4f}")
    cc3.metric("Significant?", "Yes" if p_val < 0.05 else "No")
    st.caption(
        "Genre shows no statistically significant association with success in this dataset, "
        "and only popularity showed a significant mean difference between successful and "
        "unsuccessful movies — consistent with the modest predictive power of the model below."
    )

# ---------------- Predict ----------------
with tab4:
    st.header("Predict Movie Success")
    
    # Add model performance note
    st.caption("⚠️ **Model note:** The Random Forest model shows limited predictive power. "
               "Treat predictions as illustrative of the pipeline, not reliable forecasts.")
    
    if model is not None and encoder is not None:
        # Show model accuracy if available
        try:
            X_test = df[['budget', 'popularity', 'runtime', 'vote_average', 'genre']].copy()
            X_test['genre_encoded'] = encoder.transform(X_test['genre'])
            X_test = X_test.drop('genre', axis=1)
            y_test = df['success']
            accuracy = model.score(X_test, y_test)
            st.info(f"📊 Model accuracy on training data: {accuracy:.1%}")
        except:
            pass

        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input("Budget ($)", min_value=1000, value=50_000_000, step=1_000_000)
            popularity = st.slider("Popularity", 0.0, 150.0, 50.0)
            runtime = st.slider("Runtime (minutes)", 60, 240, 120)
        with c2:
            vote_average = st.slider("Vote Average", 0.0, 10.0, 6.0)
            genre = st.selectbox("Genre", sorted(df["genre"].unique()))

        if st.button("Predict", type="primary"):
            try:
                # Encode genre
                genre_encoded = encoder.transform([genre])[0]
                
                # Create feature vector with correct order
                X_new = pd.DataFrame([{
                    "budget": budget, 
                    "popularity": popularity, 
                    "runtime": runtime,
                    "vote_average": vote_average, 
                    "genre_encoded": genre_encoded
                }])
                
                # Make prediction
                pred = model.predict(X_new)[0]
                proba = model.predict_proba(X_new)[0][1]

                if pred == 1:
                    st.success(f"✅ Predicted: **SUCCESS** (probability: {proba:.1%})")
                else:
                    st.error(f"❌ Predicted: **FAILURE** (probability of success: {proba:.1%})")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all input values are valid.")
    else:
        st.error("❌ Model not available. Please check your data file and try again.")

st.divider()
st.caption("MovieIQ | Built with scikit-learn, scipy, and Streamlit")
