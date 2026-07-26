"""
MovieIQ — Movie Success Predictor
Predicting whether a movie's revenue exceeds its budget using Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np

# Try importing sklearn with error handling
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    st.error("⚠️ scikit-learn is not available. Please check your installation.")

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try importing scipy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("# 🎬 MovieIQ")
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
    st.markdown("### 🔗 Connect")
    st.markdown("""
    - [GitHub](https://github.com)
    - [LinkedIn](https://linkedin.com)
    """)
    st.markdown("---")
    st.caption("© 2026 Sudev Patra")

@st.cache_data
def load_data():
    """Load the movie dataset"""
    try:
        df = pd.read_csv("movies_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'movies_cleaned.csv' not found!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_model():
    """Train the Random Forest model"""
    if not SKLEARN_AVAILABLE:
        return None, None
    
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
            n_estimators=50,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X, y)
        
        return model, encoder
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None

# Check if sklearn is available
if not SKLEARN_AVAILABLE:
    st.error("""
    ## ❌ scikit-learn is not installed
    
    Please check your installation. This app requires scikit-learn to run.
    
    **Steps to fix:**
    1. Click "Manage App" in the bottom right
    2. Click "Advanced Settings"
    3. Set Python version to 3.9
    4. Click "Rebuild"
    
    If the issue persists, please check the logs for more details.
    """)
    st.stop()

# Load data
df = load_data()
if df is None:
    st.error("Unable to load data. Please check your CSV file.")
    st.stop()

# Train model
model, encoder = train_model()

# Main title
st.title("🎬 MovieIQ — Movie Success Predictor")
st.caption("Predicting whether a movie's revenue exceeds its budget using Machine Learning")

# Show model status
if model is not None:
    st.success("✅ Model loaded successfully!")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 EDA", "📉 Statistical Tests", "🎯 Predict"])

# ---------------- Overview ----------------
with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{len(df):,}")
    with col2:
        st.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    with col4:
        st.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------- EDA ----------------
with tab2:
    st.header("Exploratory Data Analysis")
    
    if not MATPLOTLIB_AVAILABLE:
        st.warning("⚠️ Matplotlib not available. EDA plots are disabled.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎭 Genre Distribution")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                df["genre"].value_counts().plot(kind="bar", ax=ax, color="#3498db")
                ax.set_ylabel("Count")
                ax.set_xlabel("Genre")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")

        with col2:
            st.subheader("📊 Success Rate by Genre")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                rate = df.groupby("genre")["success"].mean().sort_values()
                rate.plot(kind="barh", ax=ax, color="#2ecc71")
                ax.set_xlabel("Success Rate")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("💰 Budget vs. Revenue")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = df["success"].map({1: "#2ecc71", 0: "#e74c3c"})
                ax.scatter(df["budget"], df["revenue"], c=colors, alpha=0.5, s=15)
                ax.set_xlabel("Budget ($)")
                ax.set_ylabel("Revenue ($)")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")

        with col4:
            st.subheader("📊 Feature Distributions")
            feature = st.selectbox("Choose a feature", ["budget", "popularity", "runtime", "vote_average"])
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                df[df["success"] == 1][feature].hist(ax=ax, alpha=0.6, label="Success", color="#2ecc71", bins=25)
                df[df["success"] == 0][feature].hist(ax=ax, alpha=0.6, label="Failure", color="#e74c3c", bins=25)
                ax.legend()
                ax.set_xlabel(feature.replace('_', ' ').title())
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")

# ---------------- Statistical Tests ----------------
with tab3:
    st.header("Statistical Testing")
    
    if not SCIPY_AVAILABLE:
        st.warning("⚠️ SciPy not available. Statistical tests are disabled.")
    else:
        st.subheader("T-tests: Numeric Features vs. Success")
        
        results = []
        numeric_cols = ["budget", "popularity", "runtime", "vote_average"]
        
        for col in numeric_cols:
            success = df[df["success"] == 1][col]
            failure = df[df["success"] == 0][col]
            
            try:
                t_stat, p_val = stats.ttest_ind(success, failure, equal_var=False)
                
                results.append({
                    "Feature": col.title(),
                    "Mean (Success)": f"${success.mean():,.0f}" if col == "budget" else f"{success.mean():.2f}",
                    "Mean (Failure)": f"${failure.mean():,.0f}" if col == "budget" else f"{failure.mean():.2f}",
                    "T-Statistic": f"{t_stat:.3f}",
                    "P-Value": f"{p_val:.4f}",
                    "Significant": "✅ Yes" if p_val < 0.05 else "❌ No"
                })
            except Exception as e:
                st.warning(f"Could not calculate t-test for {col}")
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("Chi-square: Genre vs. Success")
        try:
            contingency = pd.crosstab(df["genre"], df["success"])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi² Statistic", f"{chi2:.3f}")
            col2.metric("P-Value", f"{p_val:.4f}")
            col3.metric("Significant", "✅ Yes" if p_val < 0.05 else "❌ No")
            
            st.caption(
                "📝 Genre shows no statistically significant association with success in this dataset, "
                "and only popularity showed a significant mean difference between successful and "
                "unsuccessful movies — consistent with the modest predictive power of the model."
            )
        except Exception as e:
            st.error(f"Error performing chi-square test: {str(e)}")

# ---------------- Predict ----------------
with tab4:
    st.header("🎯 Predict Movie Success")
    
    st.caption("⚠️ **Note:** The Random Forest model has limited predictive power. Use predictions as guidance only.")
    
    if model is None or encoder is None:
        st.error("❌ Model is not available. Please reload the app.")
    else:
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input(
                "Budget ($)", 
                min_value=1000, 
                value=50_000_000, 
                step=1_000_000,
                format="%d"
            )
            
            popularity = st.slider(
                "Popularity", 
                min_value=0.0, 
                max_value=150.0, 
                value=50.0,
                step=0.1
            )
            
            runtime = st.slider(
                "Runtime (minutes)", 
                min_value=60, 
                max_value=240, 
                value=120,
                step=5
            )
        
        with col2:
            vote_average = st.slider(
                "Vote Average", 
                min_value=0.0, 
                max_value=10.0, 
                value=6.0,
                step=0.1
            )
            
            genre = st.selectbox(
                "Genre", 
                sorted(df["genre"].unique())
            )

        # Predict button
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            try:
                # Encode genre
                genre_encoded = encoder.transform([genre])[0]
                
                # Create feature vector
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
                
                # Display result
                if pred == 1:
                    st.success(f"✅ **Predicted: SUCCESS**")
                    st.info(f"📊 Probability of success: **{proba:.1%}**")
                else:
                    st.error(f"❌ **Predicted: FAILURE**")
                    st.info(f"📊 Probability of success: **{proba:.1%}**")
                
                # Show feature importance
                st.subheader("📊 Model Insights")
                
                # Simple explanation
                budget_ratio = budget / df['budget'].max()
                pop_ratio = popularity / df['popularity'].max()
                vote_ratio = vote_average / 10.0
                
                insights = []
                if budget_ratio > 0.5:
                    insights.append("💰 High budget movies tend to have higher success rates")
                if pop_ratio > 0.5:
                    insights.append("⭐ Popular movies tend to perform better")
                if vote_ratio > 0.7:
                    insights.append("🌟 High-rated movies have better success odds")
                
                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.info("📝 Based on the inputs, this movie has mixed success indicators")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.divider()
st.caption("MovieIQ | Built with scikit-learn, scipy, and Streamlit | © 2026 Sudev Patra")
