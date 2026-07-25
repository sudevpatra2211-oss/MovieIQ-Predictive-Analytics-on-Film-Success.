import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MovieIQ", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("# 🎬 MovieIQ")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Sudev Patra**")
    st.markdown("---")
    st.caption("© 2026 Sudev Patra")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movies_cleaned.csv")
        return df
    except:
        return None

@st.cache_resource
def train_model():
    df = load_data()
    if df is None:
        return None, None
    
    X = df[['budget', 'popularity', 'runtime', 'vote_average', 'genre']].copy()
    y = df['success']
    
    encoder = LabelEncoder()
    X['genre_encoded'] = encoder.fit_transform(X['genre'])
    X = X.drop('genre', axis=1)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, encoder

df = load_data()
if df is None:
    st.error("Data not found")
    st.stop()

model, encoder = train_model()

st.title("🎬 MovieIQ — Movie Success Predictor")

tab1, tab2 = st.tabs(["Overview", "Predict"])

with tab1:
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", len(df))
    col2.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
    col3.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    col4.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    st.dataframe(df.head(10))

with tab2:
    st.header("Predict Movie Success")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget ($)", value=50_000_000)
        popularity = st.slider("Popularity", 0.0, 150.0, 50.0)
        runtime = st.slider("Runtime (minutes)", 60, 240, 120)
    
    with col2:
        vote_average = st.slider("Vote Average", 0.0, 10.0, 6.0)
        genre = st.selectbox("Genre", sorted(df["genre"].unique()))
    
    if st.button("Predict"):
        genre_encoded = encoder.transform([genre])[0]
        X_new = pd.DataFrame([{
            "budget": budget,
            "popularity": popularity,
            "runtime": runtime,
            "vote_average": vote_average,
            "genre_encoded": genre_encoded
        }])
        
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]
        
        if pred == 1:
            st.success(f"✅ SUCCESS (probability: {proba:.1%})")
        else:
            st.error(f"❌ FAILURE (probability of success: {proba:.1%})")
