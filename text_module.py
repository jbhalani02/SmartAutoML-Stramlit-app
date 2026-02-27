import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def run_text_module(model_option):

    # SESSION STATE INIT
    if "text_model" not in st.session_state:
        st.session_state.text_model = None
    if "text_results" not in st.session_state:
        st.session_state.text_results = None
    if "module_result" not in st.session_state:
        st.session_state.module_result = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False
    if "training_completed" not in st.session_state:
        st.session_state.training_completed = False
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""


    # TEXT CLEANING FUNCTION
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    def clean_text(text):
        text = str(text).lower()

        # Remove HTML
        text = re.sub(r"<.*?>", " ", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", " ", text)

        # Remove emails
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove numbers
        text = re.sub(r"\d+", " ", text)

        # Remove special characters
        text = re.sub(r"[^a-z\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]

        return " ".join(words)

    # TRAIN NEW MODEL
    if model_option == "Train new model":
        st.header("Train New Model")
        
        upload_file = st.file_uploader("Upload CSV file", type=["csv"])

        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.dataframe(df.head())

            text_col = st.selectbox("Select Text Column", df.columns)
            target_col = st.selectbox("Select Target Column", df.columns)

            if st.button("Train Model"):
                st.session_state.training_completed = False
                with st.spinner("Training optimized model..."):
                    # Clean text
                    df[text_col] = df[text_col].astype(str).apply(clean_text)

                    # Remove nulls
                    df = df[[text_col, target_col]].dropna()

                    X = df[text_col]
                    y = df[target_col]

                    # If only 1 class → show error
                    if len(np.unique(y)) < 2:
                        st.error("Dataset must contain at least 2 classes.")
                        st.stop()

                    # train test split
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )
                    except:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=0.2,
                            random_state=42
                        )
                    tfidf = TfidfVectorizer(
                        stop_words="english",
                        max_features=20000,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.9,
                        sublinear_tf=True
                    )

                    pipeline = Pipeline([
                        ("tfidf", tfidf),
                        ("clf", LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced"
                        ))
                    ])

                    param_grid = {"clf__C": [0.1, 1, 5, 10]}

                    grid = GridSearchCV(
                        pipeline,
                        param_grid,
                        cv=5,
                        scoring="f1_weighted",
                        n_jobs=-1
                    )

                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_

                    # Predictions
                    y_pred = model.predict(X_test)

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")

                    st.session_state.model_trained = True
                    

                    # Cross Validation
                    cv_scores = cross_val_score(
                        model, X, y, cv=5, scoring="f1_weighted"
                    )

                    precision = precision_score(y_test, y_pred, average="weighted")
                    recall = recall_score(y_test, y_pred, average="weighted")

                    # Save to session
                    st.session_state.text_model = model
                    
                    st.session_state.text_results = {
                            "y_test": y_test,
                            "y_pred": y_pred,
                            "accuracy": acc,
                            "f1": f1,
                            "precision": precision,
                            "recall": recall,
                            "best_model": "Logistic Regression (TF-IDF)",
                            "best_score": acc,
                            "problem_type": "text_classification"
                        }
                    
                    st.session_state.current_model_type = 'text'
                    st.session_state.module_result = {"metrics": {"cv_scores": cv_scores}}
                    st.session_state.model_trained = True
                    st.session_state.show_analysis = False  # reset analysis
                    st.session_state.user_input = ""

                    # Save model
                    joblib.dump({
                        "model": model,
                        "results": st.session_state.text_results,
                        "module_result": st.session_state.module_result
                    }, "text_model.pkl")

                    st.success("Optimized model trained and saved successfully!")
                    
                    st.markdown('### Model Summary')
                    col1, col2, col3 = st.columns(3, gap= 'large')
                    col1.metric('Best Model', 'Logistic Regression (TF-IDF)')
                    col2.metric('Accuracy', f'{acc:.4f}')
                    col3.metric('Problem Type', 'Text Classification')
                    
                    st.session_state.training_completed = True

                    st.session_state.chat_history = []
                    st.session_state.agent_input = ""

    # USE SAVED MODEL
    
    elif model_option == "Use saved model":
        st.header("Use Saved Model")

        try:
            saved = joblib.load("text_model.pkl")

            st.session_state.text_model = saved['model']
            st.session_state.text_results = saved['results']
            st.session_state.module_result = saved['module_result']

            st.session_state.current_model_type = "text"
            st.session_state.model_trained = True
            st.session_state.training_completed = False

            st.success("Saved Text model loaded successfully!")

        except:
            st.error("No saved Text model found. Please train first.")


def plot_text_conf_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, cmap="Blues")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(cax)
    st.pyplot(fig)


def plot_text_metrics_chart(accuracy, f1):
    fig, ax = plt.subplots()
    metrics = ["Accuracy", "F1 Score"]
    values = [accuracy, f1]
    bars = ax.bar(metrics, values)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Metrics")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}", ha='center', va='bottom')
    st.pyplot(fig)


