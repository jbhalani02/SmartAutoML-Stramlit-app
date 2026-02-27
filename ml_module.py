import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
##from sklearn.preprocessing import OrdinalEncodern
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def run_structure_app(model_option):

    # SESSION STATE INIT
    if "ml_model" not in st.session_state:
        st.session_state.ml_model = None
    if "ml_problem_type" not in st.session_state:
        st.session_state.ml_problem_type = None
    if "ml_encoders" not in st.session_state:
        st.session_state.ml_encoders = {}
    if "ml_target_encoder" not in st.session_state:
        st.session_state.ml_target_encoder = None   # aInitialize
    if "ml_features" not in st.session_state:
        st.session_state.ml_features = []
    if "training_done" not in st.session_state:
        st.session_state.training_done = False
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False


    # TRAIN NEW MODEL
    if model_option == "Train new model":
        st.header("Train New Model")
        file = st.file_uploader("Upload CSV File", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            target = st.selectbox("Select Target Column", df.columns)
            X = df.drop(columns=[target])
            y = df[target]

            # Fill missing values
            for col in X.select_dtypes(include=['int64','float64']).columns:
                X[col] = X[col].fillna(X[col].median())
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].fillna(X[col].mode()[0])

            # Encode categorical
            encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype(str).str.strip()
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

            # Detect problem type
            if y.dtype == "object" or y.nunique() <= 10:
                problem_type = "classification"
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
            else:
                problem_type = "regression"
                target_encoder = None

            st.write(f"Detected Problem Type: **{problem_type.upper()}**")

            if st.button("Train Model"):

                stratify_y = y if problem_type == "classification" else None

                # Drop date columns automatically
                for col in X.columns:
                    if "date" in col.lower():
                        X = X.drop(columns=[col])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=stratify_y
                )

                if problem_type == "classification":

                    from sklearn.preprocessing import StandardScaler

                    # Scale ONLY for Logistic
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Improved models
                    lr = LogisticRegression(
                        max_iter=5000,
                        class_weight='balanced'
                    )

                    rf = RandomForestClassifier(
                        n_estimators=300,
                        min_samples_split=5,
                        random_state=42
                    )

                    # Train
                    lr.fit(X_train_scaled, y_train)
                    rf.fit(X_train, y_train)

                    # Predict
                    lr_pred = lr.predict(X_test_scaled)
                    rf_pred = rf.predict(X_test)

                    lr_acc = accuracy_score(y_test, lr_pred)
                    rf_acc = accuracy_score(y_test, rf_pred)

                    # Select best
                    if lr_acc > rf_acc:
                        model = lr
                        y_pred = lr_pred
                        best_accuracy = lr_acc
                    else:
                        model = rf
                        y_pred = rf_pred
                        best_accuracy = rf_acc

                    precision = precision_score(y_test, y_pred, average="weighted")
                    recall = recall_score(y_test, y_pred, average="weighted")

                    if hasattr(model, "feature_importances_"):
                        feature_importance = dict(
                            zip(X.columns, model.feature_importances_)
                        )
                    else:
                        feature_importance = None

                    st.session_state.ml_results = {
                        "X_test": X_test,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "lr_acc": lr_acc,
                        "rf_acc": rf_acc,
                        "best_model": type(model).__name__,
                        "best_score": best_accuracy,
                        "precision": precision,
                        "recall": recall,
                        "problem_type": "classification",
                        "feature_importance": feature_importance
                    }

                    st.markdown("### Model Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Model", type(model).__name__)
                    col2.metric("Best Score", f"{best_accuracy:.4f}")
                    col3.metric("Problem Type", problem_type.upper())

                else:  # Regression

                    lr = LinearRegression()

                    rf = RandomForestRegressor(
                        n_estimators=300,
                        min_samples_split=5,
                        random_state=42
                    )

                    lr.fit(X_train, y_train)
                    rf.fit(X_train, y_train)

                    lr_pred = lr.predict(X_test)
                    rf_pred = rf.predict(X_test)

                    lr_r2 = r2_score(y_test, lr_pred)
                    rf_r2 = r2_score(y_test, rf_pred)

                    if lr_r2 > rf_r2:
                        model = lr
                        y_pred = lr_pred
                        best_score = lr_r2
                    else:
                        model = rf
                        y_pred = rf_pred
                        best_score = rf_r2

                    if hasattr(model, "feature_importances_"):
                        feature_importance = dict(
                            zip(X.columns, model.feature_importances_)
                        )
                    else:
                        feature_importance = None

                    st.session_state.ml_results = {
                        "X_test": X_test,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "lr_r2": lr_r2,
                        "rf_r2": rf_r2,
                        "best_model": type(model).__name__,
                        "best_score": best_score,
                        "problem_type": "regression",
                        "feature_importance": feature_importance
                    }

                    st.markdown("### Model Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Model", type(model).__name__)
                    col2.metric("Best Score", f"{best_score:.4f}")
                    col3.metric("Problem Type", problem_type.upper())

                # Save to session
                st.session_state.ml_model = model
                st.session_state.ml_problem_type = problem_type
                st.session_state.ml_encoders = encoders
                st.session_state.ml_target_encoder = target_encoder
                st.session_state.ml_features = X.columns.tolist()
                st.session_state.training_done = True
                st.session_state.show_analysis = False
                st.session_state.trained_model = model
                st.session_state.model_trained = True

                joblib.dump({
                    'model': model,
                    'problem_type': problem_type,
                    'features': X.columns.tolist(),
                    'encoders': encoders,
                    'target_encoder': target_encoder,
                    'ml_results': st.session_state.ml_results
                }, "tabular_model.pkl")

                st.success("Model trained and saved successfully!")


                st.session_state.chat_history = []
                st.session_state.agent_input = ""
                
                st.session_state.current_model_type = 'tabuler'

    # USE SAVED MODEL

    elif model_option == "Use saved model":
        st.session_state.training_done = False

            
        st.header("Use Saved Model")
        st.session_state.show_analysis = False

        try:
            model_package = joblib.load("tabular_model.pkl")

            st.session_state.ml_model = model_package['model']
            st.session_state.ml_problem_type = model_package['problem_type']
            st.session_state.ml_features = model_package['features']
            st.session_state.ml_encoders = model_package['encoders']
            st.session_state.ml_target_encoder = model_package.get('target_encoder', None)
            st.session_state.ml_results = model_package.get("ml_results")

            st.success("Saved model loaded successfully!")

            st.session_state.current_model_type = "tabuler"
            st.session_state.model_trained = True

        except:
            st.error("No saved model found. Please train first.")
            return

def plot_conf_matrix(y_test, y_pred):
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

def plot_accuracy_chart(lr_acc, rf_acc):
    fig, ax = plt.subplots()
    models = ["Logistic Regression", "Random Forest"]
    scores = [lr_acc, rf_acc]

    bars = ax.bar(models, scores)
    ax.set_xticks([])
    ax.set_yticks([])

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    ax.set_title("Model Comparison")
    st.pyplot(fig)

def plot_r2_chart(lr_r2, rf_r2):
    fig, ax = plt.subplots()
    models = ["Linear Regression", "Random Forest"]
    scores = [lr_r2, rf_r2]

    bars = ax.bar(models, scores)
    ax.set_xticks([])
    ax.set_yticks([])

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    ax.set_title("R2 Comparison")
    st.pyplot(fig)


        



