import streamlit as st
import pandas as pd
import numpy as np
from ml_module import run_structure_app
from ml_module import plot_conf_matrix
from ml_module import plot_accuracy_chart
from ml_module import plot_r2_chart
from text_module import run_text_module
from text_module import plot_text_conf_matrix
from text_module import plot_text_metrics_chart
from ai_agent import ai_agent


if "current_model_type" not in st.session_state:
    st.session_state.current_model_type = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title = "AI Multi-Analysis platform", layout = 'wide')

st.markdown("""
            <style>

            /* Background */
            .stApp {
                background-color: #F8FAFC;
            }

            /* Main title */
            .main-title {
                font-size: 38px;
                font-weight: 700;
                text-align: center;
                color: #1E3A8A;
                margin-bottom: 30;
            }

            /* Subtitle */
            .sub-title {
                text-align: center;
                color: #475569;
                font-size: 18px;
                margin-bottom: 30px;
            }

            /* Card style */
            .block-container {
                padding-top: 2rem;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #FFFFFF;
            }

            /* Buttons */
            .stButton>button {
                background-color: #2563EB;
                color: white;
                border-radius: 6px;
                font-weight: 600;
            }

            .stButton>button:hover {
                background-color: #1D4ED8;
                color: white;
            }

            /* Metric cards */
            div[data-testid='metric-container']{
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.05);
            }

            </style>
            """, unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Multi-Model Analysis Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classification • Regression • Sentiment • Spam Detection</div>', unsafe_allow_html=True)
st.markdown('----')

with st.sidebar:
    st.markdown('### Model Settings')

    st.markdown('---')

    module_option = st.radio("Select Module",['Tabular ML', 'Text classification'])

    model_option = st.selectbox('Select Task ',["Train new model", "Use Saved Model"])

    # Track sidebar changes
    if "last_sidebar_state" not in st.session_state:
        st.session_state.last_sidebar_state = (module_option, model_option)

    if st.session_state.last_sidebar_state != (module_option, model_option):

        # Clear chat + input when switching
        st.session_state.chat_history = []

        st.session_state.last_sidebar_state = (module_option, model_option)

    st.markdown("---")
    st.markdown("### Quick Guide")
    st.caption("1️⃣  Select Module")
    st.caption("2️⃣  Train or Load Model")
    st.caption("3️⃣  Use AI Assistant")

if module_option == 'Tabular ML':
    st.markdown('## Tabular Machine Learning')
    run_structure_app(model_option)
elif module_option == 'Text classification':
    st.markdown('## Text Classification')
    run_text_module(model_option)


st.markdown('---')
st.markdown("## AI Model Assistant")
assistant_container= st.container()

with assistant_container:
    
    st.markdown("Ask intelligent questions about your trained model.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.form("agent_form", clear_on_submit=True):

        user_query = st.text_input(
            "Ask something about the trained model",
            key="agent_input"
        )

        submit = st.form_submit_button("Send")

        if submit:

            if not st.session_state.get("model_trained", False):
                st.warning("Please train or load a model first.")
            else:

                results = None

                if module_option == "Text classification":
                    results = st.session_state.get("text_results")

                elif module_option == "Tabular ML":
                    results = st.session_state.get("ml_results")

                if results is None:
                    st.warning("No model results available.")
                else:
                    response = ai_agent(user_query, results)

                    st.session_state.chat_history.append(("You", user_query))

                    if response["type"] == "text":
                        st.session_state.chat_history.append(("Agent", response["message"]))

                    elif response["type"] == "confusion_matrix":
                        st.session_state.chat_history.append(("Agent", "Here is the confusion matrix:"))
                        st.session_state.chat_history.append(("PLOT_CONFUSION", ""))

                    elif response["type"] == "metrics_chart":
                        st.session_state.chat_history.append(("Agent", "Here are the performance metrics:"))
                        st.session_state.chat_history.append(("PLOT_METRICS", ""))

                    elif response["type"] == "comparison_chart":
                        st.session_state.chat_history.append(("Agent", "Here is the model comparison chart:"))
                        st.session_state.chat_history.append(("PLOT_COMPARISON", ""))

                    elif response["type"] == "feature_importance":
                        st.session_state.chat_history.append(("Agent", "Here are the top important features:"))
                        st.session_state.chat_history.append(("PLOT_FEATURE_IMPORTANCE", ""))

    for sender, message in st.session_state.chat_history:

        if sender == "You":
            st.markdown(f"""
            <div style="
                background-color:#E2E8F0;
                padding:10px;
                border-radius:10px;
                margin-bottom:6px;">
                <b>You:</b> {message}
            </div>
            """, unsafe_allow_html=True)

        elif sender == "Agent":
            st.markdown(f"""
            <div style="
                background-color:#FFFFFF;
                padding:10px;
                border-radius:10px;
                margin-bottom:6px;">
                <b>Agent:</b> {message}
            </div>
            """, unsafe_allow_html=True)
        
        elif sender == "PLOT_CONFUSION":
            if module_option == "Text classification":
                plot_text_conf_matrix(
                    st.session_state.text_results["y_test"],
                    st.session_state.text_results["y_pred"]
                )
            else:
                plot_conf_matrix(
                    st.session_state.ml_results["y_test"],
                    st.session_state.ml_results["y_pred"]
                )

        elif sender == "PLOT_METRICS":
            plot_text_metrics_chart(
                st.session_state.text_results["accuracy"],
                st.session_state.text_results["f1"]
            )

        elif sender == "PLOT_COMPARISON":
            if st.session_state.ml_problem_type == "classification":
                plot_accuracy_chart(
                    st.session_state.ml_results["lr_acc"],
                    st.session_state.ml_results["rf_acc"]
                )
            else:
                plot_r2_chart(
                    st.session_state.ml_results["lr_r2"],
                    st.session_state.ml_results["rf_r2"]
                )

        elif sender == "PLOT_FEATURE_IMPORTANCE":

            importance = st.session_state.ml_results.get("feature_importance")

            if importance:
                import matplotlib.pyplot as plt

                sorted_items = sorted(
                    importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]

                features, scores = zip(*sorted_items)

                fig, ax = plt.subplots()
                ax.barh(features[::-1], scores[::-1])
                ax.set_title("Top 10 Feature Importance")
                ax.set_xlabel('Importance Score')
                ax.set_ylabel('Features')
                ax.grid(True)
                st.pyplot(fig)        

st.markdown('-----')
st.caption('Built with using  •  Streamlit  •  Scikit-Learn  •  NLP')
