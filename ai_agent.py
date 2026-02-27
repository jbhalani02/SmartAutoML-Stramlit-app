from rapidfuzz import process

def ai_agent(user_input, model_results):

    if model_results is None:
        return {"type": "text", "message": "No active model found."}

    user_input = user_input.lower()
    problem_type = model_results.get("problem_type")


    # GLOBAL INTELLIGENT QUESTIONS (Work for all models)


    # Why best model?
    if "why" in user_input and "best" in user_input:
        return {
            "type": "text",
            "message":
                f"The best model was selected because it achieved the highest performance score "
                f"({model_results.get('best_score', model_results.get('accuracy', 0)):.4f}) "
                f"compared to other models."
        }

    # Confusion matrix
    if "confusion" in user_input:
        return {"type": "confusion_matrix"}

    # Precision
    if "precision" in user_input:
        return {
            "type": "text",
            "message": f"Precision score is {model_results.get('precision', 'Not available')}"
        }

    # Recall
    if "recall" in user_input:
        return {
            "type": "text",
            "message": f"Recall score is {model_results.get('recall', 'Not available')}"
        }

    # Feature Importance
    if "feature" in user_input and "important" in user_input:
        if model_results.get("feature_importance"):
            return {"type": "feature_importance"}
        else:
            return {"type": "text", "message": "Feature importance not available for this model."}

    # TEXT CLASSIFICATION

    if problem_type == "text_classification":

        knowledge_base = {
            "best model":
                f"Best model is {model_results['best_model']} "
                f"with accuracy {model_results['accuracy']:.4f}",

            "accuracy":
                f"Model accuracy is {model_results['accuracy']:.4f}",

            "f1 score":
                f"Model F1 score is {model_results['f1']:.4f}",

            "model type":
                "This is a Text Classification model using TF-IDF and Logistic Regression.",

            "performance summary":
                f"Accuracy: {model_results['accuracy']:.4f}, "
                f"F1 Score: {model_results['f1']:.4f}, "
                f"Precision: {model_results.get('precision','-')}, "
                f"Recall: {model_results.get('recall','-')}"
        }

        if "chart" in user_input or "plot" in user_input:
            return {"type": "metrics_chart"}


    # TABULAR CLASSIFICATION

    elif problem_type == "classification":

        knowledge_base = {
            "best model":
                f"Best performing model is {model_results['best_model']} "
                f"with accuracy {model_results['best_score']:.4f}",

            "accuracy":
                f"Best model accuracy is {model_results['best_score']:.4f}",

            "logistic regression accuracy":
                f"Logistic Regression accuracy is {model_results.get('lr_acc',0):.4f}",

            "random forest accuracy":
                f"Random Forest accuracy is {model_results.get('rf_acc',0):.4f}",

            "both model accuracy":
                f"Logistic Regression accuracy is {model_results.get('lr_acc',0):.4f} "
                f"and Random Forest accuracy is {model_results.get('rf_acc',0):.4f}",

            "model type":
                f"This is a Tabular Classification problem. "
                f"Best model selected: {model_results['best_model']}."
        }

        if "chart" in user_input or "comparison" in user_input:
            return {"type": "comparison_chart"}

        # Priority keyword detection
        if "logistic" in user_input:
            return {"type": "text", "message": knowledge_base["logistic regression accuracy"]}

        if "random" in user_input or "forest" in user_input:
            return {"type": "text", "message": knowledge_base["random forest accuracy"]}

    # TABULAR REGRESSION

    elif problem_type == "regression":

        knowledge_base = {
            "best model":
                f"Best performing model is {model_results['best_model']} "
                f"with R2 score {model_results['best_score']:.4f}",

            "r2":
                f"Best model R2 score is {model_results['best_score']:.4f}",

            "linear regression r2":
                f"Linear Regression R2 score is {model_results.get('lr_r2',0):.4f}",

            "random forest r2":
                f"Random Forest Regressor R2 score is {model_results.get('rf_r2',0):.4f}",

            "model type":
                f"This is a Regression problem. "
                f"Best model selected: {model_results['best_model']}."
        }

        if "chart" in user_input or "comparison" in user_input:
            return {"type": "comparison_chart"}

        if "linear" in user_input:
            return {"type": "text", "message": knowledge_base["linear regression r2"]}

        if "random" in user_input or "forest" in user_input:
            return {"type": "text", "message": knowledge_base["random forest r2"]}

    else:
        return {"type": "text", "message": "Model type not recognized."}

    # COMMON BEST MODEL QUESTION

    if "best" in user_input:
        return {"type": "text", "message": knowledge_base["best model"]}

    # FUZZY MATCH FALLBACK

    best_match, score, _ = process.extractOne(
        user_input,
        knowledge_base.keys()
    )

    if score > 60:
        return {"type": "text", "message": knowledge_base[best_match]}
    else:
        return {"type": "text", "message": "Sorry, I don't understand your question."}
