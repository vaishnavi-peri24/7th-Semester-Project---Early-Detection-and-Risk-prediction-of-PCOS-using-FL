# explainability.py

import lime
import lime.lime_tabular

def generate_lime_explanation(model, X_train_scaled, feature_names, input_instance):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=["No PCOS", "PCOS"],
        mode="classification"
    )

    explanation = explainer.explain_instance(
        input_instance,
        model.predict_proba,
        num_features=6
    )

    return explanation
