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
        num_features=10
    )

    # Modify HTML to increase figure size and reduce whitespace
    html = explanation.as_html()
    html = html.replace('width: 500px', 'width: 900px')
    html = html.replace('height: 400px', 'height: 450px')
    # Remove extra margins and padding
    html = html.replace('margin: 20px', 'margin: 0px')
    html = html.replace('padding: 20px', 'padding: 0px')
    
    class HTMLExplanation:
        def __init__(self, html_str):
            self.html_str = html_str
        def as_html(self):
            return self.html_str
    
    return HTMLExplanation(html)
