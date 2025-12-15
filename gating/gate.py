# gating/gate.py
import joblib
from experts.healthcare import HealthcareExpert
from experts.finance import FinanceExpert
from experts.cybersecurity import CybersecurityExpert

# Load ML gating model
clf = joblib.load("../configs/gating_model.joblib")
vectorizer = joblib.load("../configs/vectorizer.joblib")

# Instantiate experts
EXPERTS = {
    "healthcare": HealthcareExpert(),
    "finance": FinanceExpert(),
    "cybersecurity": CybersecurityExpert()
}

def get_expert(user_input):
    input_vec = vectorizer.transform([user_input])
    expert_name = clf.predict(input_vec)[0]
    return EXPERTS[expert_name]
