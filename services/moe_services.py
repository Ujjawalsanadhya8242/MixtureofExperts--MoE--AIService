# services/moe_service.py
from fastapi import FastAPI
from gating.gate import get_expert

app = FastAPI(title="MoE AI Service")

@app.post("/predict/")
def predict(query: str):
    expert = get_expert(query)
    response = expert.predict(query)
    return {"expert": expert.__class__.__name__, "response": response}
