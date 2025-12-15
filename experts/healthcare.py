# experts/healthcare.py
from transformers import AutoTokenizer, AutoModelForCausalLM

class HealthcareExpert:
    def __init__(self):
        model_name = "Pankti99/llama-2-7b-HealthCareMagic"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
