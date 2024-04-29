from ctransformers import AutoModelForCausalLM, AutoTokenizer
import torch
# import serve

# Load Llama model
from ctransformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/CodeLlama-7B-Python-GGUF", model_file="codellama-7b-python.Q3_K_L.gguf", model_type="llama", gpu_layers=0, hf=True)
# model = AutoModelForCausalLM.from_pretrained("TheBloke/CodeLlama-7B-Python-GGUF")
tokenizer = AutoTokenizer.from_pretrained(model)

class ModelLlamaBase:
    # def __init__(self, model):
    #     self.model = model
    #     self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/CodeLlama-7B-Python-GGUF")  # Corrected tokenizer name
   
    def generate_response(self, prompt) -> str:
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # Generate response using Llama model
        with torch.no_grad():
            response = self.model.generate(
                input_ids=inputs.input_ids,
                max_length=200,
                temperature=0.8,
                num_beams=5,
                repetition_penalty=2.0,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and return response
        response = tokenizer.decode(response[0], skip_special_tokens=True)
        return response



model.eval()

model_neuron = torch.neuron.trace(model, inputs)

model_neuron.save("llama2-neuron-hf.pt")
