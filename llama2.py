from ctransformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model first
llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", hf=True, gpu_layers=0)
model = llm

# Load the tokenizer using the loaded model
tokenizer = AutoTokenizer.from_pretrained(model)

class ModelLlamaBase:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        return response

# Create an instance of the ModelLlamaBase class
model_llama_base = ModelLlamaBase(model, tokenizer)

# Set the model to evaluation mode
model.eval()

# Generate dummy input to obtain 'inputs' tensor
dummy_prompt = "This is a dummy prompt."
inputs = model_llama_base.tokenizer(dummy_prompt, return_tensors="pt", max_length=1024, truncation=True)

# Trace the model
model_neuron = torch.neuron.trace(model, inputs)
model_neuron.save("llama2-neuron-hf.pt")
