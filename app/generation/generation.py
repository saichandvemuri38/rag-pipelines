from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load an instruction-following model
model_name = "tiiuae/falcon-7b-instruct"  # smaller model: mpt-7b-instruct
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

def generate_answer(context: str, question: str, max_length: int = 300) -> str:
    """
    Generate a summary/answer using LLM based on context
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from output
    return answer.replace(prompt, "").strip()
