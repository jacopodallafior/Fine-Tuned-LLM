# **Fine-Tuned Language Model with LoRA**

Welcome to the repository for my fine-tuned language model using LoRA (Low-Rank Adaptation)! This project showcases how we can leverage parameter-efficient fine-tuning with LoRA to adapt a powerful base language model for specific use cases.

The fine-tuned model is deployed on Hugging Face Spaces, where you can interact with it directly. Try it out here: [**Launch the Model**](https://huggingface.co/spaces/Grandediw/Test).

---

## **Overview**
This project fine-tunes the `unsloth/llama-3.2-3b-instruct-bnb-4bit` base model using LoRA. The adapter layers trained during fine-tuning are lightweight and focus on modifying specific parts of the base model to achieve the desired performance without retraining the entire model.

### **Key Features**
- **Low-Rank Adaptation (LoRA)**: A parameter-efficient fine-tuning method that modifies select layers of the base model.
- **Task-Specific Fine-Tuning**: The model is optimized for generating coherent, context-aware responses to user queries.
- **Interactive Chat Application**: Deployed as a chat interface on Hugging Face Spaces for easy accessibility.

---

## **How to Use**
### **Try It Online**
You can interact with the model directly in your browser via the Hugging Face Space:
👉 **[Try the Model Here](https://huggingface.co/spaces/Grandediw/Test)**

### **Use the Model in Your Code**
You can integrate the model into your Python applications using the Hugging Face Transformers library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the base model and LoRA weights
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3.2-3b-instruct-bnb-4bit", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("Grandediw/lora_model")

# Generate responses
inputs = tokenizer("What is the oldest building in Stockholm?", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.5)

# Decode and print the result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## **How It Works**
1. **Base Model**: 
   The model starts with `unsloth/llama-3.2-3b-instruct-bnb-4bit`, a powerful and lightweight causal language model.
   
2. **LoRA Fine-Tuning**: 
   - LoRA modifies specific components of the base model (e.g., projection layers) as defined in the `adapter_config.json` file.
   - This method is efficient, allowing us to fine-tune a large model with minimal compute resources.

3. **Deployment**:
   - The model is deployed on Hugging Face Spaces using the `unsloth` library for fast and efficient inference.
   - A Gradio-based chat interface is provided for user interaction.

---

## **Model Details**
### **Base Model**
- **Name**: `unsloth/llama-3.2-3b-instruct-bnb-4bit`
- **Task Type**: Causal Language Modeling (CAUSAL_LM)
- **Quantization**: 4-bit quantization for faster inference.

### **LoRA Configuration**
- **Adapter Configuration**: 
  - `r`: 16
  - `lora_alpha`: 16
  - `lora_dropout`: 0
- **Target Modules**:
  - `down_proj`, `gate_proj`, `k_proj`, `o_proj`, `up_proj`, `v_proj`, `q_proj`

---

## **Requirements**
### **Dependencies**
To reproduce this project locally, install the following dependencies:
- `torch`
- `transformers`
- `peft`
- `unsloth`
- `gradio`

Install them via pip:
```bash
pip install torch transformers peft unsloth gradio
```

### **Environment**
- **Hardware**: NVIDIA GPU with CUDA support is recommended for inference.
- **Software**: Python 3.8+ and PyTorch 1.12+.

---

## **Try It Yourself**
Interact with the model and see how it responds to your queries. Examples of tasks it can perform:
- General Q&A
- Context-aware dialogue
- Instruction following

👉 **[Test the Model Here](https://huggingface.co/spaces/Grandediw/Test)**

---

## **Acknowledgments**
Special thanks to:
- **Hugging Face** for providing the platform for hosting the model and the fine-tuning tools.
- **Unsloth Team** for their efficient implementations and base models.
