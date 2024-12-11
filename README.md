# **Fine-Tuned Language Model with LoRA**

Welcome to the repository for our fine-tuned language model using LoRA (Low-Rank Adaptation)! This project demonstrates how we can leverage parameter-efficient fine-tuning to adapt a powerful base language model for generating domain-specific questions. Instead of having the model answer the questions, this setup allows the user to respond with their own answers. 

**Key Links:**
- **Interactive Demo (Google Colab):** [Run the Inference Notebook](https://colab.research.google.com/github/Grandediw/Fine-Tuned-LLM/blob/main/Gradio_inference.ipynb)

---

## **Overview**

This project fine-tunes the `unsloth/llama-3.2-3b-instruct-bnb-4bit` base model using LoRA to create a parameter-efficient, specialized language model. The focus of this fine-tuning is to have the model generate coherent, contextually relevant questions about a specific topic. The end-user can then provide their own answers to these questions, facilitating a more interactive and exploratory learning experience.

### **Key Features**
- **Low-Rank Adaptation (LoRA)**: A parameter-efficient fine-tuning technique that modifies specific parts of the model.
- **Topic-Specific Question Generation**: The model is trained to produce well-formed questions that users can answer themselves.
- **Accessible Demo**: A Gradio-based web interface you can access to try out question generation in your browser.

---

## **How to Use**

### **1. Run the Inference Notebook (Google Colab)**

To interact with the model, follow these steps:

1. **Open the Notebook:**
   Click the link below to open the `Gradio_inference.ipynb` notebook in Google Colab:
   👉 **[Run the Inference Notebook](https://colab.research.google.com/github/<your-username>/Fine-Tuned-LLM/blob/main/Gradio_inference.ipynb)**

2. **Set Up the Environment:**
   - Ensure you are signed into your Google account.
   - Click on `Runtime` > `Run all` to execute all cells in the notebook.
   - Follow any prompts to authorize access if necessary.

3. **Launch the Interface:**
   - After running the notebook, a Gradio interface will appear.
   - Use the interface to generate questions on your chosen topic and provide your own answers.


### **2. Integrate into Your Code**

If you want to integrate the model into your own application to generate questions programmatically:

```python
!pip install unsloth
!pip install transformers
!pip install torch

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Choose any! We auto support ROPE scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, bFloat16 for Ampere+

model_name_or_path = "jacopoda/lora_model"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
    # token = "hf_...", #se il nostro modello non è public
    # Use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Example prompt
prompt = "Generate a question about historical architecture for the user to answer."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## **How It Works**

1. **Base Model**: 
   We start with `unsloth/Llama-3.2-1B-Instruct`, a capable language model optimized for instruction-like tasks.

2. **LoRA Fine-Tuning**: 
   We apply LoRA adapters to select projection layers (`down_proj`, `gate_proj`, `k_proj`, `o_proj`, `q_proj`, `v_proj`, `up_proj`) so that the model can be efficiently specialized to generate questions about a given topic, rather than general text generation or answer completion.

3. **Deployment**:
   The model is served via a Gradio interface, accessible at the provided link. Users can submit prompts asking the model to generate questions, then answer those questions themselves, turning the model into a tool for guided exploration of a subject.

---

## **Model Details**

### **Base Model**
- **Name**: `unsloth/Llama-3.2-1B-Instruct`
- **Task Type**: Causal Language Modeling (CAUSAL_LM)
- **Quantization**: 4-bit for efficient inference.

### **LoRA Configuration**
- **r**: 16
- **lora_alpha**: 16
- **lora_dropout**: 0
- **Target Modules**: `down_proj`, `gate_proj`, `k_proj`, `o_proj`, `up_proj`, `v_proj`, `q_proj`

---

## **Requirements**

### **Dependencies**
```bash
pip install torch transformers peft unsloth gradio
```

### **Environment**
- **Hardware**: NVIDIA GPU recommended for faster inference.
- **Software**: Python 3.8+ and PyTorch 1.12+.

---

## **Try It Yourself**
- Prompt the model to generate a question about a topic of your choice.
- Provide your own answer to explore the subject deeply and interactively.

👉 **[Test the Model](https://9b7c23980211fb75b3.gradio.live/)**

---

## **Acknowledgments**
- **Hugging Face**: For providing the platform and libraries.
- **Unsloth Team**: For their efficient models and tools.
- **LoRA Researchers**: For developing parameter-efficient fine-tuning techniques.

---

**Happy experimenting with the question-generation model!**
