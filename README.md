# AI-Powered Healthcare System 🏥🤖  

This project is an **AI-powered medical assistant** that provides **informative responses to medical queries** using a **fine-tuned LLaMA 3-based model**. The system leverages **Hugging Face Transformers** and a **medical dataset** to generate reliable responses.  

---

## 🚀 Features  
✅ **AI-Powered Medical Query Assistant**  
✅ **Fine-Tuned on AI Medical Dataset**  
✅ **Efficient Query Processing & Model Inference**  
✅ **Cloud-Based Execution (Google Colab)**  
✅ **Formatted & Readable Medical Responses**  

---

## 📌 Table of Contents  
- [Introduction](#introduction)  
- [Project Architecture](#project-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Future Enhancements](#future-enhancements)  
- [References](#references)  

---

## 📖 Introduction  
The **AI-Powered Healthcare System** utilizes a **pretrained AI model** to answer **health-related questions** in a structured and informative way. The model is fine-tuned on the **AI Medical Dataset** and is based on **Meta-LLaMA 3 8B-Instruct**.  

> ⚠️ **Disclaimer:** This AI model is for **informational purposes only** and should **not be considered a substitute for professional medical advice**. Always consult a **qualified healthcare provider**.  

---

## 🏗 Project Architecture  

```mermaid
graph TD;
    A[User Input] -->|Medical Question| B[Query Processing]
    B -->|Tokenization| C[Model Inference]
    C -->|LLaMA 3 Model| D[Response Processing]
    D -->|Formatting & Clarity| E[Result Display]
1️⃣ User Input: The user enters a medical question.
2️⃣ Query Processing: The input is tokenized and sent to the AI model.
3️⃣ Model Inference: The LLaMA 3-based AI model generates a response.
4️⃣ Response Processing: The output is formatted for better readability.
5️⃣ Result Display: The AI response is displayed to the user.

⚙️ Installation
Ensure you have Python 3.8+ and the required dependencies installed.

bash
Copy
Edit
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install torch==2.2.1 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
pip install transformers bitsandbytes accelerate
🔥 Usage
Here's how to use the AI-powered healthcare model:

python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 
import torch

# Load the pre-trained model
model_name = "ruslanmv/ai-medical-model-32bit"
device_map = 'auto' 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def askme(question):
    prompt = f"<|start_header_id|>system<|end_header_id|> You are a Medical AI chatbot assistant. <|eot_id|><|start_header_id|>User: <|end_header_id|>This is the question: {question}<|eot_id|>"
    
    # Tokenizing the input
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generating the output
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    
    # Decoding the response
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Formatting the answer
    try:
        answer_parts = answer.split("\n", 1)
        if len(answer_parts) > 1:
            answers = answer_parts[1].strip()
        else:
            answers = ""
        print(f"Answer: {answers}")   
    except:
        print(answer)  

# Example Query
askme("What is the main cause of inflammatory CD4+ T cells?")
🚀 Future Enhancements
🔹 Improve Model Accuracy: Further fine-tune the model with real-world medical data.
🔹 Chatbot Integration: Develop an interactive AI-powered healthcare assistant.
🔹 Multi-Modal Capabilities: Integrate support for medical images & lab reports.
🔹 Regulatory Compliance: Ensure HIPAA & GDPR compliance for secure medical data handling.
🔹 Mobile & Web Deployment: Create mobile-friendly and cloud-based applications.

📚 References
1️⃣ Ruslanmv, "AI Medical Model 32-bit", Hugging Face.
2️⃣ Meta, "Meta-Llama-3-8B-Instruct", Hugging Face.
3️⃣ AI Medical Dataset, "Hugging Face Dataset".
4️⃣ Brown et al., "Language Models are Few-Shot Learners," NeurIPS, 2020.
5️⃣ Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.

💡 Disclaimer
This project is for educational and informational purposes only. It is not a replacement for professional medical advice. Always consult a qualified healthcare professional for medical concerns.

🤝 Contributing
Contributions are welcome! If you’d like to improve this project:

1️⃣ Fork the repository.
2️⃣ Create a feature branch (git checkout -b feature-branch).
3️⃣ Commit your changes (git commit -m "Added new feature").
4️⃣ Push to the branch (git push origin feature-branch).
5️⃣ Open a Pull Request and describe your changes.

📝 License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.