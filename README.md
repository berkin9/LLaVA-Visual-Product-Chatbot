# 🛍️ LLaVA Visual Product Chatbot

A multimodal AI assistant that analyzes product images and answers natural-language questions about them.

This project fine-tunes **LLaVA-1.5-7B** using **QLoRA** on a fashion product dataset and deploys the resulting model as an interactive **Gradio application on Hugging Face Spaces**.

The goal of the project is to demonstrate an end-to-end multimodal AI workflow: dataset preparation, visual instruction tuning, parameter-efficient fine-tuning, inference, model deployment, and interactive user experience.

---

## 🚀 Live Demo

Try the deployed application on Hugging Face Spaces:

**Hugging Face Space:**  
https://huggingface.co/spaces/berkin9/llava-visual-product-chatbot

**LoRA Adapter:**  
https://huggingface.co/berkin9/llava-visual-product-chatbot

**Merged Model:**  
https://huggingface.co/berkin9/llava-visual-product-chatbot-merged

> The live demo runs on Hugging Face ZeroGPU. The first request may take longer while GPU resources are allocated.

---

## 📸 What Can It Do?

Upload a product image and ask questions such as:

```text
What type of product is this?
What color is it?
Describe this product briefly.
What product is this and what color is it?
How much does this product cost?
What material is this product made from?
What brand is this product?
```

The model was also trained to avoid confidently inventing information that cannot reliably be inferred from an image.

For example:

```text
User: How much does this product cost?

Assistant:
The price cannot be determined from the image.
```

```text
User: What material is this product made from?

Assistant:
The material cannot be reliably determined from the image.
```

This introduces a simple form of **uncertainty-aware visual question answering** rather than forcing an answer for every question.

---

## 🧠 Project Overview

The project is built on:

- **LLaVA-1.5-7B** as the base vision-language model
- **QLoRA** for parameter-efficient fine-tuning
- **4-bit NF4 quantization** during fine-tuning
- **Hugging Face Transformers**
- **PEFT / LoRA**
- **PyTorch**
- **Gradio**
- **Hugging Face Hub**
- **Hugging Face Spaces / ZeroGPU**

The fine-tuned LoRA adapter is also merged with the original LLaVA model to create a standalone deployment model.

---

## 🏗️ Architecture

The system follows the following pipeline:

```text
                     ┌─────────────────────┐
                     │    Product Image    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   LLaVA Processor   │
                     │ Image + Text Prompt │
                     └──────────┬──────────┘
                                │
                                ▼
              ┌────────────────────────────────┐
              │       LLaVA-1.5-7B             │
              │                                │
              │   Vision Encoder               │
              │          +                     │
              │   Language Model               │
              │          +                     │
              │   Fine-Tuned LoRA Adapters     │
              └───────────────┬────────────────┘
                              │
                              ▼
                     ┌─────────────────────┐
                     │ Generated Response  │
                     └─────────────────────┘
```

For deployment, the trained LoRA weights are merged into the base model:

```text
LLaVA-1.5-7B
      +
Fine-Tuned LoRA Adapter
      │
      ▼
 merge_and_unload()
      │
      ▼
Standalone Merged Model
      │
      ▼
Hugging Face ZeroGPU
      │
      ▼
Gradio Web Application
```

This avoids loading the PEFT adapter separately during production inference.

---

## 📊 Dataset

The project uses the Hugging Face dataset:

**`ashraq/fashion-product-images-small`**

The dataset contains fashion product images and structured metadata including:

- Product category
- Article type
- Base color
- Gender
- Usage
- Season
- Product display name

For the experiment, a subset of **2,000 samples** was selected and split into:

| Split | Samples |
|---|---:|
| Training pool | 1,600 |
| Validation pool | 200 |
| Test pool | 200 |

A smaller training configuration was used for the final portfolio experiment to keep fine-tuning computationally efficient.

---

## 💬 Visual Instruction Dataset

The original product metadata was transformed into multimodal question-answer pairs.

Instead of training on a single fixed question format, multiple prompt templates were introduced to improve robustness to different user phrasings.

### Question Categories

The instruction dataset contains seven types of questions:

| Type | Example |
|---|---|
| Category | `What type of product is this?` |
| Color | `What color is this product?` |
| Description | `Describe this product briefly.` |
| Category + Color | `What product is this and what color is it?` |
| Unknown Brand | `What brand is this product?` |
| Unknown Price | `How much does this product cost?` |
| Unknown Material | `What material is this product made from?` |

Each category contains multiple alternative question templates.

For example, category questions include:

```text
What type of product is this?
What kind of item is shown here?
What product is shown in the image?
Can you identify this product?
What kind of product is this?
```

This was introduced after observing that training with fixed question wording could lead to poor generalization when semantically equivalent questions were phrased differently.

---

## 🛡️ Handling Unknown Information

Some product properties cannot reliably be inferred from an image alone.

Instead of teaching the model to hallucinate these attributes, explicit uncertainty examples were added during instruction tuning.

### Brand

```text
The brand cannot be reliably determined from the image.
```

### Price

```text
The price cannot be determined from the image.
```

### Material

```text
The material cannot be reliably determined from the image.
```

This is a lightweight experiment in teaching a multimodal model when **not** to make unsupported claims.

---

## ⚙️ Fine-Tuning Strategy

Fine-tuning a 7B parameter multimodal model directly is computationally expensive.

This project therefore uses **QLoRA (Quantized Low-Rank Adaptation)**.

### Quantization

The base model is loaded using 4-bit quantization:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

### LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

The adapter targets attention projection modules matching these names in the model architecture.

> **Implementation note:** because these projection names also occur in parts of the vision tower, the current target-module matching is not limited exclusively to the language model.

This is intentionally documented rather than claiming that the vision encoder was completely frozen.

---

## 🏋️ Training Configuration

The final fast portfolio experiment used:

| Parameter | Value |
|---|---|
| Base model | `llava-hf/llava-1.5-7b-hf` |
| Training samples | 300 |
| Validation samples | 50 |
| Epochs | 1 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | `2e-4` |
| Optimizer | Paged AdamW 8-bit |
| Precision | FP16 |
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LR scheduler | Cosine |
| Gradient checkpointing | Enabled |

The objective of this configuration was not large-scale benchmark optimization, but to build and validate a complete multimodal fine-tuning and deployment pipeline within limited GPU resources.

---

## 🎯 Training Objective

Training examples follow the LLaVA conversational structure:

```text
USER:
<image>
What color is this product?

ASSISTANT:
The product is black.
```

During training, the user prompt tokens are masked from the loss.

Therefore, the model is optimized primarily on the assistant response rather than being trained to reproduce the input prompt.

Conceptually:

```text
USER TOKENS       → ignored in loss
IMAGE TOKENS      → context
ASSISTANT TOKENS  → supervised
```

---

## 🔧 Inference

At inference time, an image and question are processed together:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }
]
```

Generation uses deterministic decoding:

```python
model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
)
```

This keeps answers short and relatively stable for product-oriented questions.

---

## 🖥️ Gradio Interface

The application provides an interactive interface where users can:

1. Upload a product image
2. Enter a natural-language question
3. Send the image and question to the model
4. Receive the generated visual response
5. Continue asking additional questions through the chat-style UI

The interface maintains the visible chat history for usability.

However, the current inference implementation processes the **current image and current question for each request**. Previous conversational turns are displayed in the UI but are not currently passed back to the model as conversational context.

This distinction is important: the current version provides a **chat-style interface**, rather than fully context-aware multi-turn reasoning.

---

## ☁️ Deployment

The application is deployed using:

```text
Hugging Face Model Hub
        │
        ├── LoRA Adapter
        │
        └── Merged Model
                │
                ▼
        Hugging Face Space
                │
                ▼
             ZeroGPU
                │
                ▼
             Gradio
```

### Why Merge the LoRA Adapter?

The original deployment architecture loaded:

```text
Base LLaVA
    +
PEFT LoRA Adapter
```

During ZeroGPU deployment, separate PEFT adapter loading introduced compatibility issues.

The adapter was therefore merged into the base model:

```python
merged_model = peft_model.merge_and_unload()
```

The resulting standalone model can be loaded directly during deployment without requiring PEFT adapter loading at runtime.

---

## 📁 Suggested Repository Structure

```text
llava-visual-product-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── llava_product_chatbot.ipynb
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── dataset.py
│
└── assets/
    ├── demo.png
    └── architecture.png
```

The training notebook can remain the primary reproducible experiment while the `src/` directory can be used for cleaner standalone implementations.

---

## 🧪 Example Results

### Product Category

```text
User:
What type of product is this?

Assistant:
The product category is shirt.
```

### Color Recognition

```text
User:
What color is it?

Assistant:
The product is black.
```

### Unknown Price

```text
User:
How much does this product cost?

Assistant:
The price cannot be determined from the image.
```

### Unknown Material

```text
User:
What material is this product made from?

Assistant:
The material cannot be reliably determined from the image.
```

Results depend on the input image and should not be interpreted as benchmark-level performance.

---

## ⚠️ Limitations

This project is an experimental portfolio implementation rather than a production product-recognition system.

Current limitations include:

- Fine-tuning uses a relatively small training subset.
- The source dataset is primarily fashion-oriented.
- Performance can degrade for images far outside the training distribution.
- Free-form questions outside the trained instruction patterns may produce unreliable answers.
- Visual attributes can occasionally be misclassified.
- The model does not have access to external product databases.
- Price, brand, and material cannot necessarily be determined visually.
- The chat interface displays conversation history, but previous turns are not yet included in model context.
- The project has not been evaluated against a large standardized multimodal benchmark.

The model should therefore be treated as a demonstration of multimodal fine-tuning and deployment rather than a production-ready product intelligence system.

---

## 🔮 Future Improvements

Possible extensions include:

- Larger and more diverse training datasets
- True context-aware multi-turn conversation
- Product retrieval using vector databases
- Retrieval-Augmented Generation (RAG)
- Integration with product catalogs and e-commerce APIs
- Structured product attribute extraction
- Improved hallucination evaluation
- Automated multimodal evaluation
- Broader prompt diversity
- Comparison of different LoRA configurations
- Evaluation against the original base LLaVA model
- Support for multiple uploaded images
- Product similarity search

A future architecture could combine visual understanding with retrieval:

```text
Product Image
     │
     ▼
Vision-Language Model
     │
     ├── Visual attributes
     │
     ▼
Vector / Product Search
     │
     ▼
Product Database
     │
     ▼
Grounded Product Assistant
```

This would allow the system to answer questions involving real product information such as price, stock, specifications, or related products instead of attempting to infer them visually.

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![LLaVA](https://img.shields.io/badge/Model-LLaVA--1.5--7B-purple)
![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![Hugging Face](https://img.shields.io/badge/Deployment-Hugging%20Face-yellow)

---

## 💡 Key Learning Outcomes

This project demonstrates practical experience with:

- Vision-Language Models (VLMs)
- Multimodal prompt formatting
- Visual Question Answering
- Instruction dataset generation
- QLoRA and parameter-efficient fine-tuning
- 4-bit model quantization
- Hugging Face Transformers
- PEFT / LoRA
- PyTorch
- Prompt-loss masking
- Multimodal inference
- Model merging
- Gradio application development
- Hugging Face Model Hub
- GPU model deployment
- ZeroGPU deployment
- Debugging ML dependency and deployment issues

---

## 📚 Models and Resources

**Base Model**

`llava-hf/llava-1.5-7b-hf`

**Dataset**

`ashraq/fashion-product-images-small`

**Fine-Tuned LoRA Adapter**

https://huggingface.co/berkin9/llava-visual-product-chatbot

**Merged Deployment Model**

https://huggingface.co/berkin9/llava-visual-product-chatbot-merged

**Live Demo**

https://huggingface.co/spaces/berkin9/llava-visual-product-chatbot

---

## 👤 Author

**Berkin Akbiyik**

Computer Engineer and MSc Artificial Intelligence student focused on software engineering, machine learning, multimodal AI, and applied AI systems.

---

## 📄 Disclaimer

This project was developed for educational and portfolio purposes.

Model outputs may be incorrect or incomplete and should not be used as authoritative product information.
