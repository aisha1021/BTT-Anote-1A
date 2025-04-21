# 📊 Anote - BTT 1A Challenge

## 🔍 Project Overview

The **Anote BTT 1A Challenge** focuses on developing a high-performing AI system capable of answering complex financial questions using the [**FinanceBench**](https://huggingface.co/datasets/PatronusAI/financebench) dataset. The dataset contains a mix of **numerical** and **categorical** questions designed to evaluate models on real-world financial reasoning.

This challenge explores multiple modeling approaches—including **retrieval-augmented generation (RAG)** and **fine-tuned large language models (LLMs)**—to determine which strategy delivers optimal performance in financial question answering.

---

## 🎯 Objectives

- Design an AI system that:
  - Accurately answers financial questions requiring both numeric and multiple-choice responses.
  - Performs competitively on **S&P Global AI Benchmarks** by **Kensho**.
  - Can operate within compute-constrained environments such as Google Colab.

---

## 📂 Dataset

All experiments were conducted on the [**FinanceBench**](https://huggingface.co/datasets/PatronusAI/financebench) dataset, a benchmark specifically designed to test the reasoning capabilities of AI models in complex financial domains. 

FinanceBench includes **expert-annotated** question-answer (QA) pairs sourced from **earnings call transcripts** of publicly traded companies. Each QA pair is structured around a specific **financial task** and labeled with:

- **Context Type**: 
  - `qa_text`: Questions based on long-form transcripts
  - `qa_structure`: Structured data-based questions
  - `qa_text+structure`: Combined context
- **Answer Format**:
  - `multiple_choice`: Answers are among predefined choices
  - `numerical`: Answers are precise numerical values

The dataset challenges models to perform a wide range of reasoning tasks, including:

- **Numerical Reasoning**: Interpreting and calculating metrics from financial discussions  
- **Categorical Inference**: Selecting correct qualitative insights
- **Contextual Understanding**: Deriving answers from dense financial text

Due to its focus on **real-world financial reporting**, FinanceBench serves as a rigorous testbed for evaluating financial language models in enterprise-grade applications.

---

## 🛠 Tools & Platforms Used

| Component                     | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| **Python**                   | Main programming language.                                                  |
| **Pandas**                   | Data manipulation and analysis.                                             |
| **NumPy**                    | Numerical operations and data conversion.                                   |
| **Hugging Face Hub**         | Authentication and model access (`transformers`, `huggingface_hub`).        |
| **LLaMA via llama.cpp**      | Lightweight, quantized model inference (`llama_cpp`).                       |
| **Transformers (HF)**        | For loading and generating text with Hugging Face models.                   |
| **Torch (PyTorch)**          | Backend for HF model inference.                                             |
| **Regular Expressions (re)** | Extracting numerical answers from model outputs.                           |
| **Jupyter/VS Code**          | Recommended environments for running and experimenting with this pipeline. |

---

## 🧠 Models Evaluated

Due to hardware limitations (RAM/CPU constraints), only **quantized** versions of LLMs were used. Larger models such as LLaMA 3.1–8B were attempted but could not be deployed in Colab due to resource exhaustion.

| Model | Parameters | Description | Pros | Cons |
|-------|------------|-------------|------|------|
| **meta-llama/Llama-3.1-8B-Instruct** | 8B | High-performance, instruction-tuned model | Strong performance on NLP tasks | Too large to run on standard Colab GPU |
| **LLaMA 3.2–1B Instruct (Q6_k_L.gguf)** | 1B (Quantized) | Efficient instruction-tuned model | Low memory usage | Fine-tuning not feasible due to quantization |
| **BLING-1B-0.1** | 1B | Lightweight instruction model for CPU usage | Extremely low resource requirement | Reduced performance due to small size |
| **TinyLLaMA-1.1B-1T OpenOrca (Q5_K_M.gguf)** | 1.1B (Quantized) | Compact model trained on OpenOrca dataset | Balanced size and performance | Still less capable for complex reasoning |

---

## 🧪 Techniques & Methodology

- **Photometric Inference**: Using LLaMA to answer questions based on prompt-driven inference.
- **Custom Prompt Engineering**: Carefully designed system prompts guide the model to return structured, numerical answers.
- **Token Sampling**: Uses `top_p` and `temperature` to control creativity and focus of language generation.
- **Chunked Data Processing**: Supports CSV splitting for batch evaluation.
- **Postprocessing & Evaluation**: Cleans responses, extracts final numbers, and evaluates them against ground truth.

---

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| **LLaMA 3.2–1B Instruct (Q6_k_L.gguf)** | 6.98% |
| **BLING-1B-0.1** | 9.68% |
| **TinyLLaMA-1.1B-1T OpenOrca** | **13.04%** |

> Note: Performance metrics are computed using custom evaluation scripts over the FinanceBench dataset. Accuracy is calculated based on exact matches for multiple-choice and numerical responses.

### 📉 Why Are Accuracies Low?

- **Small, Quantized Models**: Due to Colab constraints, all models were under 2B parameters and quantized, reducing their ability to reason over complex financial data.
- **No Fine-Tuning or RAG**: Models were used off-the-shelf without domain-specific fine-tuning or retrieval augmentation, limiting performance on specialized tasks.
- **Complexity of FinanceBench**: The dataset demands multi-hop reasoning, numerical precision, and financial literacy—challenging even for larger models.
- **Strict Evaluation Metric**: Exact match scoring penalizes near-correct answers, especially in numerical cases with minor rounding differences.

---

## ⚙️ Environment Notes

- Google Colab was the primary compute environment.
- Quantized `.gguf` models were used to reduce memory usage.
- Fine-tuning was not feasible for most models due to quantization constraints.

---
