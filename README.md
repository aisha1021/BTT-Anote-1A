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

## 🧠 Models Evaluated

Due to hardware limitations (RAM/CPU constraints), only **quantized** versions of LLMs were used. Larger models such as LLaMA 3.1–8B were attempted but could not be deployed in Colab due to resource exhaustion.

| Model | Parameters | Description | Pros | Cons |
|-------|------------|-------------|------|------|
| **meta-llama/Llama-3.1-8B-Instruct** | 8B | High-performance, instruction-tuned model | Strong performance on NLP tasks | Too large to run on standard Colab GPU |
| **LLaMA 3.2–1B Instruct (Q6_k_L.gguf)** | 1B (Quantized) | Efficient instruction-tuned model | Low memory usage | Fine-tuning not feasible due to quantization |
| **BLING-1B-0.1** | 1B | Lightweight instruction model for CPU usage | Extremely low resource requirement | Reduced performance due to small size |
| **TinyLLaMA-1.1B-1T OpenOrca (Q5_K_M.gguf)** | 1.1B (Quantized) | Compact model trained on OpenOrca dataset | Balanced size and performance | Still less capable for complex reasoning |

---

## 📈 Model Performance

| Model | Accuracy | Residual | Mean Absolute Error |
|-------|----------|----------|----------------------|
| **LLaMA 3.2–1B Instruct (Q6_k_L.gguf)** | 6.98% | 1094 | 2435 |
| **BLING-1B-0.1** | 9.68% | 3380 | 4652 |
| **TinyLLaMA-1.1B-1T OpenOrca** | **13.04%** | 1859 | 2642 |

> Note: Performance metrics are computed using custom evaluation scripts over the FinanceBench dataset. Accuracy is calculated based on exact matches for multiple-choice and numerical responses.

---

## 📂 Dataset

All experiments were conducted on the [**FinanceBench**](https://huggingface.co/datasets/PatronusAI/financebench) dataset, which contains real-world, finance-specific QA pairs categorized by context types and answer formats.

---

## ⚙️ Environment Notes

- Google Colab was the primary compute environment.
- Quantized `.gguf` models were used to reduce memory usage.
- Fine-tuning was not feasible for most models due to quantization constraints.

---
