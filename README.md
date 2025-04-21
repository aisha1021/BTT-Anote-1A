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

## 💡 RAG Pipeline Attempt

A Retrieval-Augmented Generation (RAG) pipeline was tested to enhance financial question answering by leveraging external information. The pipeline includes two main components: retrieval and generation.

### Technical Details:
- **Retrieval System**: Uses **Dense Passage Retrieval (DPR)** with **DPRQuestionEncoder** and **DPRContextEncoder** to encode queries and passages into dense vectors, selecting the most relevant documents using cosine similarity.
  
- **Generation Model**: A **generative language model** (e.g., GPT, BART, T5) uses the retrieved passages and query to generate answers.

Despite its potential, the integration was challenging due to limited resources on Google Colab. The retrieval process was computationally expensive and time-consuming, especially for fine-tuning on financial data, and didn't improve accuracy as expected. As a result, the RAG pipeline was deprioritized in favor of more reliable methods with quantized LLMs, which performed better for the task.

---

## ⚙️ Environment Notes

- Google Colab was the primary compute environment.
- Quantized `.gguf` models were used to reduce memory usage.
- Fine-tuning was not feasible for most models due to quantization constraints.

---

## 🔮 Future Work

Looking ahead, several areas can be explored to further improve the performance and efficiency of the Anote BTT 1A Challenge system:

1. **Model Expansion and Fine-Tuning**:
   - **Fine-Tune Larger Models**: Explore fine-tuning larger models on domain-specific financial data to improve their reasoning capabilities.
   - **De-Quantize Models**: Using higher-capacity models without quantization may allow for better performance, though it may require access to more powerful hardware beyond Colab.

2. **RAG Pipeline Optimization**:
   - **Efficient Retrieval**: Investigate optimizing the Dense Passage Retrieval system to reduce computational cost, potentially by employing more efficient encoding and retrieval techniques.
   - **Custom Knowledge Base**: Build a more specialized knowledge base for financial queries to improve the relevance and accuracy of the retrieved passages.

3. **Evaluation Improvements**:
   - **Loosening Evaluation Criteria**: Experiment with alternative evaluation metrics that allow for near-correct answers in numerical cases (e.g., allowing for slight rounding differences).
   - **Cross-Validation with Other Datasets**: Test models on additional financial benchmarks to ensure generalization across different domains.

4. **Multimodal Approaches**:
   - **Incorporating Structured Data**: Incorporate structured financial data such as stock prices, financial statements, and other real-time market data to improve answer accuracy, especially for numerical questions.
   - **Visual Reasoning**: Experiment with integrating visual models that can analyze financial reports or charts, expanding the types of queries the model can handle.

5. **Scalability and Resource Efficiency**:
   - **Deploy on Dedicated Infrastructure**: Transition the project to cloud platforms with better resources (e.g., AWS, Google Cloud) for running larger models and conducting fine-tuning, overcoming the limitations of Google Colab.
   - **Optimization for Edge Devices**: Investigate lightweight model deployment options for real-time applications on mobile devices or other constrained environments.

By focusing on these areas, the project can evolve to better handle complex financial queries and achieve higher performance on the FinanceBench and other benchmarks.

---
