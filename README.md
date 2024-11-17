# Project Overview: Anote - BTT 1A Challenge

## Objective
The goal of this challenge is to develop an AI model capable of accurately answering financial questions with either numerical or categorical answers, using the bizbench testing dataset. We would like to compare multiple models / approaches, such as what is found in this [github repo](https://github.com/nv78/Anote/tree/main/Benchmarking-RAG) and in this [talk](https://www.youtube.com/watch?v=0gqCpghZXEE), as well as comparing multiple fine tuning methods.

## Dataset
The bizbench testing dataset is available [here](https://drive.google.com/drive/folders/1-uIhlGUChcHeizyaTh7E6dF3D4U1srhl) in three formats:
- JSON
- Google Sheets
- CSV


## Resources
To assist in building your model, the following resources are provided:

1. **Fine Tuning Guide**: A comprehensive guide on various methods to [fine-tune large language models (LLMs)](https://docs.anote.ai/api-prompting/overview.html) using financial datasets and documents, including links to:
   - [Finance Bench dataset](https://huggingface.co/datasets/PatronusAI/financebench)
   - [RAG Instruct dataset](https://huggingface.co/datasets/llmware/rag_instruct_benchmark_tester)
   - [10-K Edgar dataset](https://www.sec.gov/search-filings)

2. **Bizbench Paper**: Research conducted by the Kensho team on Q&A models using LLMs is [here](https://arxiv.org/abs/2311.06602).

3. **Kensho Benchmarks GitHub Repo**: [Github repo](https://github.com/kensho-technologies/benchmarks-pipeline) contains sample code and results for answering questions on the bizbench dataset. FAQs are found on the main website [here](https://benchmarks.kensho.com/).

4. **Private Chatbot**: Research papers published by the Anote team, along with an [evaluation guide](https://docs.anote.ai/api-prompting/example8.html) for Q&A models. Additional information is available in the associated [talks](https://docs.google.com/document/d/1DLvX4wk_IZBUm0HMFEK4w3NG0oV75JWcsynhK_Va8fg/edit).

## Submission Instructions
To submit your results to the [S&P Global AI Benchmarks by Kensho](https://benchmarks.kensho.com/):

1. **Output CSV**: 
   - Ensure your model's output is in CSV format, with no header and only two columns: `id` and `answer`. Here is a [sample CSV](https://github.com/kensho-technologies/benchmarks-pipeline/blob/main/results/Mistral-7B-v0.1-cot.csv)
   - The `answer` column should contain either the numerical answer or the index of the answer if it is a multiple-choice question.

2. **Kensho Portal**: 
   - Sign into the Kensho portal with your school or business email. Okta Verify might be required for authentication.
   - For submission fields, use the following details:
     - **Model Name**: `[First Name - Last Name]’s Model - Attempt N`
     - **Organization**: Anote
     - **Number of Parameters**: Private / Unknown
     - **Organization Website**: [Anote Website](https://anote.ai/)
   - **Upload**: Submit the output CSV to Kensho Benchmarks.
   - **Leaderboard**: Check your performance on the S&P Global AI Benchmarks leaderboard in a few days.

## Final Submission
The final technical challenge submission should contain the following files:

1. **main.py**: Python code with your solution.
2. **README.md**: Documentation discussing your technical approach, your experience with Kensho Benchmarks, and a breakdown of different model performance on the benchmarks.
3. **output.csv**: The final dataset submitted to Kensho Benchmarks for evaluation by the Anote team.

## Final Presentation
At the end of the project, we will record a final presentation, that will be similar to what we have on our [AI day](https://anote.ai/aiday)
