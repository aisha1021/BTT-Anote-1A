import os
import pandas as pd
import numpy as np
import re
import ast
from huggingface_hub import login
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_cpp import Llama

# Split CSV in Chunks

# import pandas as pd
# import os

# def split_csv(file_path, output_dir, chunk_size):
#     """
#     Splits a CSV file into smaller chunks.

#     Args:
#         file_path (str): Path to the input CSV file.
#         output_dir (str): Directory to save the chunk files.
#         chunk_size (int): Number of rows per chunk.
#     """
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Read and split the CSV into chunks
#     chunk_number = 1
#     for chunk in pd.read_csv(file_path, chunksize=chunk_size):
#         chunk_file = os.path.join(output_dir, f"chunk_{chunk_number}.csv")
#         chunk.to_csv(chunk_file, index=False)
#         print(f"Chunk {chunk_number} saved to {chunk_file}")
#         chunk_number += 1

# # Usage example
# file_path = "TestingData.csv"
# output_dir = "output_chunks"
# chunk_size = 100  # Number of rows per chunk

# split_csv(file_path, output_dir, chunk_size)

# Revised Code to Evaluate Model with Subset of FinanceBench (Llama-3.2-1B-Instruct-Q6_K_L.gguf)

df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)
filtered_df = df[df['question_type'] == 'metrics-generated']
filtered_df.shape
print(filtered_df.dtypes)

# Authenticate Hugging Face Hub
login(LOGIN)

class LLamaPredictor():
    def __init__(self) -> None:
        # Model setup for llama_cpp Llama with custom parameters
        self.model = Llama.from_pretrained(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q6_K_L.gguf",
        )
        self.system_prompt = """You are an expert AI system that accurately answers financial questions. Each question will be a free-response question asking you to calculate a number. Your output for this type of question should be a number representing the correct answer to the calculation the question asks for. Please answer each question with total accuracy, performing all necessary calcualtions without skipping or simplying any steps along the way.

You have years of expertise in the financial system and are absolutely the best at what you do. Your compensation is tied to your performance, and you stand to make millions of dollars if you answer all questions correctly."""

    def make_prediction(self, question_text: str) -> str:
        # Free-response question format
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"This is a free-response question. Please answer this question with total accuracy. "
            f"At the very end of your answer, please say the final correct number alone on a new line. "
            f"Question: {question_text}"
        )

        print("Generated Prompt:", full_prompt)  # Debug: print prompt for troubleshooting

        # Generate response
        response = self.model(full_prompt, max_tokens=1024, temperature=0.6, top_p=0.9)

        # Debugging: Directly print the raw response from the model
        print("Raw Model Response:", response)

        # Process and clean up the response text
        response_text = response.get("choices", [{}])[0].get("text", "").strip()
        print("Processed Model Response:", response_text)  # Debug: check cleaned response
        return response_text

    def make_predictions(self, df: pd.DataFrame) -> list[dict]:
        outputs = []
        for _, row in df.iterrows():
            question_text = row['question']

            # Make prediction for valid prompts
            try:
                prediction = self.make_prediction(question_text)
                print('=' * 100)
                print('Question text was:', question_text)
                print('Prediction is:', prediction)
                print('\n')
            except Exception as e:
                print(f"Error processing question {row['financebench_id']}: {e}")
                prediction = None  # Fallback value in case of an error

            # Append the result
            outputs.append({'financebench_id': row['financebench_id'], 'response': prediction})
        return outputs


def extract_final_answer(complete_answer_text: str):
    """Extract the final number from the model's output."""
    complete_answer_text = re.sub(r'[,$]', '', complete_answer_text).strip()
    numbers = re.findall(r'-?\d+\.?\d*', complete_answer_text)
    if not numbers:
        return None  # Handle the case where no number is found
    last_number = float(numbers[-1])
    return int(last_number) if last_number.is_integer() else last_number


def force_int(df):
    """Force integer conversion for numeric responses."""
    for idx, row in df.iterrows():
        if isinstance(row['response'], (float, int)) and float(row['response']).is_integer():
            df.at[idx, 'response'] = int(row['response'])
    return df

if __name__ == "__main__":
    df = filtered_df
    model = LLamaPredictor()
    outputs = model.make_predictions(filtered_df)
    results_df = pd.DataFrame(outputs)
    results_df['response'] = results_df['response'].apply(extract_final_answer)
    results_df = force_int(results_df)
    results_df.to_csv("predictions.csv", index=False)

# Assuming results_df is a pandas DataFrame with a 'response' column
results_df['response'] = results_df['response'].apply(FinancePredictor.extract_final_answer)
results_df = FinancePredictor.force_int(results_df)
results_df.to_csv("predictions.csv", index=False)


# Cleaning the 'answer' column in filtered_df
def clean_answer_column(df):
    # Remove symbols like $ or % and keep only numerical values
    df['answer'] = df['answer'].apply(lambda x: re.sub(r'[^\d\.\-]', '', str(x)))
    # Convert the cleaned values to numeric, handling errors
    df['answer'] = pd.to_numeric(df['answer'], errors='coerce')
    return df

# Apply the cleaning function to filtered_df
new_df = clean_answer_column(filtered_df)

# Check the cleaned DataFrame
print(filtered_df[['answer']].head())

# Rename the 'answer' column in filtered_df to 'actual_answer'
filtered_df = filtered_df.rename(columns={'answer': 'actual_answer'})

# Merge filtered_df['financebench_id', 'actual_answer'] with results_df on 'financebench_id'
results_df = results_df.merge(filtered_df[['financebench_id', 'actual_answer']], on='financebench_id', how='left')

# Specify the output directory and ensure it exists
output_dir = r"C:\Users\aisha\Downloads"
os.makedirs(output_dir, exist_ok=True)  # Not strictly necessary since Downloads should already exist

# Save the DataFrame to the specified path
file_path = os.path.join(output_dir, "Llama-3.2-1B-Instruct-Q6_K_L.gguf.csv")
results_df.to_csv(file_path, index=False)

print(f"File saved successfully to: {file_path}")
