import os
import pandas as pd
import numpy as np
import re
import ast
from huggingface_hub import login
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional

df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)

filtered_df = df[df['question_type'] == 'metrics-generated']


class FinancePredictor:
    def __init__(self, model_name: str):
        # Load the Llama model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine the available device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Define system prompt
        self.system_prompt = (
            "You are an expert AI system that accurately answers financial questions. "
            "Each question will ask you to calculate a number - your output for this type of question "
            "should be a number representing the correct answer to the calculation. Please answer each "
            "question with total accuracy, performing all necessary calculations without skipping or simplifying any steps."
        )

    def make_prediction(self, question_text: str) -> str:
        """Generate a prediction for a given question."""
        prompt = f"{self.system_prompt}\nUser: {question_text}\nAI:"
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the model's output
        outputs = self.model.generate(
            input_ids=tokenized_prompt['input_ids'],
            attention_mask=tokenized_prompt['attention_mask'],
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            max_length=1024  # Adjust max_length to prevent excessive output
        )

        # Extract and decode the response
        response_tokens = outputs[0][tokenized_prompt['input_ids'].shape[-1]:]
        decoded_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        return decoded_response

    def make_predictions(self, df: pd.DataFrame) -> List[Dict[str, Optional[str]]]:
        """Generate predictions for each question in the DataFrame."""
        outputs = []
        for _, row in df.iterrows():
            question_text = row.get('question', '')
            try:
                prediction = self.make_prediction(question_text)
                print(f"{'=' * 50}\nQuestion: {question_text}\nPrediction: {prediction}\n")
            except Exception as e:
                print(f"Error processing question {row.get('financebench_id')}: {e}")
                prediction = None

            outputs.append({'financebench_id': row.get('financebench_id'), 'response': prediction})
        return outputs

    @staticmethod
    def extract_final_answer(complete_answer_text: str) -> Optional[float]:
        """Extract the final numeric answer from the model's output."""
        text = re.sub(r'[,$]', '', complete_answer_text).strip()
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if not numbers:
            return None
        last_number = float(numbers[-1])
        return int(last_number) if last_number.is_integer() else last_number

    @staticmethod
    def force_int(df: pd.DataFrame) -> pd.DataFrame:
        """Force integer conversion for numeric responses."""
        for idx, row in df.iterrows():
            response = row.get('response')
            if isinstance(response, (float, int)) and float(response).is_integer():
                df.at[idx, 'response'] = int(response)
        return df

if __name__ == "__main__":
    df = filtered_df
    model_name = "llmware/bling-1b-0.1"
    model = FinancePredictor(model_name)
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
file_path = os.path.join(output_dir, "bling-1b-0.1.csv")
results_df.to_csv(file_path, index=False)

print(f"File saved successfully to: {file_path}")

