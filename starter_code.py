
import os
import pandas as pd
import numpy as np
import re
import ast
from huggingface_hub import login
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch


class LLamaPredictor():
    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.system_prompt = """You are an expert AI system that accurately answers financial questions. Each question will either be i) multiple choice, and ask you to choose the correct answer from a list of possible answers. Each possible answer will have a numerical label associated with it. Please choose the correct numerical label; OR ii) a free-response answer asking you calculate a number - your output for this type of question should be a number representing the correct answer to the calcuation the question asks for. For each question, you will be told if it is multiple choice or free-response. Please answer each question with total accuracy, performing all necessary calcualtions without skipping or simplying any steps along the way.

You have years of expertise in the financial system and are absolutely the best at what you do. Your compensation is tied to your performance, and you stand to make millions of dollars if you answer all questions correctly."""

    def make_prediction(self, question_text: str, answer_options:list[str]) -> str:
        if len(answer_options) == 0:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"This is a free-response question. Please answer this question with total accuracy, performing all necessary calculations without skipping or simplifying any steps. At the very end of your answer, please say the final correct number alone on a new line. Question: {question_text}"}
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"This is a multiple-choice question. Please answer this question with total accuracy, perfomring all neceessary steps wtihout simplification or shortcuts to choose the correct answer. At the very end of your output, please say the numerical label associated with the correct answer alone on a new line. Remember, this label should be the first part of the input assocaited with the correct answer [for instance, for (0, correct option) you would say '0'] and can ONLY be one of the numbers you saw associated with a question choice. Question: {question_text}. Possible answers: {answer_options}"}
            ]

        tokenized_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)


        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            tokenized_prompt,
            max_new_tokens=2056,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][tokenized_prompt.shape[-1]:]
        return(self.tokenizer.decode(response, skip_special_tokens=True))



    def make_predictions(self) -> None:
        df = pd.read_csv(f'{root_path}/tech_challenge/TestingData.csv')
        df = df.drop('Unnamed: 0', axis=1)
        df['options'] = df['options'].fillna(-1)
        df['context'] = df['context'].fillna(-1)
        outputs = []
        for input_id, question_text, answer_options, context in zip(df['id'].values, df['question'].values, df['options'].values, df['context'].values):
            if answer_options == -1:
                answer_options = []
                if context != -1:
                    question_text += f"; Here's the necessary context to answer that question: {context}"
            elif type(answer_options) == type(''):
                answer_options = ast.literal_eval(answer_options)
            i = 0
            new_options = []
            for x in answer_options:
                new_options.append((i, x))
                i+=1
            prediction = self.make_prediction(question_text, new_options)
            print('='*100)
            print('question text was:', question_text)
            print('options were:', new_options)
            print('prediction is:', prediction)
            print('\n')
            outputs.append({'id':input_id, 'response':prediction})
        return outputs

def extract_final_answer(complete_answer_text: str):
    complete_answer_text = complete_answer_text.replace(",", "")
    complete_answer_text = complete_answer_text.replace("$", "")
    complete_answer_text = complete_answer_text.strip()
    last_number = re.findall('-?\d+\.?\d*', complete_answer_text)[-1]
    last_number = float(last_number)
    if last_number.is_integer():
        last_number = int(last_number)
    return last_number


def force_int(df):
    for idx, row in df.iterrows():
        if row['response'].is_integer():
            df.loc[idx, 'response'] = int(row['response'])
    return df

if __name__ == "__main__":
    model = LLamaPredictor()
    outputs = model.make_predictions()
    df = pd.DataFrame(outputs)
    df['response'] = df['response'].apply(extract_final_answer)
    df['response'] = df['response'].astype('object')
    df = force_int(df)
    df.to_csv('output.csv', index=False, header=False)