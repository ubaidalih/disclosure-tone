import os, sys
from os.path import exists
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import login
import datasets
import ast
import os
datasets.logging.set_verbosity_error()

tone_dict = {
    'Positive': "Operating cash flow improved 30 percent, reaching $120 million during fiscal 2025.",
    'Negative': "Second-quarter sales dropped 14 percent, missing forecasts by nearly $40 million.",
    'Uncertainty':  "In FY 2022, revenue from that segment made up approximately 11 % of total sales.",
    'Litigious': "A patent-infringement lawsuit is pending in federal court.",
    'Strong Modal': "Management must accelerate cost-reduction measures.",
    'Weak Modal': "Profits might decline under adverse market conditions.",
    'Constraining': "Employees are prohibited from trading shares during blackout periods."
}

@torch.inference_mode()
def get_logprobs(model, tokenizer, prompt, label_ids=None, label_attn=None, device='cuda'):
    if isinstance(prompt, str):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
        input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    if isinstance(prompt, list):
        inputs = tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt", truncation=True, max_length=8192, return_dict=True).to(device)
        input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2) * inputs['attention_mask'].unsqueeze(2), 2, output_ids.unsqueeze(2))
    return logprobs.sum().cpu()

@torch.inference_mode()
def predict_classification(model, tokenizer, prompt, labels, template_length=1, device='cuda'):
    probs = []
    for label in labels:
        final_prompt = deepcopy(prompt)
        if template_length == 0:
            final_prompt = final_prompt.replace('[LABELS_CHOICE]', label)
        else:
            final_prompt[template_length]['content'] = final_prompt[template_length]['content'].replace('[LABELS_CHOICE]', label)
        probs.append(get_logprobs(model, tokenizer, final_prompt, device=device))
    return probs

def load_data():
    data = pd.read_csv('./Text_Database.csv')
    # drop_ticker = ['MBMA', 'GOTO', 'ASII', 'BBCA', 'BBNI', 'BBRI', 'BMRI', 'MAPI', 'MEDC', 'INKP']
    # data = data[~data['Ticker'].isin(drop_ticker)]
    data = data[(data.iloc[:, 3:] != 0).any(axis=1)]
    return data.reset_index()

def load_model(model_name='CohereForAI/aya-expanse-8b'):
    login(os.environ['HF_TOKEN'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    return model, tokenizer

def generate_prompt(query, disclosure_tone, prompt_type, task):
    messages = []
    messages_length = 0
    keys = list(tone_dict)
    n = len(keys)
    instruction = "Classify the text into one or more of the following disclosure tone: Positive, Negative, Uncertainty, Litigious, Strong Modal, Weak Modal, Constraining."
    if task == 'zs':
        if prompt_type == 'multiclass':
            return instruction + '\n' + 'Text: ' + query + '\n' + 'Label: [LABELS_CHOICE]'
        else:
            question = f"Does the text contain {disclosure_tone} disclosure tone?"
            return instruction + '\n' + 'Text: ' + query + '\n' + question + '\n' + 'Answer: [LABELS_CHOICE]'
    else:
        if prompt_type == 'multiclass':
            messages_length = 15
            for tone, sentence in tone_dict.items():
                messages.append({"role": "user", "content": instruction + '\n' + 'Text: ' + sentence})
                messages.append({"role": "assistant", "content": tone})
            messages.append({"role": "user", "content": instruction + '\n' + 'Text: ' + query})
            messages.append({"role": "assistant", "content": '[LABELS_CHOICE]'})
        else:
            messages_length = 29
            for idx, key in enumerate(keys):
                next_key = keys[(idx + 1) % n]
                question = f"Does the text contain {key} disclosure tone?"
                messages.append({"role": "user", "content": f"{instruction}\nText: {tone_dict[key]}\n{question}"})
                messages.append({"role": "assistant", "content": "Yes"})
                question = f"Does the text contain {next_key} disclosure tone?"
                messages.append({"role": "user", "content": f"{instruction}\nText: {tone_dict[key]}\n{question}"})
                messages.append({"role": "assistant", "content": "No"})
            question = f"Does the text contain {disclosure_tone} disclosure tone?"
            messages.append({"role": "user", "content": f"{instruction}\nText: {query}\n{question}"})
            messages.append({"role": "assistant", "content": '[LABELS_CHOICE]'})
    
    return messages, messages_length


def zs(model, tokenizer, data, prompt_type):
    labels = ['Positive', 'Negative', 'Uncertainty', 'Litigious', 'Strong Modal', 'Weak Modal', 'Constraining']
    answers = ['Yes', 'No']
    data = data.assign(Positive_Pred=0, Negative_Pred=0, Uncertainty_Pred=0, Litigious_Pred=0, Strong_Modal_Pred=0, Weak_Modal_Pred=0, Constraining_Pred=0)
    for i in tqdm(range(len(data))):
        try:
            query = data.loc[i]['Sentence']
            if prompt_type == 'multiclass':
                messages = generate_prompt(query, None, prompt_type, 'zs')
                probs = predict_classification(model, tokenizer, messages, labels, template_length=0, device='cuda')
                top2_indices = torch.topk(torch.stack(probs, dim=-1), k=2, dim=-1).indices.tolist()
                top1_label = labels[top2_indices[0]]
                top2_label = labels[top2_indices[1]]
                data.loc[i, f"{top1_label}_Pred"] = 1
                data.loc[i, f"{top2_label}_Pred"] = 2
            else:
                for label in labels:
                    messages = generate_prompt(query, label, prompt_type, 'zs')
                    probs = predict_classification(model, tokenizer, messages, answers, template_length=0, device='cuda')
                    predicted_answer = answers[torch.argmax(torch.stack(probs, dim=-1), dim=-1).tolist()]
                    if predicted_answer == 'Yes':
                        data.loc[i, f"{label}_Pred"] = 1
        except:
            continue
    data.to_csv(f'./Result/zs_{prompt_type}_qwen.csv')

def icl(model, tokenizer, data, prompt_type):
    labels = ['Positive', 'Negative', 'Uncertainty', 'Litigious', 'Strong Modal', 'Weak Modal', 'Constraining']
    answers = ['Yes', 'No']
    data = data.assign(Positive_Pred=0, Negative_Pred=0, Uncertainty_Pred=0, Litigious_Pred=0, Strong_Modal_Pred=0, Weak_Modal_Pred=0, Constraining_Pred=0)
    for i in tqdm(range(len(data))):
        try:
            query = data.loc[i]['Sentence']
            if prompt_type == 'multiclass':
                messages, messages_length = generate_prompt(query, None, prompt_type, 'icl')
                probs = predict_classification(model, tokenizer, messages, labels, template_length=messages_length, device='cuda')
                top2_indices = torch.topk(torch.stack(probs, dim=-1), k=2, dim=-1).indices.tolist()
                top1_label = labels[top2_indices[0]]
                top2_label = labels[top2_indices[1]]
                data.loc[i, f"{top1_label}_Pred"] = 1
                data.loc[i, f"{top2_label}_Pred"] = 2
            else:
                for label in labels:
                    messages, messages_length = generate_prompt(query, label, prompt_type, 'icl')
                    probs = predict_classification(model, tokenizer, messages, answers, template_length=messages_length, device='cuda')
                    predicted_answer = answers[torch.argmax(torch.stack(probs, dim=-1), dim=-1).tolist()]
                    if predicted_answer == 'Yes':
                        data.loc[i, f"{label}_Pred"] = 1
        except:
            continue
    data.to_csv(f'./Result/icl_{prompt_type}_qwen.csv')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct')
    # model, tokenizer = load_model('SUFE-AIFLM-Lab/Fin-R1')
    data = load_data()
    zs(model, tokenizer, data, 'multiclass')
    zs(model, tokenizer, data, 'multilabel')
    icl(model, tokenizer, data, 'multiclass')
    icl(model, tokenizer, data, 'multilabel')