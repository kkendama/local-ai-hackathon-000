# パッケージのインポート
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from tqdm import tqdm
import pandas as pd

model_name = "tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
)

def generate_prompt(text):
    prompt= f'''あなたはイラストレーターです。自分のホームページを作りたいと思っています。どのような内容を掲載していくべきか、そして、どのようなデザインにすべきか意見をください。 また、自分のイラストを載せるだけでなく、他のクリエイターとの交流も深めたいと思っています。SNSを活用したコミュニティや、オンラインでのアートコミュニティについてもアドバイスをもらいたいです。 さらに、自分の技術向上のため、他のアーティストとの交流や、自分自身のアイデアを広げるためのアプローチ方法も提案していただきたいです。'''
    return prompt

def chat(prompt):
    # メッセージリストの準備
    messages = [
        {"role": "user", "content": prompt},
    ]

    # 推論の実行
    with torch.no_grad():
      token_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
      output_ids = model.generate(
          token_ids.to(model.device),
          temperature=0.9,
          do_sample=True,
          #top_p=0.95,
          #top_k=40,
          max_new_tokens=4098,
          repetition_penalty=1.05
      )
    output = tokenizer.decode(output_ids[0][token_ids.size(1) :])
    return output

prompt = generate_prompt("")
print(chat(prompt))
