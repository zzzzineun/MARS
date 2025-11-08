import gc
import os
from datasets import Dataset
import pandas as pd
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# ----------------------------------------
## base_model_path: 학습 대상 모델 경로
## new_model_path: 학습 후 모델 저장할 경로
## training_data_path: 학습 대상 경로 (prompt / chosen / rejected 컬럼 필수)
# ----------------------------------------
base_model_path = "/home/ec2-user/models/llama3.1-8b-instruct/"
new_model_path = "./llama8binst-ORPO-omi"
training_data_path = "./data_processing/answers_dpo_omi.csv"
torch_dtype = torch.bfloat16
attn_implementation = "sdpa"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_path,  # 로컬 경로 지정
    max_seq_length = 2048,
    dtype = None,  # auto detection
    load_in_4bit = True,  # 4bit 양자화로 메모리 절약
)
# LoRA 어댑터 추가
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

df = pd.read_csv(training_data_path)
df = df.dropna(how='any')
dataset = Dataset.from_pandas(df) 
splits = dataset.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

orpo_config = ORPOConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,  # 실질적 batch size = 8
    warmup_steps = 10,
    max_steps = 100,  # 또는 num_train_epochs
    learning_rate = 5e-5,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",  # 메모리 절약
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
)
trainer = ORPOTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    args = orpo_config,
)

# 학습 실행
trainer.train()
# 학습 직후 병합된 모델 저장
model = FastLanguageModel.for_inference(model)
model.save_pretrained_merged(
    new_model_path,  # 같은 경로에 덮어쓰기
    tokenizer,
    save_method = "merged_16bit",
)