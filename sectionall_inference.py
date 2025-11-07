"""
Unified Inference Module for All Sections
Section 1, 2/4, 3/5 inference를 하나의 모듈로 통합
"""

from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
import random
import os
from types import SimpleNamespace
from tqdm import tqdm


from prompts import get_prompt


# =============================================================================
# Seed 고정
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# =============================================================================
# 데이터셋 프롬프트 적용
# =============================================================================
def apply_prompt(dataset_input, tokenizer, section_type, text_column="final_text", department=None):


    # 진료과별 프롬프트 가져오기
    prompt_template = get_prompt(section_type, department)
    
    def process(item):
        input_text = item.get(text_column, "")
        prompt_text = prompt_template.replace("<<input>>", input_text.strip())
        
        item["prompt"] = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful and precise medical summarization assistant."},
                {"role": "user", "content": prompt_text}
            ],
            add_generation_prompt=True,
            tokenize=False
        )
        return item
    
    dataset = dataset_input.map(process)
    return dataset


# =============================================================================
# Batch Inference
# =============================================================================
def run_batch_inference(dataset, model, tokenizer, max_tokens, temperature, seed, batch_size):
    """
    배치 단위로 inference 수행
    """
    set_seed(seed)
    
    predictions = []
    prompts = [item["prompt"] for item in dataset]
    
    print(f"총 {len(prompts)}개 샘플을 배치 크기 {batch_size}로 처리 중...")


    model_max_length = 30000
    max_input_length = model_max_length - max_tokens - 50  # 50은 안전 마진
    print(f"모델 최대 길이: {model_max_length}, 입력 최대 길이: {max_input_length}, 생성 최대 길이: {max_tokens}")
    
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        # left padding 설정
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'

        
        # 배치 토크나이즈
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=30000,
        ).to("cuda")

        # padding_side 원래대로 복원
        tokenizer.padding_side = original_padding_side

        actual_input_length = inputs['input_ids'].shape[1]
        if actual_input_length > max_input_length:
            print(f" Warning: 배치 {i//batch_size + 1}의 입력 길이 ({actual_input_length})가 예상보다 깁니다.")

        # 배치 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            

        for j, output in enumerate(outputs):
            full_text = tokenizer.decode(output, skip_special_tokens=True)
            original_prompt = batch_prompts[j]
            
            # 프롬프트 부분 제거
            # chat template이 적용된 경우, assistant 응답만 추출
            if "assistant" in full_text:
                # "assistant" 이후의 텍스트만 추출
                parts = full_text.split("assistant")
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
                else:
                    generated_text = full_text
            else:
                # assistant 태그가 없는 경우, 프롬프트 길이만큼 제거
                # 프롬프트를 디코딩하여 비교
                prompt_length = len(tokenizer.encode(original_prompt, add_special_tokens=False))
                generated_ids = output[prompt_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            predictions.append(generated_text)

        
        
        # 메모리 정리 (매 배치마다)
        del inputs, outputs
        torch.cuda.empty_cache()
    
    # 데이터셋에 prediction 추가
    def add_prediction(item, idx):
        item["prediction"] = predictions[idx]
        return item
    
    dataset = dataset.map(add_prediction, with_indices=True)
    return dataset


# =============================================================================
# 통합 Inference 함수
# =============================================================================
def inference(dataset_input, model, tokenizer, args, section_type, text_column="final_text", department=None):
    print(f"\n{'='*80}")
    print(f"{section_type.upper()} Inference 시작")
    if department:
        print(f"진료과: {department}")
    print(f"{'='*80}")
    
    dataset = apply_prompt(dataset_input, tokenizer, section_type, text_column, department)
    
    dataset = run_batch_inference(
        dataset, model, tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.batch_size
    )
    
    # 결과 저장 (기존 코드 동일)
    os.makedirs(args.output_path, exist_ok=True)
    dataset.save_to_disk(args.output_path)
    
    json_path = os.path.join(args.output_path, "predictions.json")
    dataset.to_json(json_path, orient="records", lines=True, force_ascii=False)
    
    csv_path = os.path.join(args.output_path, "predictions.csv")
    dataset.to_pandas().to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ Inference 완료! 결과 저장 경로: {args.output_path}")
    
    return dataset