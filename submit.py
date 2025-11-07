import pandas as pd
import numpy as np
import torch
import os
import gc
import unsloth
from types import SimpleNamespace
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel
import time
from datetime import timedelta

from section1_preprocessing import preprocess_section1
from section24_preprocessing import preprocess_section24
from sectionall_inference import inference


# =============================================================================
# 1. ARGS 설정
# =============================================================================
args = SimpleNamespace(
    # 경로 설정
    base_model_path="/home/ec2-user/models/llama3.1-8b-instruct",
    lora_adapter_path="/home/ec2-user/forDPO/to_eunjin/llama8binst-ORPO-kor3-golden-omi",  # LoRA 어댑터 경로
    output_dir="./allresults/09_1015model_load2_batch2_revise_test",
    
    # 데이터 경로
    #data_dir="/home/ec2-user/data/sample",
    data_dir="/home/ec2-user/data",
    
    # Section별 설정
    section1_max_tokens=1000,
    section24_max_tokens=2000,
    section35_max_tokens=1000,
    
    # 공통 설정
    temperature=0.0,
    seed=777,

    # 섹션 별 배치 사이즈 설정
    section1_batch_size=4,
    section24_batch_size=2,
    section35_batch_size=4,
    
    # 진료과별 프롬프트 사용여부
    use_department_prompt=True

)


# =============================================================================
# 2. 데이터 전처리
# =============================================================================
def load_and_preprocess_data(args):
    """
    5개 데이터셋을 로드하고 section별로 전처리
    """
    print("=" * 80)
    print("데이터 로드 중...")
    print("=" * 80)
    
    # 데이터 로드
    admission = pd.read_csv(f"{args.data_dir}/admission.csv")
    admission = admission.iloc[-36:]
    nursing_note = pd.read_csv(f"{args.data_dir}/nursing_note.csv")
    chief_complaint = pd.read_csv(f"{args.data_dir}/chief_complaint.csv")
    medical_note = pd.read_csv(f"{args.data_dir}/medical_note.csv")
    surgery = pd.read_csv(f"{args.data_dir}/surgery_anesthesia.csv")
    
    print(f"✓ Admission: {len(admission)} rows")
    print(f"✓ Nursing Note: {len(nursing_note)} rows")
    print(f"✓ Chief Complaint: {len(chief_complaint)} rows")
    print(f"✓ Medical Note: {len(medical_note)} rows")
    print(f"✓ Surgery: {len(surgery)} rows")
    
    # 2-1. Section 1 전처리
    print("\n" + "=" * 80)
    print("Section 1 전처리 중...")
    print("=" * 80)
    section1_results = preprocess_section1(
        admission=admission,
        nursing_note=nursing_note,
        chief_complaint=chief_complaint,
        medical_note=medical_note
    )
    
    df_section1 = pd.DataFrame(section1_results)
    section1_dataset = Dataset.from_pandas(df_section1)
    print(f"✓ Section 1 데이터셋 생성 완료: {len(section1_dataset)} 환자")
    
    # 2-2. Section 2,4 전처리
    print("\n" + "=" * 80)
    print("Section 2,4 전처리 중...")
    print("=" * 80)
    section24_results = preprocess_section24(
        admission=admission,
        medical_note=medical_note,
        surgery=surgery
    )
    
    df_section24 = pd.DataFrame(section24_results)
    section24_dataset = Dataset.from_pandas(df_section24)
    print(f"✓ Section 2,4 데이터셋 생성 완료: {len(section24_dataset)} 환자")
    
    return section1_dataset, section24_dataset


# =============================================================================
# 3. 모델 초기화 (LoRA 포함)
# =============================================================================
def load_lora_model(args):
    """
    Base 모델 + LoRA 어댑터 로드
    """
    print("\n" + "=" * 80)
    print("LoRA 모델 로드 중...")
    print("=" * 80)
    
    # Base 모델 로드
    print("1. Base 모델 로드...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_path,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )
    
    # LoRA 어댑터 적용
    print("2. LoRA 어댑터 적용...")
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    FastLanguageModel.for_inference(model)
    
    print("✓ LoRA 모델 로드 완료!")
    return model, tokenizer


def unload_model(model, tokenizer):
    """
    모델 메모리 해제
    """
    print("\n모델 메모리 해제 중...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ 메모리 해제 완료")


# =============================================================================
# 4. Section 1 Inference
# =============================================================================
def run_section1_inference(section1_dataset, model, tokenizer, args):
    """
    Section 1 inference 실행
    """
    print("\n" + "=" * 80)
    print("Section 1 Inference 실행 중...")
    print("=" * 80)
    
    inference_args = SimpleNamespace(
        output_path=f"{args.output_dir}/section1",
        max_tokens=args.section1_max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.section1_batch_size
    )
    
    section1_result = inference(
        dataset_input=section1_dataset,
        model=model,
        tokenizer=tokenizer,
        args=inference_args,
        section_type="section1",
        text_column="final_text"
    )
    
    print("✓ Section 1 Inference 완료!")
    return section1_result



# =============================================================================
# 5. Section 2,4 Inference
# =============================================================================
def run_section24_inference(section24_dataset, model, tokenizer, args):
    """
    Section 2,4 inference 실행
    """
    inference_args = SimpleNamespace(
        output_path=f"{args.output_dir}/section24",
        max_tokens=args.section24_max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.section24_batch_size
    )
    
    section24_result = inference(
        dataset_input=section24_dataset,
        model=model,
        tokenizer=tokenizer,
        args=inference_args,
        section_type="section24",
        text_column="final_text"
    )
    
    return section24_result


# =============================================================================
# 6. Section 3,5 Inference
# =============================================================================
def run_section35_inference(section1_result, section24_result, args):
    """
    Section 3,5 inference 실행
    Section 1 + Section 2,4 결과를 합쳐서 요약
    """
    print("\n" + "=" * 80)
    print("Section 3,5 Inference 실행 중... (Base 모델)")
    print("=" * 80)
    
    # Base 모델 로드 (LoRA 없이)
    print("1. Base 모델 로드...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_path,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Section 1 + Section 2,4 결과 합치기
    print("2. Section 1 + Section 2,4 결과 병합...")
    combined_data = []
    
    for i in range(len(section1_result)):
        combined_item = {
            "patient_id": section1_result[i].get("patient_id", i),
            "section1_output": section1_result[i]["prediction"],
            "section24_output": section24_result[i]["prediction"],
            "combined_input": section1_result[i]["prediction"] + "\n\n" + section24_result[i]["prediction"]
        }
        combined_data.append(combined_item)
    
    df_combined = pd.DataFrame(combined_data)
    section35_dataset = Dataset.from_pandas(df_combined)
    
    # Section 3,5 inference 실행
    print("3. Section 3,5 Inference 실행...")
    inference_args = SimpleNamespace(
        output_path=f"{args.output_dir}/section35",
        max_tokens=args.section35_max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.section35_batch_size
    )
    
    section35_result = inference(
        dataset_input=section35_dataset,
        model=model,
        tokenizer=tokenizer,
        args=inference_args,
        section_type="section35",
        text_column="combined_input"
    )
    
    # 모델 메모리 해제
    unload_model(model, tokenizer)
    
    return section35_result






# =============================================================================
# 6.5 출력 결과 후처리 함수 (section 1,2,3,4,5 순서대로 나열)
# =============================================================================
def extract_section_parts(text, section_markers):
    """
    텍스트에서 특정 섹션을 추출하는 함수
    """
    import re
    
    result = {}
    
    for marker in section_markers:
        # ===== 수정: 각 글자 사이에 선택적 공백 추가 =====
        # 예: "입원경과" → "입\s*원\s*경\s*과"
        marker_chars = list(marker)
        marker_pattern = r'\s*'.join(re.escape(char) for char in marker_chars)
        # ================================================
        
        # 섹션 시작 찾기 (대소문자 구분 없이, 앞뒤 공백 허용)
        pattern = rf'(?i)^\s*[-\s]*{marker_pattern}\s*$'
        
        lines = text.split('\n')
        start_idx = None
        
        # 시작 지점 찾기
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                start_idx = i
                break
        
        if start_idx is None:
            result[marker] = ""
            continue
        
        # 다음 섹션 시작 지점 찾기
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            # ===== 수정: 다른 주요 섹션 패턴도 공백 선택적 처리 =====
            # 다른 주요 섹션이 시작되면 종료
            if re.match(r'^\s*[-\s]*(입\s*원\s*경\s*과|검\s*사\s*결\s*과|입\s*원\s*결\s*과|P\s*a\s*t\s*i\s*e\s*n\s*t\s*S\s*u\s*m\s*m\s*a\s*r\s*y)\s*$', line, re.IGNORECASE):
                end_idx = i
                break
            # ======================================================
        
        # 섹션 내용 추출 (섹션 제목 포함)
        section_content = '\n'.join(lines[start_idx:end_idx]).strip()
        result[marker] = section_content
    
    return result


def postprocess_predictions(sec1_pred, sec24_pred, sec35_pred):
    """
    각 섹션의 예측 결과를 후처리하여 원하는 순서로 재구성
    
    Args:
        sec1_pred: Section 1 예측 결과
        sec24_pred: Section 2,4 예측 결과
        sec35_pred: Section 3,5 예측 결과
    
    Returns:
        str: 재구성된 최종 출력
    """
    # Section 2,4에서 입원경과와 검사결과 추출
    sec24_parts = extract_section_parts(sec24_pred, ['입원경과', '검사결과'])
    
    # Section 3,5에서 입원결과와 Patient Summary 추출
    sec35_parts = extract_section_parts(sec35_pred, ['입원결과', 'Patient Summary'])
    
    # 원하는 순서대로 재구성
    final_parts = []
    
    # 1. 입원 사유 및 병력요약 (Section 1 전체)
    if sec1_pred.strip():
        final_parts.append(sec1_pred.strip())
    
    # 2. 입원경과 (Section 2,4에서 추출)
    if sec24_parts.get('입원경과', '').strip():
        final_parts.append(sec24_parts['입원경과'].strip())
    
    # 3. 입원결과 (Section 3,5에서 추출)
    if sec35_parts.get('입원결과', '').strip():
        final_parts.append(sec35_parts['입원결과'].strip())
    
    # 4. 검사결과 (Section 2,4에서 추출)
    if sec24_parts.get('검사결과', '').strip():
        final_parts.append(sec24_parts['검사결과'].strip())
    
    # 5. Patient Summary (Section 3,5에서 추출)
    if sec35_parts.get('Patient Summary', '').strip():
        final_parts.append(sec35_parts['Patient Summary'].strip())
    
    # 최종 결합 (각 파트 사이에 빈 줄 2개)
    final_output = '\n\n'.join(final_parts)
    
    return final_output




# =============================================================================
# 7. 최종 결과 통합 및 저장
# =============================================================================
def save_final_results(section1_result, section24_result, section35_result, args):
    """
    모든 section 결과를 통합하여 저장
    - result.csv: 최종 통합 결과
    - specific_result.csv: 각 section별 상세 결과
    """
    print("\n" + "=" * 80)
    print("최종 결과 저장 중...")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    final_results = []
    specific_results = []
    
    for i in range(len(section1_result)):
        patient_id = section1_result[i].get("patient_id", i)
        
        # Section별 결과
        sec1_pred = section1_result[i]["prediction"]
        sec24_pred = section24_result[i]["prediction"]
        sec35_pred = section35_result[i]["prediction"]
        
        # ===== 추가: 후처리를 통한 최종 결과 재구성 =====
        final_output = postprocess_predictions(sec1_pred, sec24_pred, sec35_pred)
        # =================================================
        
        # ===== 기존 코드 주석 처리 =====
        # final_output = f"{sec1_pred}\n\n{sec24_pred}\n\n{sec35_pred}"
        # ==============================
        
        final_results.append({
            "patient_id": patient_id,
            "final_output": final_output
        })
        
        # 상세 결과 (specific_result.csv용)
        specific_results.append({
            "patient_id": patient_id,
            # Section 1
            "section1_input_raw": section1_result[i].get("final_text", ""),
            "section1_input_with_prompt": section1_result[i].get("prompt", ""),
            "section1_output": sec1_pred,
            # Section 2,4
            "section24_input_raw": section24_result[i].get("final_text", ""),
            "section24_input_with_prompt": section24_result[i].get("prompt", ""),
            "section24_output": sec24_pred,
            # Section 3,5
            "section35_input_raw": section35_result[i].get("combined_input", ""),
            "section35_input_with_prompt": section35_result[i].get("prompt", ""),
            "section35_output": sec35_pred,
            # 최종 결과
            "final_output": final_output
        })
    
    # result.csv 저장
    df_final = pd.DataFrame(final_results)
    result_path = f"{args.output_dir}/result.csv"
    df_final.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"✓ 최종 결과 저장: {result_path}")
    
    # specific_result.csv 저장
    df_specific = pd.DataFrame(specific_results)
    specific_path = f"{args.output_dir}/specific_result.csv"
    df_specific.to_csv(specific_path, index=False, encoding='utf-8-sig')
    print(f"✓ 상세 결과 저장: {specific_path}")
    
    print("\n" + "=" * 80)
    print("모든 처리 완료!")
    print("=" * 80)
    print(f"총 처리 환자 수: {len(final_results)}")
    print(f"결과 저장 경로: {args.output_dir}")


# =============================================================================
# 8. Main Pipeline
# =============================================================================
def main():
    """
    전체 파이프라인 실행
    """
    # 전체 실행 시간 측정 시작 
    start_time = time.time()

    print("\n" + "=" * 80)
    print("Medical Discharge Summary Generation Pipeline 시작")
    print("=" * 80)
    

    step_times = {}
    
    # 1. 데이터 전처리
    step_start = time.time()
    
    section1_dataset, section24_dataset = load_and_preprocess_data(args)
    step_times['1_데이터_전처리'] = time.time() - step_start

    
    # 2. LoRA 모델 로드
    step_start = time.time()
    model, tokenizer = load_lora_model(args)
    step_times['2_LoRA_모델_로드'] = time.time() - step_start

    
    # 3. Section 1 Inference
    step_start = time.time()
    section1_result = run_section1_inference(section1_dataset, model, tokenizer, args)
    step_times['3_Section1_Inference'] = time.time() - step_start
    
    # 4. Section 2,4 Inference
    step_start = time.time()
    section24_result = run_section24_inference(section24_dataset, model, tokenizer, args)
    step_times['4_Section24_Inference'] = time.time() - step_start
    
    
    # 5. LoRA 모델 메모리 해제
    step_start = time.time()
    unload_model(model, tokenizer)
    step_times['5_LoRA_모델_메모리_해제'] = time.time() - step_start
    
    
    # 6. Section 3,5 Inference (Base 모델)
    step_start = time.time()
    section35_result = run_section35_inference(section1_result, section24_result, args)
    step_times['6_Section35_Inference'] = time.time() - step_start
    
    # 7. 최종 결과 저장
    step_start = time.time()
    save_final_results(section1_result, section24_result, section35_result, args)
    step_times['7_최종_결과_저장'] = time.time() - step_start

    
    # 전체 실행 시간 계산 
    total_time = time.time() - start_time
    
    print("\n✓ Pipeline 완료!")
    

    print("\n" + "=" * 80)
    print("실행 시간 통계")
    print("=" * 80)
    
    # 단계별 시간 출력
    for step_name, elapsed in step_times.items():
        print(f"{step_name}: {str(timedelta(seconds=int(elapsed)))}")
    
    print(f"\n총 실행 시간: {str(timedelta(seconds=int(total_time)))}")
    print("=" * 80)
    
    # runtime.txt 파일로 저장
    runtime_path = os.path.join(args.output_dir, "runtime.txt")
    with open(runtime_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Medical Discharge Summary Generation Pipeline - 실행 시간 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        # 단계별 시간
        f.write("[ 단계별 실행 시간 ]\n")
        f.write("-" * 80 + "\n")
        for step_name, elapsed in step_times.items():
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            elapsed_seconds = f"{elapsed:.2f}s"
            f.write(f"{step_name:<30} : {elapsed_str:>15} ({elapsed_seconds:>10})\n")
        
        f.write("-" * 80 + "\n")
        
        # 총 실행 시간
        total_str = str(timedelta(seconds=int(total_time)))
        total_seconds = f"{total_time:.2f}s"
        f.write(f"\n{'총 실행 시간':<30} : {total_str:>15} ({total_seconds:>10})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # 추가 정보
        f.write("\n[ 설정 정보 ]\n")
        f.write(f"- Base Model: {args.base_model_path}\n")
        f.write(f"- LoRA Adapter: {args.lora_adapter_path}\n")
        f.write(f"- Section 1 Max Tokens: {args.section1_max_tokens}\n")
        f.write(f"- Section 2,4 Max Tokens: {args.section24_max_tokens}\n")
        f.write(f"- Section 3,5 Max Tokens: {args.section35_max_tokens}\n")
        f.write(f"- Temperature: {args.temperature}\n")
        f.write(f"- Seed: {args.seed}\n")
    
    print(f"\n✓ 실행 시간 보고서 저장: {runtime_path}")


if __name__ == "__main__":
    main()