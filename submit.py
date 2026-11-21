import pandas as pd
import torch
import os
import gc
import unsloth
from types import SimpleNamespace
from datasets import Dataset
from unsloth import FastLanguageModel
from peft import PeftModel
import time
from datetime import timedelta

from section1_preprocessing import preprocess_section1
from section24_preprocessing import preprocess_section24
from sectionall_inference import inference

args = SimpleNamespace(
    base_model_path=os.getenv("MARS_BASE_MODEL_PATH", "./models/llama3.1-8b-instruct"),
    lora_adapter_path=os.getenv("MARS_LORA_ADAPTER_PATH", "./models/section12-adapter"),
    section35_adapter_path=os.getenv("MARS_SECTION35_ADAPTER_PATH", "./models/section35-adapter"),
    output_dir=os.getenv("MARS_OUTPUT_DIR", "./results"),

    data_dir=os.getenv("MARS_DATA_DIR", "./data"),

    section1_max_tokens=1000,
    section24_max_tokens=2000,
    section35_max_tokens=1000,

    temperature=0.0,
    seed=777,

    section1_batch_size=4,
    section24_batch_size=1,
    section35_batch_size=4,

)

def load_and_preprocess_data(args):
    """
    5개 데이터셋을 로드하고 section별로 전처리
    """
    print("데이터 로드 중...")

    admission = pd.read_csv(f"{args.data_dir}/admission.csv")
    nursing_note = pd.read_csv(f"{args.data_dir}/nursing_note.csv")
    chief_complaint = pd.read_csv(f"{args.data_dir}/chief_complaint.csv")
    medical_note = pd.read_csv(f"{args.data_dir}/medical_note.csv")
    surgery = pd.read_csv(f"{args.data_dir}/surgery_anesthesia.csv")

    print(f"Admission: {len(admission)} rows")
    print(f"Nursing Note: {len(nursing_note)} rows")
    print(f"Chief Complaint: {len(chief_complaint)} rows")
    print(f"Medical Note: {len(medical_note)} rows")
    print(f"Surgery: {len(surgery)} rows")

    print("Section 1 전처리 중...")
    section1_results = preprocess_section1(
        admission=admission,
        nursing_note=nursing_note,
        chief_complaint=chief_complaint,
        medical_note=medical_note
    )

    df_section1 = pd.DataFrame(section1_results)
    section1_dataset = Dataset.from_pandas(df_section1)
    print(f"Section 1 데이터셋 생성 완료: {len(section1_dataset)} 환자")

    print("Section 2,4 전처리 중...")
    section24_results = preprocess_section24(
        admission=admission,
        medical_note=medical_note,
        surgery=surgery
    )

    df_section24 = pd.DataFrame(section24_results)
    section24_dataset = Dataset.from_pandas(df_section24)
    print(f"Section 2,4 데이터셋 생성 완료: {len(section24_dataset)} 환자")

    return section1_dataset, section24_dataset

def load_lora_model(args):
    """
    Base 모델 + LoRA 어댑터 로드
    """
    print("LoRA 모델 로드 중...")

    print("1. Base 모델 로드...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_path,
        max_seq_length=30000,
        dtype=None,
        load_in_4bit=True,
    )

    print("2. LoRA 어댑터 적용...")
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    FastLanguageModel.for_inference(model)

    print("LoRA 모델 로드 완료!")
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
    print("메모리 해제 완료")

def run_section1_inference(section1_dataset, model, tokenizer, args):
    """
    Section 1 inference 실행
    """
    print("Section 1 Inference 실행 중...")

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

    print("Section 1 Inference 완료!")
    return section1_result

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

def run_section35_inference(section1_result, section24_result, args):
    """
    Section 3,5 inference 실행
    Section 1 + Section 2,4 결과를 합쳐서 요약
    """
    print("Section 3,5 Inference 실행 중... (Base 모델)")

    print("1. Base 모델 로드...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_path,
        max_seq_length=30000,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.section35_adapter_path)
    FastLanguageModel.for_inference(model)

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

    unload_model(model, tokenizer)

    return section35_result

def extract_section_parts(text, section_markers):
    """
    텍스트에서 특정 섹션을 추출하는 함수
    Returns:
        dict: {section_name: section_content}
    """
    import re

    result = {}

    for marker in section_markers:
        marker_chars = list(marker)
        marker_pattern = r'\s*'.join(re.escape(char) for char in marker_chars)
        pattern = rf'(?i)^\s*[-\s]*{marker_pattern}\s*$'

        lines = text.split('\n')
        start_idx = None

        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                start_idx = i
                break

        if start_idx is None:
            result[marker] = ""
            continue

        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if re.match(r'^\s*[-\s]*(입\s*원\s*경\s*과|검\s*사\s*결\s*과|입\s*원\s*결\s*과|P\s*a\s*t\s*i\s*e\s*n\s*t\s*S\s*u\s*m\s*m\s*a\s*r\s*y)\s*$', line, re.IGNORECASE):
                end_idx = i
                break

        section_content = '\n'.join(lines[start_idx:end_idx]).strip()
        result[marker] = section_content

    return result

def postprocess_predictions(sec1_pred, sec24_pred, sec35_pred):
    """
    각 섹션의 예측 결과를 후처리하여 원하는 순서로 재구성

    Returns:
        str: 재구성된 최종 출력
    """
    sec24_parts = extract_section_parts(sec24_pred, ['입원경과', '검사결과'])

    sec35_parts = extract_section_parts(sec35_pred, ['입원결과', 'Patient Summary'])

    final_parts = []

    if sec1_pred.strip():
        final_parts.append(sec1_pred.strip())

    if sec24_parts.get('입원경과', '').strip():
        final_parts.append(sec24_parts['입원경과'].strip())

    if sec35_parts.get('입원결과', '').strip():
        final_parts.append(sec35_parts['입원결과'].strip())

    if sec24_parts.get('검사결과', '').strip():
        final_parts.append(sec24_parts['검사결과'].strip())

    if sec35_parts.get('Patient Summary', '').strip():
        final_parts.append(sec35_parts['Patient Summary'].strip())

    final_output = '\n\n'.join(final_parts)

    return final_output

def save_final_results(section1_result, section24_result, section35_result, args):
    """
    모든 section 결과를 통합하여 저장
    - result.csv: 최종 통합 결과
    - specific_result.csv: 각 section별 상세 결과
    """
    print("최종 결과 저장 중...")

    os.makedirs(args.output_dir, exist_ok=True)

    final_results = []
    specific_results = []

    for i in range(len(section1_result)):
        patient_id = section1_result[i].get("patient_id", i)

        sec1_pred = section1_result[i]["prediction"]
        sec24_pred = section24_result[i]["prediction"]
        sec35_pred = section35_result[i]["prediction"]

        final_output = postprocess_predictions(sec1_pred, sec24_pred, sec35_pred)

        final_results.append({
            "patient_id": patient_id,
            "final_output": final_output
        })

        specific_results.append({
            "patient_id": patient_id,
            "section1_input_raw": section1_result[i].get("final_text", ""),
            "section1_input_with_prompt": section1_result[i].get("prompt", ""),
            "section1_output": sec1_pred,
            "section24_input_raw": section24_result[i].get("final_text", ""),
            "section24_input_with_prompt": section24_result[i].get("prompt", ""),
            "section24_output": sec24_pred,
            "section35_input_raw": section35_result[i].get("combined_input", ""),
            "section35_input_with_prompt": section35_result[i].get("prompt", ""),
            "section35_output": sec35_pred,
            "final_output": final_output
        })

    df_final = pd.DataFrame(final_results)
    result_path = f"{args.output_dir}/result.csv"
    df_final.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"최종 결과 저장: {result_path}")

    df_specific = pd.DataFrame(specific_results)
    specific_path = f"{args.output_dir}/specific_result.csv"
    df_specific.to_csv(specific_path, index=False, encoding='utf-8-sig')
    print(f"상세 결과 저장: {specific_path}")

    print("모든 처리 완료!")
    print(f"총 처리 환자 수: {len(final_results)}")
    print(f"결과 저장 경로: {args.output_dir}")

def main():
    """
    전체 파이프라인 실행
    """
    start_time = time.time()

    print("Medical Discharge Summary Generation Pipeline 시작")

    step_times = {}

    step_start = time.time()
    section1_dataset, section24_dataset = load_and_preprocess_data(args)
    step_times['1_데이터_전처리'] = time.time() - step_start

    step_start = time.time()
    model, tokenizer = load_lora_model(args)
    step_times['2_LoRA_모델_로드'] = time.time() - step_start

    step_start = time.time()
    section1_result = run_section1_inference(section1_dataset, model, tokenizer, args)
    step_times['3_Section1_Inference'] = time.time() - step_start

    step_start = time.time()
    section24_result = run_section24_inference(section24_dataset, model, tokenizer, args)
    step_times['4_Section24_Inference'] = time.time() - step_start

    step_start = time.time()
    unload_model(model, tokenizer)
    step_times['5_LoRA_모델_메모리_해제'] = time.time() - step_start

    step_start = time.time()
    section35_result = run_section35_inference(section1_result, section24_result, args)
    step_times['6_Section35_Inference'] = time.time() - step_start

    step_start = time.time()
    save_final_results(section1_result, section24_result, section35_result, args)
    step_times['7_최종_결과_저장'] = time.time() - step_start

    total_time = time.time() - start_time

    print("\nPipeline 완료!")

    print("실행 시간 통계")

    for step_name, elapsed in step_times.items():
        print(f"{step_name}: {str(timedelta(seconds=int(elapsed)))}")

    print(f"\n총 실행 시간: {str(timedelta(seconds=int(total_time)))}")

    runtime_path = os.path.join(args.output_dir, "runtime.txt")
    with open(runtime_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Medical Discharge Summary Generation Pipeline - 실행 시간 보고서\n")
        f.write("=" * 80 + "\n\n")

        f.write("[ 단계별 실행 시간 ]\n")
        f.write("-" * 80 + "\n")
        for step_name, elapsed in step_times.items():
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            elapsed_seconds = f"{elapsed:.2f}s"
            f.write(f"{step_name:<30} : {elapsed_str:>15} ({elapsed_seconds:>10})\n")

        f.write("-" * 80 + "\n")

        total_str = str(timedelta(seconds=int(total_time)))
        total_seconds = f"{total_time:.2f}s"
        f.write(f"\n{'총 실행 시간':<30} : {total_str:>15} ({total_seconds:>10})\n")

        f.write("\n" + "=" * 80 + "\n")

        f.write("\n[ 설정 정보 ]\n")
        f.write(f"- Base Model: {args.base_model_path}\n")
        f.write(f"- LoRA Adapter: {args.lora_adapter_path}\n")
        f.write(f"- Section 1 Max Tokens: {args.section1_max_tokens}\n")
        f.write(f"- Section 2,4 Max Tokens: {args.section24_max_tokens}\n")
        f.write(f"- Section 3,5 Max Tokens: {args.section35_max_tokens}\n")
        f.write(f"- Temperature: {args.temperature}\n")
        f.write(f"- Seed: {args.seed}\n")

    print(f"\n실행 시간 보고서 저장: {runtime_path}")

    bucket_name = os.getenv("MARS_S3_BUCKET")
    if bucket_name:
        import boto3

        s3_prefix = os.getenv("MARS_S3_PREFIX", "submissions/")
        s3 = boto3.client("s3")
        for filename in ("result.csv", "runtime.txt"):
            file_path = os.path.join(args.output_dir, filename)
            if os.path.isfile(file_path):
                s3.upload_file(file_path, bucket_name, f"{s3_prefix}{filename}")
                print(f"Uploaded {filename} to s3://{bucket_name}/{s3_prefix}{filename}")
            else:
                print(f"Skipping missing file: {file_path}")

if __name__ == "__main__":
    main()
