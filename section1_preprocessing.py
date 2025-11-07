"""
Section 1 Preprocessing Module
입원경과 및 검사결과 섹션을 위한 데이터 전처리
"""

import pandas as pd
import re
from tqdm import tqdm


# =============================================================================
# 유틸리티 함수
# =============================================================================
def clean_item_and_value(item_name, item_value):
    """
    서식 항목명을 정제하고, 항목값에 따라 'Yes'/'No'로 치환하거나 내용을 그대로 반환합니다.
    """
    parts = [p.strip() for p in item_name.split(' <- ')]
    clean_name = parts[0]
    
    if pd.isna(item_value) or str(item_value).strip() == '':
        return clean_name, 'No Information'
    
    value_str = str(item_value).strip()
    
    if value_str == '1' or value_str.lower() == 'yes':
        final_value = 'Yes'
    elif value_str == '0' or value_str.lower() == 'no':
        final_value = 'No'
    else:
        final_value = value_str.replace('\r\r\n', ' ').replace('\n', ' ')
    
    return clean_name, final_value


def map_to_category(item_name):
    """
    서식 항목명을 원하는 6가지 범주 중 하나로 매핑합니다.
    """
    if 'Present illness' in item_name:
        return 'A_Present Illness'
    elif 'Past history' in item_name or '과거병력' in item_name:
        return 'B_Past History'
    elif 'General condition' in item_name or 'ROS' in item_name:
        return 'C_Review of System'
    elif 'PE' in item_name or 'Physical' in item_name:
        return 'D_Physical Examination'
    elif 'Ass' in item_name:
        return 'E_Assessment'
    elif 'Plan' in item_name:
        return 'F_Plan'
    else:
        return 'Z_Others'


def merge_field_values_keep_structure(df, field_col="서식 항목명", value_col="항목별 서식 값"):
    """
    동일한 서식 항목명의 '항목별 서식 값'을 \n으로 연결하여,
    원본 행 구조는 그대로 유지하면서 병합된 값을 각 행에 반영합니다.
    """
    # 항목별 서식 값 병합 (NaN 제외)
    merged_map = (
        df.groupby(field_col)[value_col]
        .apply(lambda x: "\n".join(map(str, x.dropna().unique())))
        .to_dict()
    )
    # 원본 구조 유지하며 병합값을 각 행에 매핑
    df[value_col] = df[field_col].map(merged_map)
    return df


def extract_medications(text: str):
    """
    텍스트에서 약물명을 추출합니다.
    """
    # 예외 처리: 입력이 없거나 비어있을 경우
    if not text or not text.strip():
        return ""
    
    # 약 이름 추출 - 약 제형으로 find
    pattern = r'([A-Z][A-Za-z0-9\-\. ]+(?=\s*(?:tab|inj|bag|sus|vial|cap|solution|peri|premix)))'
    drugs = re.findall(pattern, text)
    
    # 숫자, 단위 제거
    cleaned = [
        re.sub(r'\b\d+(?:mg|ml|g|/2ml|/100ml)?\b', '', d)  # 용량 단위 제거
        for d in drugs
    ]
    cleaned = [re.sub(r'\s+', ' ', d).strip() for d in cleaned]  # 공백 정리
    
    # 중복 제거 및 정렬
    unique_drugs = sorted(set(cleaned))
    
    # '--'로 이어진 약 분리 후 다시 정리
    split_drugs = []
    for d in unique_drugs:
        parts = [p.strip() for p in d.split('--')]
        split_drugs.extend(parts)
    
    # 중복 제거 및 최종 정렬
    final_drugs = sorted(set(split_drugs))
    
    return ", ".join(final_drugs)


# =============================================================================
# 환자별 데이터 처리
# =============================================================================
def process_patient(p_num, admission_df, nursing_note_df, chief_complaint_df, medical_note_df):
    """
    단일 환자의 데이터를 처리하여 final_text를 생성합니다.
    """
    # ----------------------------------------
    # admission.csv 처리 - 입원/퇴원 날짜 추출
    # ----------------------------------------
    row = admission_df[admission_df["환자번호"] == p_num]
    
    if row.empty:
        return None  # 해당 환자 데이터가 없으면 None 반환
    
    hospt_d = pd.to_datetime(row["입원일자"].iloc[0]).date()
    disch_d = pd.to_datetime(row["퇴원일자"].iloc[0]).date()

    # 입원시 진료과 정보 추출
    department = row["입원시 진료과"].iloc[0] if "입원시 진료과" in row.columns else None
    
    cutoff_date_last = disch_d
    cutoff_date_first = hospt_d - pd.Timedelta(days=1)
    
    # ----------------------------------------
    # 각 데이터프레임 필터링
    # ----------------------------------------
    admission_filtered = admission_df[admission_df["환자번호"] == p_num]
    
    nursing_note_filtered = nursing_note_df[nursing_note_df["환자번호"] == p_num].copy()
    nursing_note_filtered["수진일자"] = pd.to_datetime(nursing_note_filtered["수진일자"], errors="coerce").dt.date
    nursing_note_filtered["간호기록 작성일자"] = pd.to_datetime(nursing_note_filtered["간호기록 작성일자"], errors="coerce").dt.date
    nursing_note_filtered = nursing_note_filtered[
        (nursing_note_filtered["수진일자"] <= cutoff_date_last) & 
        (nursing_note_filtered["간호기록 작성일자"] <= cutoff_date_last) & 
        (nursing_note_filtered["수진일자"] >= cutoff_date_first) & 
        (nursing_note_filtered["간호기록 작성일자"] >= cutoff_date_first)
    ]
    
    chief_complaint_filtered = chief_complaint_df[chief_complaint_df["환자번호"] == p_num].copy()
    chief_complaint_filtered["수진일자"] = pd.to_datetime(chief_complaint_filtered["수진일자"], errors="coerce").dt.date
    chief_complaint_filtered["작성일자"] = pd.to_datetime(chief_complaint_filtered["작성일자"], errors="coerce").dt.date
    chief_complaint_filtered = chief_complaint_filtered[
        (chief_complaint_filtered["수진일자"] <= cutoff_date_last) & 
        (chief_complaint_filtered["작성일자"] <= cutoff_date_last) & 
        (chief_complaint_filtered["수진일자"] >= cutoff_date_first) & 
        (chief_complaint_filtered["작성일자"] >= cutoff_date_first)
    ]
    
    medical_note_filtered = medical_note_df[medical_note_df["환자번호"] == p_num].copy()
    medical_note_filtered["수진일자"] = pd.to_datetime(medical_note_filtered["수진일자"], errors="coerce").dt.date
    medical_note_filtered["서식 작성일자"] = pd.to_datetime(medical_note_filtered["서식 작성일자"], errors="coerce").dt.date
    medical_note_filtered = medical_note_filtered[
        (medical_note_filtered["수진일자"] <= cutoff_date_last) & 
        (medical_note_filtered["서식 작성일자"] <= cutoff_date_last) & 
        (medical_note_filtered["수진일자"] >= cutoff_date_first) & 
        (medical_note_filtered["서식 작성일자"] >= cutoff_date_first)
    ]
    medical_note_filtered = medical_note_filtered[~medical_note_filtered["서식명"].str.contains("퇴원", na=False)]
    
    # ----------------------------------------
    # CC 처리
    # ----------------------------------------
    if not chief_complaint_filtered.empty:
        chief_complaint_list = chief_complaint_filtered["원내 CC명"].astype(str).str.strip().tolist()
    else:
        chief_complaint_list = []
    
    cc_filtered_df = nursing_note_filtered[nursing_note_filtered["항목명"].str.contains('내원 주증상', na=False)]
    nursing_note_cc_list = cc_filtered_df["항목값"].dropna().tolist()
    nursing_note_cc_list = [item for item in nursing_note_cc_list if str(item).strip() != '']
    
    combined_list = chief_complaint_list + nursing_note_cc_list
    cc_list = list(dict.fromkeys(combined_list))
    chief_complaint_concat = ", ".join(cc_list)
    
    # ----------------------------------------
    # nursing_note 처리 - Present Illness
    # ----------------------------------------
    target_detail = "입원(내원)동기 및 주증상 > 상세내용"
    detail_data = nursing_note_filtered[nursing_note_filtered["항목명"] == target_detail]
    
    detail_list = detail_data["항목값"].dropna().astype(str).tolist()
    detail_list_cleaned = [item.strip() for item in detail_list if item is not None and item.strip() != '']
    
    if detail_list_cleaned:
        present_illness_concat = "\n".join(detail_list_cleaned)
    else:
        target_status = "내원동기 및 현상태"
        status_data = nursing_note_filtered[nursing_note_filtered["항목명"] == target_status]
        status_list = status_data["항목값"].dropna().astype(str).tolist()
        status_list_cleaned = [item.strip() for item in status_list if item is not None and item.strip() != '']
        present_illness_concat = "\n".join(status_list_cleaned)
    
    # ----------------------------------------
    # medical_note 처리 - 초진만 추출
    # ----------------------------------------
    admin_notes = medical_note_filtered[medical_note_filtered["서식명"].str.contains("초진", na=False)].copy()
    
    if admin_notes.empty:
        final_blocks = {}
    else:
        admin_notes['Sort_Category'] = admin_notes['서식 항목명'].apply(map_to_category)
        admin_notes_sorted = admin_notes.sort_values(
            by='Sort_Category', 
            kind='stable'
        ).reset_index(drop=True)
        
        # Plan 제거
        admin_notes_sorted = admin_notes_sorted[admin_notes_sorted["Sort_Category"] != "F_Plan"]
        
        # 동일한 항목명의 값 병합 및 중복 제거
        admin_notes_sorted = merge_field_values_keep_structure(admin_notes_sorted).drop_duplicates()
        
        # medication 정리 (약물명 추출)
        medication_col = admin_notes_sorted["서식 항목명"].str.contains("medications", case=False, na=False)
        admin_notes_sorted.loc[medication_col, "항목별 서식 값"] = (
            admin_notes_sorted.loc[medication_col, "항목별 서식 값"]
            .apply(extract_medications)
        )
        
        # Sort_Category별 그룹화 및 텍스트 생성
        final_blocks = {}
        grouped = admin_notes_sorted.groupby('Sort_Category', sort=False)
        
        for category, group_df in grouped:
            category_lines = []
            
            for _, row in group_df.iterrows():
                if (category == 'A_Present Illness') | (category == 'E_Assessment'):
                    item_value = row['항목별 서식 값']
                    if not pd.isna(item_value) and str(item_value).strip() != '':
                        line = str(item_value).strip().replace('\r\r\n', ' ').replace('\n', ' ')
                    else:
                        continue
                else:
                    clean_name, final_value = clean_item_and_value(
                        row['서식 항목명'], 
                        row['항목별 서식 값']
                    )
                    line = f"{clean_name}: {final_value}"
                
                category_lines.append(line)
            
            unique_lines = list(dict.fromkeys(category_lines))
            
            if unique_lines:
                final_blocks[category] = "\n".join(unique_lines)
    
    # ----------------------------------------
    # 최종 텍스트 생성
    # ----------------------------------------
    final_output = []
    
    if chief_complaint_concat:
        final_output.append(f"<chief_complaint>\n{chief_complaint_concat}\n</chief_complaint>")
    
    if present_illness_concat:
        final_output.append(f"<present_illness_in_nursing_note>\n{present_illness_concat}\n</present_illness_in_nursing_note>")
    
    if not admin_notes.empty:
        for category_code in admin_notes_sorted['Sort_Category'].unique():
            header = category_code.split('_')[1]
            if category_code in final_blocks:
                content = final_blocks[category_code].strip()
                xml_tag = header.lower().replace(' ', '_')
                final_output.append(f"<{xml_tag}>\n{content}\n</{xml_tag}>")
    
    final_text = "\n\n".join(final_output)
    
    return final_text, department


# =============================================================================
# 메인 전처리 함수
# =============================================================================
def preprocess_section1(admission, nursing_note, chief_complaint, medical_note):
    """
    Section 1을 위한 전체 환자 데이터 전처리
    """
    print("Section 1 전처리 시작...")
    
    # 고유한 환자 번호 리스트 가져오기
    unique_patients = admission["환자번호"].unique()
    print(f"총 {len(unique_patients)}명의 환자 데이터 처리 중...")
    
    # 결과를 저장할 리스트
    results = []
    
    # 각 환자에 대해 처리
    for p_num in tqdm(unique_patients, desc="환자 처리"):
        result = process_patient(
            p_num, 
            admission, 
            nursing_note, 
            chief_complaint, 
            medical_note
        )
        
        if result is not None:
            final_text, department = result  # 튜플 언팩
            results.append({
                "patient_id": p_num,
                "final_text": final_text,
                "입원시_진료과": department  # 추가: 진료과 정보
            })
    
    print(f"✓ Section 1 전처리 완료! 총 {len(results)}명의 환자 데이터가 생성되었습니다.")
    
    return results