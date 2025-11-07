"""
Section 2,4 Preprocessing Module
수술/시술/마취기록 및 경과기록 섹션을 위한 데이터 전처리
"""

import pandas as pd
from tqdm import tqdm


# =============================================================================
# 수술 노트 처리
# =============================================================================
def process_operation_notes(medical_note):
    """
    [수술] 관련 노트를 처리하여 정제된 텍스트를 반환합니다.
    """
    op_notes = medical_note[medical_note["서식명"].str.contains("수술", na=False)].copy()
    
    # 우선순위에 따른 fallback 처리
    if op_notes.empty:
        fallback_keywords = ["24시간이내", "수술실퇴실전기록", "수술전기록"]
        
        for keyword in fallback_keywords:
            op_notes = medical_note[medical_note["서식명"].str.contains(keyword, na=False)].copy()
            if not op_notes.empty:
                break
        
        if op_notes.empty:
            return "<operation_note>\n</operation_note>"
    
    def clean_op_item(item_name, item_value):
        """서식 항목명을 '1순위 - 0순위' 형식으로 정제하고, 항목값과 연결된 문자열을 반환합니다."""
        parts = [p.strip() for p in item_name.split(' <- ')]
        
        if len(parts) > 1 and parts[0] != parts[1]:
            if '병원공통' in parts[1]:
                clean_name = parts[0]
            else:
                clean_name = f"{parts[1]} - {parts[0]}"
        else:
            clean_name = parts[0]
        
        if pd.isna(item_value) or str(item_value).strip() == '':
            final_value = 'No Information'
        else:
            value_str = str(item_value).strip()
            final_value = value_str.replace('\r\r\n', ' ').replace('\n', ' ')
        
        return f"{clean_name}: {final_value}"
    
    # 카테고리 매핑
    category_order = {
        '수술일시': 'A_수술일시',
        '마취종류': 'B_마취종류',
        '수술 소견': 'C_수술 소견',
        '수술전진단명': 'D_수술전진단명',
        '수술후진단명': 'E_수술후진단명',
        '수술명': 'F_수술명',
        '수술 절차': 'G_수술 절차'
    }
    
    def map_op_category(item_name):
        """서식 항목명을 기반으로 정렬 카테고리를 결정합니다."""
        key = item_name.split(' <- ')[0].strip()
        
        for category_name, sort_code in category_order.items():
            if category_name in key or key in category_name:
                return sort_code
        
        if '수술명' in key:
            return category_order['수술명']
        if '진단명' in key:
            if '수술전' in key:
                return category_order['수술전진단명']
            elif '수술후' in key:
                return category_order['수술후진단명']
        
        return 'Z_기타'
    
    # 카테고리 매핑 및 정렬
    op_notes['Sort_Category'] = op_notes['서식 항목명'].apply(map_op_category)
    op_notes_sorted = op_notes.sort_values(
        by=['Sort_Category', '서식 항목명'], 
        ascending=[True, True],
        kind='stable'
    ).reset_index(drop=True)
    
    # 카테고리별 그룹화 및 텍스트 생성
    final_blocks = {}
    grouped_by_category = op_notes_sorted.groupby('Sort_Category', sort=False)
    
    for category_code, group_df in grouped_by_category:
        category_lines = []
        
        for _, row in group_df.iterrows():
            line = clean_op_item(
                row['서식 항목명'], 
                row['항목별 서식 값']
            )
            category_lines.append(line)
        
        unique_lines = list(dict.fromkeys(category_lines))
        final_blocks[category_code] = "\n".join(unique_lines)
    
    # 최종 출력 생성
    final_output = []
    final_output.append('<operation_note>')
    
    for category_code in op_notes_sorted['Sort_Category'].unique():
        header = category_code.split('_')[1]
        
        if category_code in final_blocks and final_blocks[category_code].strip():
            content = final_blocks[category_code].strip()
            xml_tag = header.lower().replace(' ', '_')
            final_output.append(f'  <{xml_tag}>')
            final_output.append(f'    {content}')
            final_output.append(f'  </{xml_tag}>')
    
    final_output.append('</operation_note>')
    op_notes_final_text = "\n".join(final_output)
    
    return op_notes_final_text


# =============================================================================
# 타과 협진 노트 처리
# =============================================================================
def process_consult_notes(medical_note):
    """
    [타과] 노트를 처리하여 정제된 텍스트를 반환합니다.
    """
    # 수술 제외
    exclude_pattern = "수술"
    medical_note_no_op = medical_note[~medical_note["서식명"].str.contains(exclude_pattern, na=False)]
    
    # 타과 포함
    consult = medical_note_no_op[medical_note_no_op["서식명"].str.contains("타과", na=False)].copy()
    
    if consult.empty:
        return ""
    
    # 진단 제외
    consult_no_diagnosis = consult[~consult["서식 항목명"].str.contains("진단", na=False)].copy()
    
    # 날짜별로 그룹화
    consult_no_diagnosis['서식 작성일자'] = pd.to_datetime(consult_no_diagnosis['서식 작성일자']).dt.date
    consult_no_diagnosis = consult_no_diagnosis.sort_values('서식 작성일자')
    
    # 날짜별로 데이터 처리
    date_groups = {}
    
    for date, group in consult_no_diagnosis.groupby('서식 작성일자'):
        consult_content = {
            "수신진료과": [],
            "의뢰내용": [],
            "회신내용": []
        }
        
        for _, row in group.iterrows():
            item_name = row["서식 항목명"]
            item_value = row["항목별 서식 값"]
            
            if pd.isna(item_value) or str(item_value).strip() == '':
                clean_value = 'No Information'
                if "수신진료과" in item_name:
                    continue
            else:
                clean_value = str(item_value).strip().replace('\r\r\n', ' ').replace('\n', ' ')
            
            if "수신진료과" in item_name:
                department_name = clean_value.split()[-1]
                consult_content["수신진료과"].append(department_name)
            elif "의뢰내용" in item_name:
                consult_content["의뢰내용"].append(clean_value)
            elif "회신내용" in item_name:
                consult_content["회신내용"].append(clean_value)
        
        date_groups[date] = consult_content
    
    # 최종 출력 생성
    all_notes = []
    
    for date in sorted(date_groups.keys()):
        consult_content = date_groups[date]
        date_output = []
        
        if consult_content["수신진료과"]:
            unique_departments = list(dict.fromkeys(consult_content["수신진료과"]))
            departments_text = ", ".join(unique_departments)
            date_output.append(f"<수신 진료과>\n{departments_text}\n</수신 진료과>")
        
        if consult_content["의뢰내용"]:
            unique_requests = list(dict.fromkeys(consult_content["의뢰내용"]))
            request_lines = "\n".join([f"- {req}" for req in unique_requests])
            date_output.append(f"<의뢰내용>\n{request_lines}\n</의뢰내용>")
        
        if consult_content["회신내용"]:
            unique_responses = list(dict.fromkeys(consult_content["회신내용"]))
            response_lines = "\n".join([f"- {res}" for res in unique_responses])
            date_output.append(f"<회신내용>\n{response_lines}\n</회신내용>")
        
        if date_output:
            date_content = "\n\n".join(date_output)
            note_with_date = f'<consult_note date="{date}">\n{date_content}\n</consult_note>'
            all_notes.append(note_with_date)
    
    consult_notes_concat = "\n\n".join(all_notes)
    
    return consult_notes_concat


# =============================================================================
# 경과 기록 노트 처리
# =============================================================================
def process_progress_notes(medical_note):
    """
    경과 기록 노트를 처리하여 정제된 텍스트를 반환합니다.
    """
    # 수술/타과/초진/퇴원 제외
    exclude_pattern = "수술|타과|초진|퇴원"
    progress_notes = medical_note[~medical_note["서식명"].str.contains(exclude_pattern, na=False)].copy()
    
    if progress_notes.empty:
        return ""
    
    # 서식 항목명 필터링
    exclude_pattern = "시술 후 24시간|동의 받은|환자와의 관계|통화 대상|환자 성명|사유|추후 병원 방문시|동의서의 모든 내"
    progress_notes = progress_notes[
        ~progress_notes["서식 항목명"].astype(str).str.contains(exclude_pattern, na=False)
    ].copy()
    
    def map_progress_category(item_name):
        """Progress Note 항목명을 카테고리로 매핑하여 정렬 우선순위 지정"""
        key = item_name.split(' <- ')[0].strip()
        
        if key in ["소견", "Symptoms", "Objective findings"]:
            return 'A_SO_소견'
        elif key in ["Assessment & Plan", "A/P", "plan", "problem", "Assessment"]:
            return 'B_AP_평가 및 계획'
        else:
            return 'Z_기타'
    
    # 날짜 및 카테고리 매핑
    progress_notes['Sort_DateTime'] = pd.to_datetime(progress_notes['서식 작성일자'], errors='coerce')
    progress_notes['Sort_Category'] = progress_notes['서식 항목명'].apply(map_progress_category)
    
    # 정렬
    progress_notes_sorted = progress_notes.sort_values(
        by=['Sort_DateTime', 'Sort_Category', '서식 항목명'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # 날짜별 그룹화 후 텍스트 생성
    final_progress_text = []
    grouped_by_date = progress_notes_sorted.groupby(
        progress_notes_sorted['Sort_DateTime'].dt.date, sort=True
    )
    
    for date, group_df in grouped_by_date:
        final_progress_text.append(f'<progress_note date="{date.strftime("%Y-%m-%d")}">')
        
        categories = {}
        for _, row in group_df.iterrows():
            category = row['서식 항목명']
            value = row['항목별 서식 값']
            
            # 카테고리명에서 [] 제거
            if pd.notna(category):
                category_clean = category.strip('[]')
            else:
                category_clean = 'content'
            
            if category_clean not in categories:
                categories[category_clean] = []
            
            if pd.notna(value):
                categories[category_clean].append(value.strip())
        
        # 각 카테고리별로 XML 태그 생성
        for category, values in categories.items():
            # 중복 제거
            unique_values = list(dict.fromkeys(values))
            content_text = "\n      ".join(unique_values)
            
            # XML 태그명 생성 (공백을 _로 변환)
            xml_tag = category.lower().replace(' ', '_').replace('&', 'and')
            final_progress_text.append(f'    <{xml_tag}>')
            final_progress_text.append(f'      {content_text}')
            final_progress_text.append(f'    </{xml_tag}>')
        
        # note 종료
        final_progress_text.append('  </progress_note>')
    
    progress_notes_concat = "\n".join(final_progress_text)
    progress_notes_concat = "<progress_note>\n" + progress_notes_concat + "\n</progress_note>"
    
    return progress_notes_concat


# =============================================================================
# 환자별 데이터 처리
# =============================================================================
def process_patient(p_num, admission_df, medical_note_df, surgery_df):
    """
    단일 환자의 의료 기록을 전처리하여 최종 텍스트를 생성합니다.
    """
    try:
        # ----------------------------------------
        # admission.csv 처리 - 입원/퇴원 날짜 추출
        # ----------------------------------------
        row = admission_df[admission_df["환자번호"] == p_num]
        
        if row.empty:
            return None
        
        # 입원 일자
        hospt_d = pd.to_datetime(row["입원일자"].iloc[0])
        # 퇴원 일자
        disch_d = pd.to_datetime(row["퇴원일자"].iloc[0])
        
        # 입원시 진료과 
        department = row["입원시 진료과"].iloc[0] if "입원시 진료과" in row.columns else None
        
        cutoff_date_last = disch_d.date() if hasattr(disch_d, "date") else disch_d
        cutoff_date_first = hospt_d.date() if hasattr(hospt_d, "date") else hospt_d
        cutoff_date_first = cutoff_date_first - pd.Timedelta(days=1)
        
        # ----------------------------------------
        # 날짜 처리 - 입원전날 ~ 퇴원날짜만 추출
        # ----------------------------------------
        
        # medical_note 필터링
        medical_note = medical_note_df[medical_note_df["환자번호"] == p_num].copy()
        medical_note["수진일자"] = pd.to_datetime(medical_note["수진일자"], errors="coerce").dt.date
        medical_note["서식 작성일자"] = pd.to_datetime(medical_note["서식 작성일자"], errors="coerce").dt.date
        medical_note = medical_note[
            (medical_note["수진일자"] <= cutoff_date_last) & 
            (medical_note["서식 작성일자"] <= cutoff_date_last) & 
            (medical_note["수진일자"] >= cutoff_date_first) & 
            (medical_note["서식 작성일자"] >= cutoff_date_first)
        ]
        
        # 퇴원/초진 문서 제외
        medical_note = medical_note[~medical_note["서식명"].str.contains("퇴원", na=False)]
        medical_note = medical_note[~medical_note["서식명"].str.contains("초진", na=False)]
        
        # surgery 필터링
        surgery = surgery_df[surgery_df["환자번호"] == p_num].copy()
        surgery["수술 일자"] = pd.to_datetime(surgery["수술 일자"], errors="coerce").dt.date
        surgery = surgery[
            (surgery["수술 일자"] <= cutoff_date_last) & 
            (surgery["수술 일자"] >= (cutoff_date_first + pd.Timedelta(days=1)))
        ]
        
        # ----------------------------------------
        # medical_note 기본 필터링
        # ----------------------------------------
        
        # 원본 medical_note를 백업 (fallback용)
        medical_note_backup = medical_note.copy()
        
        exclude_note = "수술전기록|수술실퇴실전기록|24시간이내"
        medical_note = medical_note[~medical_note["서식명"].str.contains(exclude_note, na=False)]
        
        # 서식 항목명 필터링
        exclude_keywords = [
            "수신의", "의뢰진료과", "의뢰의",
            "환자현위치", "응급여부", "집도의", "보조의", "동의 받은 시간", "전담간호사"
        ]
        medical_note = medical_note.dropna(subset=["서식 항목명", "항목별 서식 값", "서식 작성일자"]).copy()
        medical_note["서식 항목명"] = medical_note["서식 항목명"].astype(str).str.strip().str.lower()
        
        pattern = "|".join(exclude_keywords)
        medical_note = medical_note[~medical_note["서식 항목명"].str.contains(pattern, case=False, na=False)]
        
        # 백업 데이터도 동일하게 필터링 (exclude_note 제외)
        medical_note_backup = medical_note_backup.dropna(subset=["서식 항목명", "항목별 서식 값", "서식 작성일자"]).copy()
        medical_note_backup["서식 항목명"] = medical_note_backup["서식 항목명"].astype(str).str.strip().str.lower()
        medical_note_backup = medical_note_backup[~medical_note_backup["서식 항목명"].str.contains(pattern, case=False, na=False)]
        
        # ========================================
        # [수술] 섹션 처리
        # ========================================
        op_notes_final_text = process_operation_notes(medical_note)
        
        # 수술 노트가 비어있으면 백업 데이터에서 fallback 시도
        if not op_notes_final_text or op_notes_final_text.strip() == '<operation_note>\n</operation_note>':
            op_notes_final_text = process_operation_notes(medical_note_backup)
        
        # ========================================
        # [타과] 섹션 처리
        # ========================================
        consult_notes_concat = process_consult_notes(medical_note)
        
        # ========================================
        # [경과] 섹션 처리
        # ========================================
        progress_notes_concat = process_progress_notes(medical_note)
        
        # ========================================
        # 최종 텍스트 생성
        # ========================================
        final_text = progress_notes_concat + '\n\n' + op_notes_final_text + '\n\n' + consult_notes_concat
        
        return final_text, department
        
    except Exception as e:
        print(f"환자 {p_num} 처리 중 오류 발생: {str(e)}")
        return None


# =============================================================================
# 메인 전처리 함수
# =============================================================================
def preprocess_section24(admission, medical_note, surgery):
    """
    Section 2,4를 위한 전체 환자 데이터 전처리
    """
    print("Section 2,4 전처리 시작...")
    
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
            medical_note,
            surgery
        )
        
        if result is not None:
            final_text, department = result  # 튜플 언팩
            results.append({
                "patient_id": p_num,
                "final_text": final_text,
                "입원시_진료과": department  # 추가: 진료과 정보
            })
    
    print(f"✓ Section 2,4 전처리 완료! 총 {len(results)}명의 환자 데이터가 생성되었습니다.")
    
    return results