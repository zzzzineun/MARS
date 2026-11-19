# M.A.R.S. — Medical Record Generation Datathon

**Grand Award (1st place) | Seoul National University Hospital, 2025**

Code from the **M.A.R.S. Medical Record Generation Datathon**, focused on **reliable clinical document generation and diagnosis prediction**.

The project explored workflow-guided *Brief Hospital Course* generation using MIMIC-IV-Note, preference learning from constructed omission and hallucination errors, retrieval-based factual verification, and diagnosis-relevant information for ICD-10 prediction.

## Repository contents

- **`section1_preprocessing.py`** — Prepares patient background, admission information, and clinical notes.
- **`section24_preprocessing.py`** — Processes surgical, consultation, and progress notes.
- **`prompts.py`** — Section-specific generation prompts.
- **`orpo.py`** — Preference-based model training with ORPO.
- **`sectionall_inference.py`** — Batched generation for multiple clinical sections.
- **`submit.py`** — Orchestrates preprocessing, model loading, and inference.

## Notes

These are research/competition scripts with environment-specific data and model paths. Update the configurations before execution; the clinical datasets and trained model weights are not included.
