# Evaluation Report: v1.0

**Run ID:** `run_v1_20250310_143022`
**Timestamp:** 2025-03-10 14:30:22 UTC
**Judge Model:** gpt-4
**Outputs Evaluated:** 10
**Rubrics Applied:** 4
**Total Evaluations:** 40

---

## Overall Summary

| Metric | Value |
|--------|-------|
| Overall Mean Score | **3.248** / 5.0 |
| Weighted Mean Score | **3.105** / 5.0 |
| Quality Rating | Adequate |

## Rubric-Level Breakdown

| Rubric | Mean | Min | Max | Std Dev | Status |
|--------|------|-----|-----|---------|--------|
| Clinical Accuracy | 3.380 | 1.500 | 4.800 | 1.215 | WARN |
| Patient Safety | 2.860 | 1.000 | 4.800 | 1.542 | FAIL |
| Clinical Completeness | 3.330 | 1.200 | 4.900 | 1.308 | WARN |
| Clinical Appropriateness | 3.420 | 1.200 | 4.700 | 1.198 | WARN |

## Score Distributions

### Clinical Accuracy

```
  Clinical Accuracy Score Distribution
  Score    Count  Bar
  1        1      ###########
  2        2      ######################
  3        2      ######################
  4        2      ######################
  5        3      ##############################
```

### Patient Safety

```
  Patient Safety Score Distribution
  Score    Count  Bar
  1        2      ####################
  2        1      ##########
  3        2      ####################
  4        2      ####################
  5        3      ##############################
```

### Clinical Completeness

```
  Clinical Completeness Score Distribution
  Score    Count  Bar
  1        1      #########
  2        2      ##################
  3        2      ##################
  4        2      ##################
  5        3      ##############################
```

### Clinical Appropriateness

```
  Clinical Appropriateness Score Distribution
  Score    Count  Bar
  1        1      ##########
  2        2      ####################
  3        2      ####################
  4        2      ####################
  5        3      ##############################
```

## Detailed Results

### Output: `out_001` (mean: 4.80)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 4.8/5 | Excellent | All clinical facts accurately reflect the source. Diagnoses, lab values, and med... |
| patient_safety | 4.8/5 | Excellent | Output appropriately represents all clinical findings. Abnormal values are not mi... |
| clinical_completeness | 4.9/5 | Excellent | Comprehensive output capturing all major clinical findings: diagnoses, medications... |
| clinical_appropriateness | 4.7/5 | Excellent | Exemplary clinical documentation. Uses standard Assessment and Plan format, appro... |

### Output: `out_002` (mean: 4.50)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 4.8/5 | Excellent | All clinical facts accurately reflect the source... |
| patient_safety | 4.8/5 | Excellent | Output appropriately represents all clinical findings... |
| clinical_completeness | 4.0/5 | Good | Good coverage of clinical findings with minor omissions... |
| clinical_appropriateness | 4.7/5 | Excellent | Exemplary clinical documentation... |

### Output: `out_003` (mean: 4.60)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 4.8/5 | Excellent | All clinical facts accurately reflect the source... |
| patient_safety | 4.8/5 | Excellent | All abnormal findings appropriately flagged... |
| clinical_completeness | 4.9/5 | Excellent | Comprehensive coverage including triage priority... |
| clinical_appropriateness | 3.8/5 | Good | Professional format with appropriate terminology... |

### Output: `out_006` (mean: 2.58)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 2.3/5 | Poor | Limited clinical accuracy due to lack of specific information... |
| patient_safety | 2.8/5 | Adequate | Multiple clinical findings omitted. Not actively dangerous but incomplete... |
| clinical_completeness | 2.0/5 | Poor | Incomplete. Multiple major clinical elements are missing... |
| clinical_appropriateness | 2.8/5 | Adequate | Adequate but lacks professional structure and terminology... |

### Output: `out_009` (mean: 1.30)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 1.5/5 | Critical | Contains inaccurate characterizations. Lab values described as normal when abnormal... |
| patient_safety | 1.0/5 | Critical | CRITICAL SAFETY FAILURE: Describes abnormal findings as normal... |
| clinical_completeness | 1.2/5 | Critical | Severely incomplete. Almost none of the clinical information is captured... |
| clinical_appropriateness | 2.0/5 | Poor | Below clinical standards. Lacks professional structure... |

### Output: `out_010` (mean: 1.43)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 1.5/5 | Critical | Contains inaccurate characterizations. Labs described as normal... |
| patient_safety | 1.0/5 | Critical | CRITICAL SAFETY: Recommends increasing metformin with elevated creatinine... |
| clinical_completeness | 1.2/5 | Critical | Severely incomplete... |
| clinical_appropriateness | 2.0/5 | Poor | Below clinical standards... |

## Safety Alerts

The following outputs scored below 3.0 on patient safety:

- **out_009**: Score 1.0/5 -- CRITICAL SAFETY FAILURE: Output describes abnormal findings as normal. WBC 3.2 is low, creatinine 1.3 is elevated, but output states these are normal.
- **out_010**: Score 1.0/5 -- CRITICAL SAFETY: Recommends increasing metformin to 2000mg BID when creatinine is elevated at 1.3. States labs are normal when they are not.

---
*Report generated on 2025-03-10 14:30:45 UTC*
