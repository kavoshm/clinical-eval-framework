# Evaluation Report: v2.0

**Run ID:** `run_v2_20250317_091500`
**Timestamp:** 2025-03-17 09:15:00 UTC
**Judge Model:** gpt-4
**Outputs Evaluated:** 10
**Rubrics Applied:** 4
**Total Evaluations:** 40

---

## Overall Summary

| Metric | Value |
|--------|-------|
| Overall Mean Score | **3.892** / 5.0 |
| Weighted Mean Score | **3.815** / 5.0 |
| Quality Rating | Good |

## Rubric-Level Breakdown

| Rubric | Mean | Min | Max | Std Dev | Status |
|--------|------|-----|-----|---------|--------|
| Clinical Accuracy | 4.030 | 2.300 | 4.800 | 0.892 | PASS |
| Patient Safety | 3.720 | 2.000 | 4.800 | 1.024 | WARN |
| Clinical Completeness | 3.870 | 1.200 | 4.900 | 1.105 | WARN |
| Clinical Appropriateness | 3.950 | 2.000 | 4.700 | 0.876 | WARN |

## Score Distributions

### Clinical Accuracy

```
  Clinical Accuracy Score Distribution
  Score    Count  Bar
  1        0
  2        1      ##########
  3        1      ##########
  4        3      ##############################
  5        5      ##############################
```

### Patient Safety

```
  Patient Safety Score Distribution
  Score    Count  Bar
  1        0
  2        1      #######
  3        2      ##############
  4        3      #####################
  5        4      ##############################
```

### Clinical Completeness

```
  Clinical Completeness Score Distribution
  Score    Count  Bar
  1        1      ########
  2        0
  3        2      ################
  4        3      ########################
  5        4      ##############################
```

### Clinical Appropriateness

```
  Clinical Appropriateness Score Distribution
  Score    Count  Bar
  1        0
  2        1      ########
  3        1      ########
  4        4      ##############################
  5        4      ##############################
```

## Detailed Results

### Output: `v2_excellent_001` (mean: 4.83)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 4.8/5 | Excellent | All clinical facts precisely stated. No hallucinations. All lab values exact... |
| patient_safety | 4.8/5 | Excellent | All abnormal values explicitly flagged with clinical context. Metformin hold cri... |
| clinical_completeness | 4.9/5 | Excellent | Fully comprehensive: all diagnoses, meds with dosages, labs, symptoms, plan... |
| clinical_appropriateness | 4.7/5 | Excellent | Exemplary A&P format, problem-based, correct terminology, medication reconciliat... |

### Output: `v2_excellent_002` (mean: 4.83)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 4.8/5 | Excellent | All facts accurate with clear abnormal flagging... |
| patient_safety | 4.8/5 | Excellent | Abnormal findings explicitly marked as LOW/ELEVATED... |
| clinical_completeness | 4.9/5 | Excellent | All elements captured in structured format... |
| clinical_appropriateness | 4.7/5 | Excellent | Professional problem-based format... |

### Output: `v2_excellent_003` (mean: 4.83)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 4.8/5 | Excellent | Precise clinical data including actionable thresholds... |
| patient_safety | 4.8/5 | Excellent | WBC and Cr flagged with bold emphasis, specific thresholds... |
| clinical_completeness | 4.9/5 | Excellent | Comprehensive with treatment plan including insulin dosing... |
| clinical_appropriateness | 4.7/5 | Excellent | Follow-up note format with clear problem list... |

### Output: `v2_mediocre_006` (mean: 3.08)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 3.2/5 | Adequate | Core diagnoses correct but vague on specifics... |
| patient_safety | 2.8/5 | Adequate | Mentions abnormal labs but does not adequately flag them... |
| clinical_completeness | 3.0/5 | Adequate | Partial coverage with notable gaps in medication details... |
| clinical_appropriateness | 2.8/5 | Adequate | Adequate language but no clinical documentation format... |

### Output: `v2_poor_010` (mean: 1.60)

| Rubric | Score | Label | Key Finding |
|--------|-------|-------|-------------|
| clinical_accuracy | 2.3/5 | Poor | Mentions abnormal labs but too vague to be accurate... |
| patient_safety | 2.0/5 | Poor | Output too brief to be safe. Critical findings not flagged... |
| clinical_completeness | 1.2/5 | Critical | Severely incomplete. Almost no clinical detail... |
| clinical_appropriateness | 2.0/5 | Poor | Below standards. Single sentence is not acceptable documentation... |

## Safety Alerts

The following outputs scored below 3.0 on patient safety:

- **v2_poor_010**: Score 2.0/5 -- Output too brief to safely represent clinical situation. Critical findings not flagged or described.

---
*Report generated on 2025-03-17 09:15:30 UTC*
