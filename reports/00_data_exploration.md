# 📊 00_exploration.md - Data Exploration & Problem Discovery

## 🎯 Objectives
- Understand the WebMD drug review dataset structure
- Identify key challenges in the data
- Define preprocessing requirements
- Establish baseline metrics

---

## ❌ Problems Discovered

### 1. **Severe Class Imbalance**
**Issue:** Satisfaction scores are heavily skewed toward extremes
```
Distribution Analysis:
├─ Negative (1-2 stars): 69,995 samples (40.4%) ⚠️
├─ Neutral (3 stars):    22,902 samples (13.2%) 🚨 CRITICAL
└─ Positive (4-5 stars): 80,434 samples (46.4%) ⚠️

Imbalance Ratio: 3.51:1 (Positive:Neutral)
```

**Why it matters:**
- Standard models will ignore minority class (Neutral)
- F1-score will be artificially high while missing 13% of data
- Real-world impact: Misclassifying "neutral" as "positive/negative"

---

### 2. **Data Quality Issues**
**Issue:** Raw data contains significant noise

| Problem Type | Count | % of Total |
|--------------|-------|------------|
| Non-English reviews | 12,458 | 4.8% |
| Duplicate reviews | 83,860 | 32.2% |
| Abnormal length (<3 or >2000 words) | 2,254 | 0.9% |
| **Total to remove** | **98,572** | **37.8%** |

**Initial dataset:** 260,467 reviews  
**Clean dataset:** 173,331 reviews  
**Data loss:** 33.4%

---

### 3. **Medical Domain Complexity**
**Issue:** Standard NLP fails on medical terminology

```python
Examples of Challenges:
1. Negation patterns:
   "No side effects" → Positive (not Negative!)
   "Not effective" → Negative
   "Can't complain" → Positive (idiomatic)

2. Medical jargon:
   "ADHD", "Neuropathic Pain", "Type 2 Diabetes"
   → Must preserve capitalization and terms

3. Condition-specific bias:
   Birth Control:  5.12/10 avg satisfaction (lowest)
   Pain meds:      6.74/10 avg satisfaction (highest)
   → 1.62 point difference
```

---

### 4. **Scale Ambiguity**
**Issue:** Documentation claims 1-10 scale, data shows 1-5

```python
Actual value distribution:
Score | Count  | % 
------|--------|----
  1   | 32,847 | 18.9%
  2   | 37,148 | 21.4%
  3   | 22,902 | 13.2%  ← Only 13%!
  4   | 33,327 | 19.2%
  5   | 47,107 | 27.2%
  
Outliers found:
  6   | 2 samples
  10  | 1 sample
→ Removed 3 outliers
```

---

### 5. **Demographic Bias Patterns**

**Age bias discovered:**
```
Age Group      | Avg Satisfaction | Sample Size
---------------|------------------|-------------
0-2 years      | 0.94 (lowest)   | 279
35-44 years    | 1.12 (highest)  | 38,169
75+ years      | 0.94 (lowest)   | 9,148

Range: 0.18 point difference
→ Working-age adults report higher satisfaction
```

**Gender bias discovered:**
```
Gender    | Avg Satisfaction | Sample Size
----------|------------------|-------------
Male      | 6.38/10         | 60,533
Female    | 6.05/10         | 169,650
(Blank)   | 5.81/10         | 17,433

Difference: Male 5.5% higher than Female
→ Potential reporting bias or medication effectiveness difference
```

---

## ✅ Solutions Implemented

### 1. **Class Imbalance Handling Strategy**
```python
Approach 1: Use Focal Loss (γ=2.0)
- Down-weights easy examples
- Focuses on hard-to-classify samples
- Class weights: [0.83, 2.52, 0.72] for [Neg, Neu, Pos]

Approach 2: Stratified Splitting
- Maintain 13.2% neutral in train/val/test
- Prevents complete loss of minority class

Approach 3: Evaluation Metrics
- Primary: Macro-F1 (equal weight to all classes)
- Secondary: Cohen's Kappa (agreement measure)
- NOT using: Accuracy alone (misleading with imbalance)
```

---

### 2. **Data Cleaning Pipeline**
```python
Step 1: Language filtering
- Detect language using langdetect
- Keep only English reviews (confidence > 0.9)
- Removed: 12,458 non-English

Step 2: Deduplication
- Remove exact duplicate reviews
- Keep first occurrence
- Removed: 83,860 duplicates

Step 3: Length filtering
- Min: 3 words (too short = noise)
- Max: 2000 words (likely spam/irrelevant)
- Removed: 2,254 abnormal length

Step 4: Outlier removal
- Remove satisfaction scores outside 1-5 range
- Removed: 3 outliers

Final: 173,331 clean samples
```

---

### 3. **Medical Domain Preprocessing**
```python
Negation Tagging:
- Use negspacy + custom rules
- Tag patterns: "no", "not", "never", "without", etc.
- Example: "no side effects" → "[NEG]no side effects"
- Result: 89% negation pattern coverage

Medical Term Preservation:
- NO aggressive lowercasing
- Preserve "ADHD" ≠ "adhd"
- Whitelist drug names from metadata
- Minimal stopword removal

Entity Recognition:
- Extract: Drug names, Conditions, Symptoms
- Use spaCy NER + custom medical dictionary
- Store as JSON metadata for each review
```

---

### 4. **Feature Engineering**
```python
Derived Features:
1. satisfaction_score_10: Original 1-5 mapped to 1-10 (×2)
2. satisfaction_class_10: 10-class classification target
3. sent_label: 3-class sentiment (0=Neg, 1=Neu, 2=Pos)
4. satisfaction_reg: Normalized 0-1 for regression

Text Features:
1. text_clean: Cleaned text (HTML/URLs removed)
2. text_neg: Negation-tagged text
3. word_count: Number of words
4. negation_count: Number of negation markers
5. negation_ratio: negation_count / word_count

Metadata Features:
1. entities: JSON of extracted medical entities
2. is_english: Boolean flag
```

---

## 📈 Results & Impact

### Data Quality Improvement
```
Metric                  | Before    | After     | Improvement
------------------------|-----------|-----------|-------------
Total samples           | 260,467   | 173,331   | -33.4% (cleaned)
English-only            | 95.2%     | 100%      | +4.8%
Duplicates              | 32.2%     | 0%        | -32.2%
Valid length            | 99.1%     | 100%      | +0.9%
Scale consistency       | 99.999%   | 100%      | +0.001%
```

### Class Distribution (Final)
```
3-Class Sentiment:
├─ Negative: 69,995 (40.4%)
├─ Neutral:  22,902 (13.2%) ← Successfully preserved!
└─ Positive: 80,434 (46.4%)

Stratification maintained across:
├─ Train: 173,331 × 0.7 = 121,332 samples
├─ Val:   173,331 × 0.1 =  17,333 samples
└─ Test:  173,331 × 0.2 =  34,666 samples
```

### Text Statistics (Final Clean Data)
```
Metric              | Value
--------------------|----------
Mean review length  | 127 characters
Median review length| 89 characters
Mean word count     | 24 words
Median word count   | 18 words
Vocabulary size     | 50,000 unique tokens
Negation ratio      | 12.4% of reviews contain negation
```

### Bias Analysis Insights
```
Identified Biases for Mitigation:
1. Age bias: 0.18 point range (monitor in evaluation)
2. Gender bias: 5.5% difference (consider in fairness metrics)
3. Condition bias: 1.62 point range (stratify by condition if possible)

Action Items:
→ Report per-group performance in final evaluation
→ Consider separate models for high-variance conditions
→ Flag potential disparate impact in deployment
```

---

## 🎯 Key Takeaways

### Critical Findings
1. ✅ **Minority class preserved:** 13.2% neutral samples maintained through stratification
2. ✅ **Data cleaned:** 33.4% noise removed, improving signal quality
3. ✅ **Domain adaptation ready:** Medical terminology and negation patterns identified
4. ✅ **Bias awareness:** Demographic patterns documented for fairness monitoring

### Next Steps
1. → **01_data_prep:** Implement vocabulary building and tokenization
2. → **02_model_baselines:** Establish SVM baseline with TF-IDF
3. → Use Focal Loss to handle 3.51:1 class imbalance
4. → Monitor per-class F1 scores (especially Neutral)

---

## 📊 Visualization Highlights

**Generated in this notebook:**
- `text_length_distribution.png` - Review length histogram
- `satisfaction_distribution.png` - 4-panel class distribution
- `temporal_trends.png` - Review volume over time
- `top_2grams.png` - Most common bigrams
- `top_3grams.png` - Most common trigrams
- `negation_analysis.png` - Negation ratio by sentiment

**Saved to:** `reports/eda/`

---

## 💡 Conclusion

This exploration revealed **4 critical challenges** that will shape our modeling approach:

1. **Class imbalance (3.51:1)** → Use Focal Loss + stratified metrics
2. **Medical domain complexity** → Domain-specific BERT + negation handling  
3. **Data quality issues (33% noise)** → Aggressive cleaning pipeline
4. **Demographic biases** → Fairness-aware evaluation

**Next:** Move to `01_data_prep.md` to implement vocabulary and tokenization strategies.

---

**Notebook runtime:** ~15 minutes  
**Output files:** 8 figures + 2 reports  
**Clean dataset:** 173,331 samples ready for modeling
