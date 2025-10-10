# 🧬 WebMD Drug Reviews Dataset — Data Dictionary & Provenance

## 1️⃣ Overview

This project uses the **[WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)**  
created by **Rohan Harode** on Kaggle.

- **License:** Used under [Kaggle Terms of Service](https://www.kaggle.com/terms) for educational and research purposes.  
- **Purpose:** Analyze patient-generated reviews to extract sentiment and aspect-based patterns related to drug efficacy, side effects, convenience, and cost.

---

## 2️⃣ Dataset Structure

| Column | Type | Description |
|--------|------|-------------|
| **Drug Name** | string | Name of the medication reviewed |
| **Condition** | string | Condition for which the drug was taken |
| **Review** | string | Free-text review content |
| **Rating** | integer (1–10) | User’s satisfaction rating |
| **Date** | string (YYYY-MM-DD) | Date of the review |
| **UsefulCount** | integer | Number of “helpful” votes received |
| **Age** | categorical | Age group of the reviewer (e.g., “25–34”) |
| **Sex** | categorical | Gender of the reviewer |
| **Duration** | string | How long the reviewer took the drug |
| **Satisfaction_Label** | integer (mapped 0–2) | Sentiment category: 0 = Negative, 1 = Neutral, 2 = Positive |

---

## 3️⃣ Preprocessing Summary

- **Text cleaning:** punctuation removal, contraction expansion  
- **Negation tagging:** handled via `negspacy` for “no”, “not”, “without”, “denied” constructs  
- **Stopword filtering:** domain-specific stopwords (e.g., “drug”, “tablet”, “medication”) removed  
- **Tokenization:** spaCy + custom tokenizer for medical slang  
- **Sampling:** stratified by `Satisfaction_Label` for balanced training sets  

---

## 4️⃣ Exploratory Data Analysis Summary

### 📏 Text Length Statistics
| Metric | Value |
|:-------|------:|
| Mean Length | 347.45 |
| Median Length | 254.00 |
| Std Dev | 318.61 |
| Min / Max | 5 / 3145 |
| Mean Word Count | 65.49 |

---

### 😀 Satisfaction Distribution
| Rating | Count | Sentiment |
|:------:|------:|-----------|
| 2 | 76,094 | Negative |
| 4 | 23,898 | Negative |
| 6 | 32,718 | Neutral |
| 8 | 39,835 | Positive |
| 10 | 75,071 | Positive |

**10-class imbalance ratio:** 3.18  
**Sentiment imbalance ratio:** 3.51

---

### 🧠 Top Bigrams
| Bigram | Frequency |
|:--------|-----------:|
| started taking | 9,393 |
| weight gain | 8,631 |
| blood pressure | 8,514 |
| feel like | 7,619 |
| taking medication | 6,734 |

### 💬 Top Trigrams
| Trigram | Frequency |
|:---------|-----------:|
| high blood pressure | 1,945 |
| just started taking | 926 |
| started taking medication | 845 |
| flu like symptoms | 725 |
| birth control pills | 477 |

---

### 🚫 Negation Analysis
| Metric | Value |
|:--------|-------:|
| Mean Negation Count | 1.23 |
| Mean Negation Ratio | 0.0189 |
| Reviews with Negation | 146,541 |

Negation by sentiment:  
- Negative: 0.0214  
- Neutral: 0.0165  
- Positive: 0.0175  

---

### 👥 Subgroup Analysis

**By Age Group (mean sentiment label):**
| Age | Mean | Std | Count |
|:----|-----:|----:|-----:|
| 25–34 | 1.07 | 0.93 | 35,854 |
| 35–44 | 1.12 | 0.92 | 38,169 |
| 45–54 | 1.10 | 0.93 | 54,274 |
| 55–64 | 1.05 | 0.93 | 50,650 |
| 65–74 | 1.02 | 0.93 | 27,078 |
| 75+ | 0.94 | 0.92 | 9,148 |

**By Sex (satisfaction mean):**
| Sex | Mean | Std | Count |
|:----|-----:|----:|-----:|
| Female | 6.05 | 3.30 | 169,650 |
| Male | 6.38 | 3.21 | 60,533 |

**By Condition (top 10):**
| Condition | Mean | Std | Count |
|:-----------|----:|----:|----:|
| Attention Deficit Disorder (ADHD) | 6.70 | 3.06 | 4,820 |
| Birth Control | 5.12 | 3.22 | 11,864 |
| Depression | 6.37 | 3.11 | 11,021 |
| High Blood Pressure | 5.40 | 3.12 | 14,702 |
| Pain | 6.74 | 3.04 | 11,832 |
| Type 2 Diabetes Mellitus | 5.78 | 3.24 | 4,114 |

---

## 5️⃣ Data Quality Notes

- Duplicate reviews: <2% (removed)
- Missing Age/Sex: imputed with mode  
- Outlier review lengths (>3,000 chars): capped at 99th percentile
- Class imbalance handled by SMOTETomek hybrid resampling

---

## 6️⃣ Version & Provenance

- **Source:** Kaggle Dataset by Rohan Harode (WebMD Drug Reviews)  
- **Download Date:** 2025-10-10  
- **Local Path:** `data/raw/webmd_reviews.csv`  
- **Processed Outputs:**  
  - `data/interim/cleaned_reviews.parquet`  
  - `data/processed/train.csv`, `test.csv`  

---

## 7️⃣ Citation

If you use this dataset, please cite the original Kaggle author:

> Rohan Harode, *WebMD Drug Reviews Dataset*, Kaggle (2023).  
> URL: [https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)

---

## 8️⃣ Contact

Maintainer: **Seongryung Jenny Kim**  
Email: *(optional, e.g.,)* jenny.kim@wustl.edu  
Repository: [GitHub Link]

---

📄 *Generated automatically based on exploratory data analysis (EDA Report) and preprocessing logs.*
