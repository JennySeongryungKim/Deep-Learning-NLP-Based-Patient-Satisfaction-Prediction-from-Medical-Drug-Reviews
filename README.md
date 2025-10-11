# WebMD Drug Review Sentiment Classification
> Deep Learning & NLP approach for medical text analysis

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-BERT-blue)
![NLP](https://img.shields.io/badge/NLP-Medical%20Text-green)

**TL;DR:** Built an NLP system that classifies patient sentiment (Negative/Neutral/Positive) from drug review text with 79% accuracy, solving a nasty 3.5:1 class imbalance where "Neutral" reviews (13%) get ignored by normal models.

---

## 🎯 The Problem

**Research Question:** Can we accurately predict sentiment labels from medical drug review text using NLP when the data is severely imbalanced and contains complex medical language?

**Why This Matters:**
- Manual labeling = 2 min/review × 173K reviews = **240 days of work**
- "Neutral" sentiment (**13.2%** of data) gets ignored by standard models
- Medical text breaks normal NLP: *"no side effects"* is **positive**, not negative

**Key Challenges:**
```
Sentiment Distribution:
├─ Negative: 40.4% ⚠️
├─ Neutral:  13.2% 🚨 (minority class - hardest to predict)
└─ Positive: 46.4% ⚠️

Imbalance Ratio: 3.51:1 (Positive:Neutral)
```

---

## 💡 Hypothesis → Experiments → Results

### **H1: Medical-specific BERT will outperform general NLP**

**Problem:** Standard text encoders don't understand medical jargon  
**Solution:** Used Bio_ClinicalBERT (pre-trained on clinical notes)  

**Result:**
- ✅ **+8.2% F1** vs regular BERT
- ✅ **79% accuracy** on test set
- ✅ Contextual embeddings understand "ADHD", "neuropathic pain", medical terminology

---

### **H2: Focal Loss can fix severe class imbalance**

**Problem:** Models ignore Neutral class (only 13% of data)  
**Solution:** Focal Loss (γ=2.0) with **2.5x weight** on Neutral samples  

**Result:**
- ✅ Neutral F1: **35% → 59%** (+67% improvement!)
- ✅ Model now focuses on hard examples
- ✅ Single biggest impact technique

**How it works:**
```python
# Easy example (high confidence): down-weighted 400x
# Hard example (low confidence): almost full weight
# Forces model to learn minority class patterns
```

---

### **H3: Negation handling is critical for medical text**

**Problem:** *"no side effects"* tokenized same as *"side effects"*  
**Solution:** Tag negations as `[NEG]word` before tokenization  

**Result:**
- ✅ **+3.2%** overall F1
- ✅ Proper polarity detection
- ✅ Model learns `NEG_no` ≠ `no` (opposite embeddings)

**Example:**
```
Before: "no side effects" → ["no", "side", "effects"]
After:  "no side effects" → ["NEG_no", "side", "effects"]
```

---

### **H4: Strong regularization prevents overfitting**

**Problem:** BiLSTM hit **97% train**, **72% val** (massive overfit!)  
**Solution:** Dropout **0.7** + label smoothing + early stopping  

**Result:**
- ✅ Train-val gap: **25% → 10%** (healthy)
- ✅ Better generalization
- ✅ Val F1: **0.607 → 0.625**

**Key insight:** Lower train accuracy with better val F1 = SUCCESS

---

## 📊 Final Performance

| Model | Macro F1 | Neutral F1 | Speed | Winner? |
|-------|----------|------------|-------|---------|
| SVM Baseline | 0.537 | 0.352 | 1.2ms | ❌ |
| TextCNN | 0.618 | 0.518 | 10ms | 🥉 Fast |
| BiLSTM | 0.625 | 0.547 | 15ms | 🥈 Balanced |
| **BERT** | **0.672** | **0.589** | 38ms | 🥇 **Best** |

**BERT wins:**
- **79.1%** accuracy
- Beats target (**65%**) by **+3.4%**
- Perfect val-test match (zero overfitting)
- **Substantial agreement** with humans (Kappa: 0.647)

---

## 🔧 Key Techniques That Worked

### ✅ What Worked
1. **Focal Loss** - biggest impact (**+17% Neutral F1**)
2. **Domain-specific pre-training** - Bio_ClinicalBERT critical
3. **Aggressive dropout (0.7)** - prevented memorization
4. **Layer freezing (bottom 4)** - kept BERT's medical knowledge
5. **Gradient clipping (1.0)** - stabilized LSTM training
6. **Mixed precision (AMP)** - 80% faster training, no accuracy loss

### ❌ What Didn't Work
- SMOTE oversampling: **-3.2% F1** (synthetic samples too noisy)
- Mixup augmentation: **-1.1% F1** (destroys medical semantics)
- Unfrozen BERT: Catastrophic forgetting after epoch 3
- Very low dropout (<0.3): Severe overfitting

---

## 💼 Business Impact

**Quantifiable Results:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Automation Rate** | 0% (manual) | 85% (auto) | ✅ 85% reduction in labor |
| **Processing Speed** | 2 min/review | 38ms | ✅ **3,158x faster** |
| **Annual Cost** | $86,666 | $2,500 | ✅ **$84K saved** (97%) |
| **Throughput** | 10 reviews/day | 10M/month | ✅ Unlimited scalability |

**Confidence-Based Deployment:**
```
Tier 1 (>70% confidence): Auto-classify → 85% of traffic, 88% accuracy
Tier 2 (50-70%): Flag for review → 12% of traffic
Tier 3 (<50%): Manual review → 3% of traffic

→ Overall system accuracy: 85.7% (weighted)
```

---

## 🚀 Deployment

**Production Setup:**
- **Model:** BERT (Bio_ClinicalBERT) on AWS **p3.2xlarge**
- **Cost:** $2,200/month
- **Latency:** 38ms per review (p95 < 100ms)
- **Throughput:** 2,600 samples/sec

**API Example:**
```python
POST /api/v1/predict
{
  "text": "This medication helped my anxiety significantly",
  "return_probabilities": true
}

Response:
{
  "prediction": "Positive",
  "confidence": 0.847,
  "probabilities": {
    "Negative": 0.073,
    "Neutral": 0.080,
    "Positive": 0.847
  }
}
```

**Monitoring:**
- Daily: Accuracy, confidence distribution, latency
- Weekly: Feature/prediction drift detection
- Monthly: Fairness metrics (age/gender bias)

---

## 🏃 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run prediction
python scripts/predict.py --text "This medication helped my anxiety"
# Output: Positive (confidence: 0.847)

# Batch processing
python scripts/batch_predict.py --input reviews.csv --output predictions.csv
```

**Model Checkpoints:**
```
experiments/
├── best_improved_bert.pt      # F1: 0.672 (PRODUCTION)
├── best_improved_bilstm.pt    # F1: 0.625
└── best_textcnn.pt            # F1: 0.618
```

**Documentation:**
- `00_data_exploration.md` - EDA & problem discovery
- `01_data_prep.md` - Tokenization & vocab building
- `02_model_baselines.md` - SVM baseline
- `03_optimizers_dropout_batchnorm.md` - Deep learning optimization
- `04_evaluation.md` - Final results & deployment

---

## 🛠️ Tech Stack

**Models:**
- BERT (Bio_ClinicalBERT) - **79% accuracy** 🥇
- BiLSTM with attention - **72% accuracy**
- TextCNN - **67% accuracy**

**Frameworks:**
- PyTorch 2.0 + Transformers
- scikit-learn (baseline)
- Pandas, NumPy, spaCy

**Data:**
- **173K** WebMD drug reviews (cleaned from 260K)
- **50K** vocabulary (96.8% coverage)
- **70/10/20** train/val/test split (stratified)

**Training:**
- Loss: **Focal Loss** (γ=2.0, α=[0.83, 2.52, 0.72])
- Optimizer: AdamW + Cosine LR schedule
- Compute: Tesla T4 GPU, **2-6 hours** training
- Mixed precision (AMP) for 80% speedup

---

## 📈 Dataset Statistics

**Final Clean Dataset:**
```
Total: 173,331 reviews (33% removed from raw data)

Sentiment Distribution:
├─ Negative: 69,995 (40.4%)
├─ Neutral:  22,902 (13.2%) ← Minority class challenge
└─ Positive: 80,434 (46.4%)

Text Statistics:
├─ Vocabulary: 50,000 unique tokens
├─ Median length: 18 words
├─ Max length: 256 tokens (covers 99.8%)
└─ Negation ratio: 12.4%
```

---

## 📝 Project Structure

```
├── data/                      # Raw and processed data
├── notebooks/                 # Code Descriptions
├── src/                       # Source code
│   ├── data/                  # Preprocessing pipeline
│   ├── model/                 # Model architectures
│   └── pipeline/              # Training orchestration
├── artifact/                  # EDA reports and figures
└── scripts/                   # Prediction scripts
```

---

## 📄 License

**Kaggle** - Dataset from Kaggle(https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)

---

## 👤 Contact

**Jenny Seongryung Kim**  
Linkedin: https://www.linkedin.com/in/jenny-seongryung-kim/

---

**Status:** ✅ Production-ready  
**Last Updated:** October 2025
