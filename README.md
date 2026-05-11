

# 🛡️ PhishLens — ML Phishing Email Detection System

[![CI](https://github.com/CyberSec-Sagar-Security/PhishSentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/CyberSec-Sagar-Security/PhishSentinel/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://bandit.readthedocs.io/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-orange)](https://huggingface.co/spaces/CyberSec-Sagar-Security/PhishLens)

> **Portfolio Project** — B.Sc. Information Technology → MSc Cybersecurity  
> A production-ready, multi-layer phishing email detection system combining classical ML,  
> deep NLP embeddings, threat intelligence APIs, and ChatGPT AI.

---

## 📋 Overview

PhishLens detects phishing emails using a **defence-in-depth ML architecture** — seven independent
detection layers that an adversary must simultaneously evade to successfully deliver a phishing email.

```
Raw Email
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Header Forensics (SPF/DKIM/DMARC, hop analysis)        │  12 features
│ Layer 2: URL Intelligence (lexical + WHOIS + cert transparency) │  29 features
│ Layer 3: HTML Structural Analysis (forms, obfuscation, pixels)  │  11 features
│ Layer 4: Semantic Embeddings (all-MiniLM-L6-v2, 384-dim)        │ 384 features
│ Layer 5: TF-IDF Vocabulary Signals                              │ 500 features
│ Layer 6: Threat Intelligence (VirusTotal, GSB, AbuseIPDB, etc.) │  13 features
│ Layer 7: Isolation Forest Zero-Day Anomaly Detection            │   1 feature
│ Layer 8: ChatGPT AI (LLM contextual analysis — optional)        │   2 features
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
ML Ensemble (XGBoost + LightGBM + CatBoost + RF + LR)
    │
    ▼
SHAP + LIME Explanation + MITRE ATT&CK Mapping + IOC Export
```


## 🏆 Model Performance

> All metrics measured on the **held-out test set of 103,067 emails** never seen during training (stratified 80/20 split).  
> Training corpus: 412,265 emails total. Full pipeline: **21.4 minutes with GPU acceleration**.

| Model | F1 | AUC-ROC | FNR | FPR | MCC |
|---|---|---|---|---|---|
| 🥇 **LightGBM** *(primary)* | **0.9505** | **0.9941** | **5.64%** | 3.11% | **0.9143** |
| 🥈 XGBoost | 0.9482 | 0.9935 | 6.48% | 2.78% | 0.9109 |
| Random Forest | 0.9436 | 0.9926 | 6.19% | 3.73% | 0.9021 |
| CatBoost | 0.9356 | 0.9895 | 6.76% | 4.52% | 0.8880 |
| LR | 0.9334 | 0.9905 | 9.87% | 2.22% | 0.8886 |

> **FNR = False Negative Rate** (phishing emails missed). LightGBM missed **2,480 out of 43,956** phishing emails in the test set.  
> Logistic Regression achieving 93.34% F1 — near the best model — demonstrates that **feature engineering quality is the primary driver**, not model complexity.

**Training corpus:** CASIS/Kaggle · Enron · SpamAssassin · phishing_pot · Nigerian Fraud → **412,265 total emails** (test set: 103,067 emails)

### ⚠️ Adversarial Stress Test (Honest Disclosure)

LightGBM stress-tested with Gaussian noise injected across all 961 feature dimensions simultaneously:

| Noise Level | F1 | Notes |
|---|---|---|
| 0% (baseline) | 0.9505 | Normal operation |
| 5% noise | ~0.60 | Synthetic worst-case (formal benchmark pending) |
| 10% noise | ~0.25 | Synthetic worst-case (formal benchmark pending) |
| 20% noise | ~0.14 | Synthetic worst-case (formal benchmark pending) |

> This is a *synthetic* scenario (all 961 features perturbed simultaneously). Real adversarial attacks manipulate specific semantic features while leaving others unchanged. Disclosed for full transparency — omitting it would be dishonest.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.13+
- 16GB RAM recommended (sentence-transformers + CatBoost)
- GPU optional but highly recommended — reduces embedding stage from ~4–5 hours to ~10–15 minutes (see [GPU_SETUP.md](GPU_SETUP.md))

### Installation

```bash
git clone https://github.com/CyberSec-Sagar-Security/PhishSentinel.git
cd PhishSentinel/phishlens

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Training

```bash
# Build processed datasets first (recommended):
python download_datasets.py

# Fast training (XGBoost only, no Optuna):
python train.py --data-dir data/processed --models xgboost --no-network --eval --save models

# Full training pipeline (all models + Optuna tuning):
python train.py --data-dir data/processed --models all --tune --eval --save models

# Offline mode (no network calls — useful for CI):
python train.py --data-dir data/processed --no-network --eval

# With adversarial testing:
python train.py --data-dir data/processed --adversarial --eval
```

### Streamlit Web Interface

```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📂 Project Structure

```
phishlens/
├── src/
│   ├── ingestion/
│   │   ├── eml_parser.py          # .eml file parsing (RFC 2822 + MIME)
│   │   └── dataset_loader.py      # SpamAssassin / CASIS / phishing_pot / Enron loaders
│   ├── features/
│   │   ├── header_features.py     # 12 header forensics features
│   │   ├── url_features.py        # 29 URL lexical + WHOIS + cert features
│   │   ├── html_features.py       # 11 HTML structural anomaly features
│   │   ├── text_features.py       # 384-dim embeddings + TF-IDF + urgency
│   │   ├── intelligence.py        # VirusTotal / GSB / AbuseIPDB / URLScan
│   │   ├── gemini_analyzer.py     # ChatGPT AI analysis (gpt-4o-mini)
│   │   └── pipeline.py            # Master FeaturePipeline (fit/transform)
│   ├── detection/
│   │   └── anomaly.py             # Isolation Forest zero-day detector
│   ├── models/
│   │   ├── trainer.py             # Optuna + MLflow model training
│   │   ├── evaluator.py           # Metrics, confusion matrix, stress test
│   │   ├── explainer.py           # SHAP + LIME dual explainability
│   │   ├── adversarial_tester.py  # 5 adversarial attack simulations
│   │   └── transformer_model.py   # DistilBERT fine-tuning (optional)
│   ├── utils/
│   │   ├── config.py              # PhishLensConfig dataclass + defaults
│   │   └── logger.py              # loguru structured logging
│   ├── ioc_extractor.py           # IOC extraction → MISP JSON + CEF syslog
│   └── attack_mapping.py          # MITRE ATT&CK T1566 technique mapping
├── tests/
│   ├── test_eml_parser.py         # 14 unit tests
│   ├── test_url_features.py       # 13 unit tests
│   ├── test_html_features.py      # 11 unit tests
│   └── test_pipeline.py           # Integration tests (50 synthetic emails)
├── data/
│   ├── raw/                       # Dataset storage (gitignored)
│   └── README.md                  # Dataset download instructions
├── models/                        # Trained model artifacts (gitignored)
├── reports/                       # Evaluation reports + confusion matrix PNGs
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── train.py                       # Training CLI
├── app.py                         # Streamlit web interface
├── requirements.txt               # Pinned dependencies
└── .env                           # API keys (gitignored)
```

---

## 🔌 API Key Configuration

PhishLens integrates with five threat intelligence services. Keys are stored in `.env`:

```env
OPENAI_API_KEY=your_openai_key          # platform.openai.com/api-keys
VIROTOTAL_API_KEY=your_vt_key           # Free: 4 requests/min, 500/day
GOOGLE_SAFE_BROWSING_API_KEY=your_gsb_key  # Free: 10,000 requests/day
ABUSEIPDB_API_KEY=your_abuseipdb_key    # Free: 1,000 requests/day
URLSCAN_API_KEY=your_urlscan_key        # Free tier available
```

> **Privacy note**: URL enrichment APIs send URL/IP data to third-party services.
> Disable `use_intelligence_apis` for sensitive email analysis.

---

## 🧠 Explainability

Every prediction includes:

1. **SHAP Feature Importance** — Which features contributed most to the phishing score  
2. **LIME Local Explanation** — Model-agnostic local approximation  
3. **SHAP/LIME Agreement Score** — Jaccard similarity of top-5 features (trust indicator)  
4. **MITRE ATT&CK Mapping** — Detected techniques with evidence and confidence scores  
5. **ChatGPT AI Narrative** — Plain English explanation (optional, requires OpenAI API key)

---

## 🎯 MITRE ATT&CK Coverage

PhishLens maps detections to:

| Technique ID | Technique Name | Tactic |
|---|---|---|
| T1566 | Phishing | Initial Access |
| T1566.001 | Spearphishing Attachment | Initial Access |
| T1566.002 | Spearphishing Link | Initial Access |
| T1036 | Masquerading | Defense Evasion |
| T1204 | User Execution | Execution |
| T1056 | Input Capture | Collection |
| T1078 | Valid Accounts | Persistence |
| T1027 | Obfuscated Files or Information | Defense Evasion |
| T1071.003 | Application Layer Protocol: Mail | Command and Control |

---

## 📊 Architecture Decisions

### Why Multi-Layer?
Single-feature-type models (URL-only, text-only) are easily evaded. PhishLens requires
simultaneous evasion of header forensics + URL analysis + HTML analysis + semantic embeddings
+ threat intelligence. This dramatically increases the cost for an adversary.

### Why SHAP + LIME Both?
SHAP (game-theoretic, exact) + LIME (model-agnostic approximation) provide independent
explanations. High agreement score = trustworthy explanation. Low agreement = investigate further.

### Why Isolation Forest?
Supervised models only detect phishing patterns seen in training data. Isolation Forest
trained on legitimate emails flags **any** structural anomaly — including novel campaigns
targeting brands not in training data.

### Limitations
| Limitation | Impact | Mitigation |
|---|---|---|
| Training data staleness | 6-month decay estimated | Retrain quarterly |
| Targeted spear-phishing | May evade without brand in training | ChatGPT AI + Isolation Forest |
| API rate limits | Intelligence enrichment slowed | Fallback to -1 (offline mode) |
| Encrypted content | Can't analyse S/MIME encrypted bodies | Header-only analysis |
| Image-only phishing | Minimal text features | HTML image ratio feature |
| Adversarial noise (5% Gaussian, all dims) | F1 drops 0.9505 → ~0.60 | Synthetic worst-case — see stress test table above |

---

## ⚡ Training Performance

The full training pipeline on 168,608 emails completes in **21.4 minutes** with GPU acceleration (NVIDIA RTX 2000 Ada, 8 GB VRAM). Without GPU, the same pipeline requires approximately 4–5 hours — the sentence-transformer embedding stage alone accounts for the majority of that time. GPU acceleration delivers a **12–15× speedup** on this hardware configuration.

See [GPU_SETUP.md](GPU_SETUP.md) for CUDA installation instructions and performance benchmarks.


## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sagar**  
Portfolio project demonstrating ML security engineering skills.

> *"The best firewall is a good detector."*
