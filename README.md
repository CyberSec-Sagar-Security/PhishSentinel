
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
>
> **GitHub:** https://github.com/CyberSec-Sagar-Security/PhishSentinel | **Branch:** `main` | **Python:** 3.13 | **License:** MIT

---

## 📋 What It Is

PhishLens detects phishing emails using a **defence-in-depth ML architecture** — eight independent
detection layers that an adversary must simultaneously evade to successfully deliver a phishing email.

---

## 🏗️ Architecture — 8-Layer Defence-in-Depth

```
Raw Email
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 1  │  Header Forensics          │  src/features/header_features.py  │  12 features  │
│  Layer 2  │  URL Intelligence          │  src/features/url_features.py     │  29 features  │
│  Layer 3  │  HTML Structural Analysis  │  src/features/html_features.py    │  11 features  │
│  Layer 4  │  Semantic Embeddings       │  src/features/text_features.py    │ 384 features  │
│  Layer 5  │  TF-IDF Vocabulary         │  src/features/text_features.py    │ 500 features  │
│  Layer 6  │  Threat Intelligence APIs  │  src/features/intelligence.py     │  13 features  │
│  Layer 7  │  Isolation Forest Anomaly  │  src/detection/anomaly.py         │   1 feature   │
│  Layer 8  │  ChatGPT AI (optional)     │  src/features/openai_analyzer.py  │   2 features  │
└─────────────────────────────────────────────────────────────────────────────────┘
                         TOTAL: 961 features (fixed vector — never change without full retrain)
    │
    ▼
ML Ensemble (XGBoost + LightGBM + CatBoost + Random Forest + Logistic Regression)
    │
    ▼
SHAP + LIME Explanation + MITRE ATT&CK Mapping + IOC Export
```

---

## 🔢 Feature Vector — 961 Dimensions (Fixed)

| # | Layer | Module | What Is Extracted | Count |
|---|---|---|---|---|
| 1 | Header Forensics | `header_features.py` | SPF/DKIM/DMARC pass/fail, From↔Reply-To mismatch, From↔Return-Path mismatch, freemail sender, timezone anomaly, X-Mailer bulk fingerprint, hop count, relay IP count, display name spoofing, MX record missing | **12** |
| 2 | URL Intelligence | `url_features.py` | Lexical (length, entropy, special chars, TLD, subdomain depth, @ symbol, IP in URL, brand in subdomain), WHOIS (domain age, registrar age), cert transparency, homoglyph detection — per-URL min/max aggregations + URL count | **29** |
| 3 | HTML Structure | `html_features.py` | Hidden form fields, external form actions, pixel tracking, image-only ratio, script obfuscation, data URI count, meta refresh, hidden iFrame, link text mismatch, comment ratio, DOM depth | **11** |
| 4 | Semantic Embeddings | `text_features.py` | `all-MiniLM-L6-v2` 384-dim sentence embeddings + 8 urgency/subject signals | **384** |
| 5 | TF-IDF Vocabulary | `text_features.py` | 500-token phishing vocabulary signal | **500** |
| 6 | Threat Intelligence | `intelligence.py` | VirusTotal malicious count, GSB threat type, AbuseIPDB confidence score, URLScan verdict, URLhaus status, IPQS fraud score + sub-scores | **13** |
| 7 | Anomaly Detection | `anomaly.py` | Isolation Forest score trained on legitimate emails only — detects novel zero-day campaigns not seen in training | **1** |
| 8 | ChatGPT AI | `openai_analyzer.py` | `gemini_is_phishing` (0/1), `gemini_confidence` (0.0–1.0) — key names retained for trained model weight compatibility | **2** |

> **Note on Layer 8 key names:** The dict keys `gemini_is_phishing` / `gemini_confidence` are intentionally kept as-is. They are baked into the 961-feature vector that all five trained models depend on. The analysis is now performed by ChatGPT (`openai_analyzer.py`) — the key names are a historical artifact.

---

## 🏆 Model Performance

> All metrics on the **held-out test set of 103,067 emails** — never seen during training (stratified 80/20 split).
> Training corpus: 412,265 emails total. Full pipeline: **21.4 minutes with GPU acceleration**.

| Rank | Model | F1 | AUC-ROC | FNR | FPR | MCC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **LightGBM** *(primary)* | **0.9505** | **0.9941** | **5.64%** | 3.11% | **0.9143** | 41,476 | 57,272 | 1,839 | 2,480 |
| 🥈 2 | XGBoost | 0.9482 | 0.9935 | 6.48% | 2.78% | 0.9109 | 41,109 | 57,468 | 1,643 | 2,847 |
| 🥉 3 | Random Forest | 0.9436 | 0.9926 | 6.19% | 3.73% | 0.9021 | 41,234 | 56,904 | 2,207 | 2,722 |
| 4 | CatBoost | 0.9356 | 0.9895 | 6.76% | 4.52% | 0.8880 | 40,984 | 56,438 | 2,673 | 2,972 |
| 5 | Logistic Regression | 0.9334 | 0.9905 | 9.87% | 2.22% | 0.8886 | 39,617 | 57,801 | 1,310 | 4,339 |

> **FNR = False Negative Rate** (phishing emails missed).
> LR at 93.34% F1 — near the top — proves **feature engineering quality is the primary driver**, not model complexity.

### ⚠️ Adversarial Stress Test (Honest Disclosure)

LightGBM stress-tested with Gaussian noise injected across all 961 feature dimensions simultaneously:

| Noise Level | F1 | Notes |
|---|---|---|
| 0% (baseline) | 0.9505 | Normal operation |
| 5% noise | ~0.60 | Synthetic worst-case (formal benchmark pending) |
| 10% noise | ~0.25 | Synthetic worst-case (formal benchmark pending) |
| 20% noise | ~0.14 | Synthetic worst-case (formal benchmark pending) |

> Real adversarial attacks manipulate specific features while leaving others unchanged — not all 961 at once. Disclosed for full transparency.

---

## 📊 Training Datasets

| Dataset | Source | Type |
|---|---|---|
| CASIS | Kaggle/CASIS | Phishing + Spam |
| Enron (Kaggle) | Kaggle | Legitimate |
| SpamAssassin HAM | Apache | Legitimate |
| SpamAssassin SPAM | Apache | Spam |
| Phishing Pot | GitHub | Phishing |
| Nigerian Fraud | Public | Phishing |
| Nazario | Public | Phishing |
| Ling | Public | Phishing |
| CEAS 2008 | Public | Spam |
| Meajor | Custom | Mixed |
| Umbrella Top 1M | Cisco | Legitimate domain reference |

**Total: 412,265 emails** (training) + **103,067 emails** (test, held-out)

---

## 📂 Project Structure (Git-Tracked Files)

```
PhishLens/                              ← Git root
├── app.py                              # Streamlit web interface
├── train.py                            # Training CLI
├── download_datasets.py                # Kaggle + dataset setup
├── eval_all.py                         # Evaluate all saved models
├── install_and_verify.py               # Dependency verification
├── setup_and_verify.py                 # Environment setup check
├── monitor_training.py                 # Live training progress monitor
├── phishlens.ps1                       # PowerShell launcher
├── requirements.txt                    # Pinned dependencies
├── README.md
├── GPU_SETUP.md                        # CUDA setup + benchmarks
├── .env.example                        # API key template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── ioc_extractor.py                # IOC extraction → MISP JSON + CEF syslog
│   ├── attack_mapping.py               # MITRE ATT&CK T1566 technique mapper
│   │
│   ├── ingestion/
│   │   ├── eml_parser.py               # RFC 2822 + MIME .eml parser
│   │   └── dataset_loader.py           # Multi-dataset loader (SpamAssassin, CASIS, Enron, etc.)
│   │
│   ├── features/
│   │   ├── header_features.py          # Layer 1 — 12 header forensics features
│   │   ├── url_features.py             # Layer 2 — 29 URL lexical + WHOIS + cert features
│   │   ├── url_cleaner.py              # URL normalisation helper
│   │   ├── html_features.py            # Layer 3 — 11 HTML structural anomaly features
│   │   ├── text_features.py            # Layers 4+5 — 384-dim embeddings + TF-IDF + 8 text signals
│   │   ├── intelligence.py             # Layer 6 — VirusTotal / GSB / AbuseIPDB / URLScan / IPQS
│   │   ├── openai_analyzer.py          # Layer 8 — ChatGPT AI analysis (gpt-4.1-mini)
│   │   └── pipeline.py                 # Master FeaturePipeline (fit/transform/save/load)
│   │
│   ├── detection/
│   │   └── anomaly.py                  # Layer 7 — Isolation Forest zero-day detector
│   │
│   ├── models/
│   │   ├── trainer.py                  # Optuna + MLflow training (50 trials, 5-fold CV)
│   │   ├── evaluator.py                # Metrics, confusion matrix, stress test
│   │   ├── explainer.py                # SHAP + LIME dual explainability
│   │   ├── adversarial_tester.py       # 5 adversarial attack simulations
│   │   └── transformer_model.py        # DistilBERT fine-tuning (optional)
│   │
│   └── utils/
│       ├── config.py                   # PhishLensConfig dataclass + defaults
│       └── logger.py                   # loguru structured logging
│
├── data/
│   ├── raw/                            # Raw datasets (gitignored)
│   └── README.md                       # Dataset download instructions
│
├── models/saved/                       # Trained .pkl artifacts (gitignored)
│
├── reports/
│   ├── metrics.json                    # Exact per-model test metrics (all 5 models)
│   └── figures/
│       ├── cm_lightgbm.png
│       ├── cm_xgboost.png
│       ├── cm_rf.png
│       ├── cm_catboost.png
│       └── cm_lr.png
│
└── .github/workflows/ci.yml            # GitHub Actions CI/CD
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.13+
- 16 GB RAM recommended (sentence-transformers + CatBoost)
- GPU optional but highly recommended — reduces embedding stage from ~4–5 hours to ~10–15 minutes (see [GPU_SETUP.md](GPU_SETUP.md))

### Installation

```bash
git clone https://github.com/CyberSec-Sagar-Security/PhishSentinel.git
cd PhishSentinel

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env
# Edit .env and add your keys
```

### Training

```bash
# Build processed datasets first:
python download_datasets.py

# Fast — XGBoost only, no Optuna:
python train.py --data-dir data/processed --models xgboost --no-network --eval --save models

# Full — all 5 models + Optuna hyperparameter tuning:
python train.py --data-dir data/processed --models all --tune --eval --save models

# With adversarial stress test:
python train.py --data-dir data/processed --models all --tune --adversarial --eval --save models

# Offline mode (no DNS/WHOIS/API calls — useful for CI):
python train.py --data-dir data/processed --no-network --eval
```

**Key CLI flags:**

| Flag | Effect |
|---|---|
| `--models xgboost\|lightgbm\|catboost\|rf\|lr\|all` | Which models to train |
| `--tune` | Optuna hyperparameter search (50 trials, 5-fold CV) |
| `--adversarial` | Run 5 adversarial attack simulations after training |
| `--no-network` | Disable DNS + WHOIS + intelligence API calls |
| `--eval` | Evaluate on test split, write `reports/metrics.json` |
| `--save DIR` | Save `.pkl` model artifacts to DIR |

### Streamlit Web Interface

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## ⚙️ ML Training Details

| Setting | Value |
|---|---|
| Hyperparameter search | Optuna, 50 trials per model |
| Cross-validation | 5-fold stratified |
| Class balancing | SMOTE oversampling |
| Experiment tracking | MLflow |
| Training time with GPU | ~21.4 minutes (NVIDIA RTX 2000 Ada, 8 GB VRAM) |
| Training time CPU-only | ~4–5 hours |
| GPU speedup | 12–15× |
| Feature scaler | StandardScaler (fitted on train, applied to test) |
| Threshold | 0.5 (default for all models) |

See [GPU_SETUP.md](GPU_SETUP.md) for CUDA installation instructions.

---

## 🔌 API Key Configuration

All keys stored in `.env` (gitignored):

```env
OPENAI_API_KEY=your_openai_key                # platform.openai.com/api-keys
VIRUSTOTAL_API_KEY=your_vt_key                # Free: 4 req/min, 500/day
GOOGLE_SAFE_BROWSING_API_KEY=your_gsb_key     # Free: 10,000 req/day
ABUSEIPDB_API_KEY=your_abuseipdb_key          # Free: 1,000 req/day
URLSCAN_API_KEY=your_urlscan_key              # Free tier available
```

| Service | Free Tier | Data Provided |
|---|---|---|
| VirusTotal | 4 req/min, 500/day | Malicious/suspicious URL + IP counts |
| Google Safe Browsing | 10,000 req/day | Threat type (phishing, malware, unwanted) |
| AbuseIPDB | 1,000 req/day | IP abuse confidence score |
| URLScan.io | Free tier | URL scan verdict + screenshot |
| IPQS (IP Quality Score) | Free tier | Fraud score, proxy/VPN/bot detection flags |

> All intelligence features fall back to `-1` when offline or rate-limited. Training always runs with `use_intelligence_apis=False`.

> **Privacy note:** URL enrichment APIs send URL/IP data to third-party services. Disable `use_intelligence_apis` for sensitive email analysis.

---

## 🤖 ChatGPT AI Layer (`openai_analyzer.py`)

- **Model:** `gpt-4.1-mini`
- **Input sent to API:** Full email headers + extracted IOCs + threat intelligence verdicts + ML probability + email body (up to 3,000 chars)
- **Output fields:**

| Field | Type | Description |
|---|---|---|
| `gemini_is_phishing` | `int` (0/1) | Binary phishing verdict |
| `gemini_confidence` | `float` (0.0–1.0) | Confidence score |
| `gemini_risk_level` | `str` | LOW / MEDIUM / HIGH / CRITICAL |
| `gemini_impersonated_brand` | `str\|None` | Brand being impersonated (if any) |
| `gemini_phishing_signals` | `list` | List of detected phishing signals |
| `gemini_social_engineering` | `list` | Social engineering techniques detected |
| `gemini_explanation` | `str` | Plain English narrative |
| `gemini_recommended_action` | `str` | BLOCK / QUARANTINE / MONITOR |
| `gemini_ioc_verdicts` | `dict` | Per-IOC AI verdicts |
| `_ai_provider` | `str` | `"ChatGPT gpt-4.1-mini"` |

- **Completely optional** — falls back to `-1` values when `OPENAI_API_KEY` is not set
- **Graceful degradation** — API failures log a warning and return defaults, pipeline never crashes

---

## 🧠 Explainability Output (Per Prediction)

1. **SHAP Feature Importance** — Game-theoretic exact attribution per feature
2. **LIME Local Explanation** — Model-agnostic local approximation
3. **SHAP/LIME Agreement Score** — Jaccard similarity of top-5 features (trust indicator; low agreement = investigate further)
4. **MITRE ATT&CK Mapping** — Technique IDs, tactics, confidence scores, evidence strings
5. **ChatGPT AI Narrative** — Plain English explanation (requires `OPENAI_API_KEY`)

---

## 🎯 MITRE ATT&CK Coverage

| Technique ID | Technique Name | Tactic |
|---|---|---|
| T1566 | Phishing | Initial Access |
| T1566.001 | Spearphishing Attachment | Initial Access |
| T1566.002 | Spearphishing Link | Initial Access |
| T1036 | Masquerading (brand impersonation) | Defense Evasion |
| T1204 | User Execution | Execution |
| T1056 | Input Capture | Collection |
| T1078 | Valid Accounts | Persistence |
| T1027 | Obfuscated Files or Information | Defense Evasion |
| T1071.003 | Application Layer Protocol: Mail | Command and Control |

---

## 🔄 CI/CD (`.github/workflows/ci.yml`)

- **Trigger:** Push + PR to `main`
- **Python:** 3.13
- **Jobs:**
  1. Install dependencies from `requirements.txt`
  2. Smoke-import test — verifies all 13 source modules import without error
  3. Bandit security scan
- **No API keys in CI** — all intelligence and AI features degrade gracefully to `-1`

---

## 📐 Architecture Decisions

### Why 8 Layers?
Single-feature-type models (URL-only, text-only) are easily evaded. PhishLens requires
simultaneous evasion of header forensics + URL analysis + HTML analysis + semantic embeddings
+ threat intelligence + anomaly detection. This dramatically increases adversarial cost.

### Why SHAP + LIME Both?
SHAP (game-theoretic, exact) and LIME (model-agnostic approximation) provide independent explanations.
High agreement = trustworthy explanation. Low agreement = flag for manual review.

### Why Isolation Forest?
Supervised models only detect patterns seen in training data. Isolation Forest trained on
legitimate emails flags **any** structural anomaly — including novel campaigns targeting
brands not in the training corpus.

### Why an Ensemble of 5 Models?
Each model learns slightly different feature interactions. An adversary must simultaneously
craft an email that evades all five architectures — significantly harder than evading one.

---

## ⚠️ Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Training data staleness | ~6-month F1 decay estimated | Retrain quarterly |
| Targeted spear-phishing | May evade without matching training brand | ChatGPT AI + Isolation Forest |
| API rate limits | Intelligence enrichment throttled | Fallback to -1 (offline mode) |
| Encrypted content | S/MIME encrypted bodies unreadable | Header-only analysis |
| Image-only phishing | Minimal text features | HTML image ratio feature |
| Adversarial noise (5% Gaussian, all dims) | F1 drops 0.9505 → ~0.60 | Synthetic worst-case scenario only |
| `app.py` is empty | Streamlit UI not functional | Known gap — requires rebuild |

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sagar**
Portfolio project demonstrating ML security engineering skills — B.Sc. IT → MSc Cybersecurity.

> *"The best firewall is a good detector."*

