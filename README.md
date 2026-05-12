---
title: PhishSentinel
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🛡️ PhishSentinel — ML Phishing Email Detection System

[![CI](https://github.com/CyberSec-Sagar-Security/PhishSentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/CyberSec-Sagar-Security/PhishSentinel/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://bandit.readthedocs.io/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/SagarTony90265/PhishSentinel)

> **MSc Cybersecurity Portfolio Project — B.Sc. IT → MSc Cybersecurity**
> A production-ready, multi-layer phishing email detection system combining
> classical ML, deep NLP embeddings, threat intelligence APIs, and ChatGPT AI.
>
> **🚀 Live Demo:** https://huggingface.co/spaces/SagarTony90265/PhishSentinel
> **GitHub:** https://github.com/CyberSec-Sagar-Security/PhishSentinel &nbsp;|&nbsp; **Python:** 3.13 &nbsp;|&nbsp; **License:** MIT

---

## 🎯 Live Demo

> **✅ PhishSentinel is live on HuggingFace Spaces!**
> Try it now → **https://huggingface.co/spaces/SagarTony90265/PhishSentinel**
>
> Upload a `.eml` / `.msg` file, paste raw email headers, or use the built-in sample emails to see the detection pipeline in action — no installation required.

<img width="1853" height="913" alt="Demo " src="https://github.com/user-attachments/assets/1017cd3b-1865-40db-838d-3a1c79e85277" />
<img width="1656" height="966" alt="image" src="https://github.com/user-attachments/assets/3ab1f6fa-5792-42e9-8334-78fd7c1191e1" />
<img width="1660" height="1016" alt="image" src="https://github.com/user-attachments/assets/74ad91b8-e007-4d0c-ae0a-0b0241b3d381" />
<img width="1591" height="867" alt="image" src="https://github.com/user-attachments/assets/fe212f30-0b35-4bf4-acab-73b91c04617a" />



*PhishSentinel catching a professionally crafted phishing simulation email —
ML probability 37.3%, escalated to PHISHING via VirusTotal (1 malicious hit)
and ChatGPT AI (HIGH risk, 85% confidence). Microsoft SafeLinks URLs
deobfuscated to reveal true malicious destination.*

---

## 📋 What It Does

PhishSentinel detects phishing emails using a **defence-in-depth ML architecture** — eight
independent detection layers that an adversary must simultaneously evade to deliver a
phishing email undetected. When the ML model is uncertain, external threat intelligence
APIs and ChatGPT AI provide independent confirmation — catching sophisticated attacks
that evade any single detection method.

**Key capabilities:**
- 🔍 Analyses `.eml` files, pasted raw email text, or batch CSV files
- 🔗 Deobfuscates wrapped URLs (Microsoft SafeLinks, Proofpoint, Mimecast — 9 wrapper types)
- 🧠 Explains every verdict with SHAP + LIME dual explainability and agreement scoring
- 🚨 Maps detections to MITRE ATT&CK techniques (T1566 family + 6 secondary techniques)
- 📤 Exports IOCs as MISP JSON and CEF Syslog for direct SOC integration
- 🤖 Provides ChatGPT gpt-4.1-mini narrative analysis with per-IOC verdicts
- 🛡️ Escalates uncertain ML verdicts when threat intelligence confirms risk

---

## 🏗️ Architecture — 8-Layer Defence-in-Depth

```
Raw Email (.eml / paste / batch CSV)
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  Layer 1  │  Header Forensics          │  header_features.py    │  12 features │
│  Layer 2  │  URL Intelligence          │  url_features.py       │  29 features │
│  Layer 3  │  HTML Structural Analysis  │  html_features.py      │  11 features │
│  Layer 4  │  Semantic Embeddings       │  text_features.py      │ 384 features │
│  Layer 5  │  TF-IDF Vocabulary Signals │  text_features.py      │ 500 features │
│  Layer 6  │  Threat Intelligence APIs  │  intelligence.py       │  13 features │
│  Layer 7  │  Isolation Forest Anomaly  │  anomaly.py            │   1 feature  │
│  Layer 8  │  ChatGPT AI (optional)     │  openai_analyzer.py    │   2 features │
└───────────────────────────────────────────────────────────────────────────────┘
                     TOTAL: 961 features — fixed vector, never change without retraining
    │
    ▼
ML Ensemble → LightGBM · XGBoost · CatBoost · Random Forest · Logistic Regression
    │
    ▼
URL Deobfuscation → SafeLinks · Proofpoint · Mimecast · 9 wrapper types
    │
    ▼
Verdict: PHISHING / LEGITIMATE / UNCERTAIN  +  Confidence %
    │
    ├── SHAP + LIME Explainability (dual, with agreement score)
    ├── MITRE ATT&CK Mapping (10 techniques, confidence-scored)
    ├── IOC Extraction (IPs · URLs · domains · attachment hashes)
    ├── ChatGPT AI Narrative (optional, requires OPENAI_API_KEY)
    └── IOC Export: MISP JSON · CEF Syslog · Plain Text Report
```

---

## 🔢 Feature Vector — 961 Dimensions (Fixed Schema)

| # | Layer | Module | What Is Extracted | Count |
|---|---|---|---|---|
| 1 | Header Forensics | `header_features.py` | SPF/DKIM/DMARC pass/fail, From↔Reply-To mismatch, From↔Return-Path mismatch, freemail sender, timezone anomaly, X-Mailer bulk fingerprint, hop count, relay IP count, display name spoofing | **12** |
| 2 | URL Intelligence | `url_features.py` | Lexical analysis (length, entropy, special chars, TLD risk, subdomain depth, IP-in-URL, brand-in-subdomain), WHOIS domain age, cert transparency, homoglyph/punycode detection — per-URL aggregations | **29** |
| 3 | HTML Structure | `html_features.py` | Hidden form fields, external form actions, tracking pixels, image-only ratio, script obfuscation, data URI count, meta refresh, hidden iFrame, link text mismatch, comment ratio | **11** |
| 4 | Semantic Embeddings | `text_features.py` | `all-MiniLM-L6-v2` 384-dimensional sentence embeddings + 8 urgency/subject scalar signals | **384** |
| 5 | TF-IDF Vocabulary | `text_features.py` | 500-token phishing vocabulary n-gram signal (unigrams + bigrams) | **500** |
| 6 | Threat Intelligence | `intelligence.py` | VirusTotal malicious/suspicious counts, GSB threat type, AbuseIPDB confidence score, URLScan verdict, URLhaus status, IPQS fraud score | **13** |
| 7 | Anomaly Detection | `anomaly.py` | Isolation Forest score trained on legitimate emails only — detects zero-day campaigns not seen in training | **1** |
| 8 | ChatGPT AI | `openai_analyzer.py` | `is_phishing` (0/1), `confidence` (0.0–1.0) — keys retain `gemini_` prefix for trained model weight compatibility (historical artifact from Gemini → GPT migration) | **2** |

> **Layer 8 note:** During training all 412,265 emails were processed with `--no-network`.
> Layer 8 features are always `-1` during training. At inference time, if `OPENAI_API_KEY`
> is configured, real ChatGPT values replace the `-1` defaults.

---

## 🏆 Model Performance

> All metrics on the **held-out test set of 103,067 emails** — never seen during training.
> Stratified 80/20 split from 412,265-email corpus across 8 dataset sources.

| Rank | Model | F1 | AUC-ROC | FNR ↓ | FPR | MCC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **LightGBM** *(default)* | **0.9505** | **0.9941** | **5.64%** | 3.11% | **0.9143** | 41,476 | 57,272 | 1,839 | 2,480 |
| 🥈 2 | XGBoost | 0.9482 | 0.9935 | 6.48% | 2.78% | 0.9109 | 41,109 | 57,468 | 1,643 | 2,847 |
| 🥉 3 | Random Forest | 0.9436 | 0.9926 | 6.19% | 3.73% | 0.9021 | 41,234 | 56,904 | 2,207 | 2,722 |
| 4 | CatBoost | 0.9356 | 0.9895 | 6.76% | 4.52% | 0.8880 | 40,984 | 56,438 | 2,673 | 2,972 |
| 5 | Logistic Regression | 0.9334 | 0.9905 | 9.87% | 2.22% | 0.8886 | 39,617 | 57,801 | 1,310 | 4,339 |

> **FNR = False Negative Rate** — phishing emails missed. Lower is better for a security tool.
> LightGBM missed **2,480 out of 43,956** phishing emails in the held-out test set.
> Logistic Regression at 93.34% F1 — near the top — demonstrates that
> **feature engineering quality is the primary driver**, not model complexity.

### ⚠️ Adversarial Stress Test — Honest Disclosure

LightGBM tested with Gaussian noise injected simultaneously across all 961 feature dimensions:

| Noise Level | F1 | Context |
|---|---|---|
| 0% (baseline) | 0.9505 | Normal operation |
| 5% noise | ~0.60 | Synthetic worst-case — all 961 dims perturbed simultaneously |
| 10% noise | ~0.25 | Synthetic worst-case |
| 20% noise | ~0.14 | Synthetic worst-case |

> This is a **synthetic scenario** — real adversarial attacks manipulate specific features
> while leaving most others unchanged. Disclosed for full transparency.

---

## 🔗 URL Deobfuscation

Many phishing emails wrap malicious URLs inside corporate email security gateways.
Without deobfuscation, TI tools check the **wrapper** domain (e.g. `microsoft.com`)
and return **Clean** — completely missing the real malicious destination.

PhishSentinel automatically unwraps before scanning:

| Wrapper Type | Detection | Extraction |
|---|---|---|
| Microsoft SafeLinks | `*.safelinks.protection.outlook.com/?url=` | URL-decode `?url=` parameter |
| Proofpoint URLDefense v2 | `urldefense.proofpoint.com/v2/url?u=` | Custom hex decode |
| Proofpoint URLDefense v3 | `urldefense.com/v3/__<url>__` | Regex between `__` markers |
| Google Redirect | `google.com/url?q=` | URL-decode `?q=` parameter |
| Mimecast URL Protect | `protect-*.mimecast.com/s/` | Follow HTTP redirect |
| Barracuda Email Security | `links.barracudanetworks.com/` | Follow HTTP redirect |
| Cisco IronPort/ESA | `*.cisco.com/c/r/` | Follow HTTP redirect |
| URL Shorteners | bit.ly, t.co, tinyurl.com, ow.ly, etc. | HEAD request with redirect |
| Percent-encoding | `%XX%XX...` | Recursive `urllib.parse.unquote` |

TI APIs always scan the **real destination** — never the wrapper domain.

---

## 🔺 Verdict Escalation

When ML probability is below threshold (UNCERTAIN), PhishSentinel escalates to
**PHISHING** based on external evidence:

- VirusTotal: ≥1 malicious engine detection on any extracted URL
- Google Safe Browsing: any URL flagged as phishing or malware
- ChatGPT AI: verdict = phishing with confidence > 0.70

This catches sophisticated emails crafted to fool ML classifiers while still leaving
detectable traces in threat intelligence databases or AI analysis.

---

## 📊 Training Datasets

| Source | Emails | Type | Notes |
|---|---|---|---|
| locuoco/biggest-spam-ham-phish | 179,776 | Phishing + Spam + Ham | Largest single source |
| JinqiangDing/seven-phishing-email-datasets | 116,333 | Phishing | Multi-corpus, updated Apr 2026 |
| CASIS (Kaggle) | 57,166 | Phishing + BEC | Business Email Compromise heavy |
| Enron Email Corpus (Kaggle) | 24,033 | Legitimate | Corporate email baseline |
| puyang2025/phish-email-datasets | 16,699 | Phishing | Recent 2025 campaigns |
| SpamAssassin Public Corpus | 6,583 | Legitimate + Spam | Gold-standard labelled |
| phishing_pot | 6,306 | Phishing | Real collected phishing .eml files |
| Dizzzy0x00/LLMGen-Phishing | 5,369 | Phishing | **AI-generated** phishing (Dec 2025) |
| **TOTAL** | **412,265** | | After deduplication |

> The LLMGen dataset enables detection of phishing emails crafted by ChatGPT and
> similar LLMs — an attack vector that 2022-era datasets do not cover.

---

## 📂 Project Structure

```
PhishSentinel/
├── app.py                              # Streamlit web interface (localhost:8501)
├── train.py                            # Training CLI
├── download_datasets.py                # Automated dataset downloader
├── install_and_verify.py               # Dependency + environment verification
├── requirements.txt                    # Pinned dependencies
├── README.md
├── GPU_SETUP.md                        # CUDA setup + performance benchmarks
├── .env.example                        # API key template (copy → .env)
├── .gitignore
│
├── src/
│   ├── ioc_extractor.py                # IOC extraction → MISP JSON + CEF Syslog
│   ├── attack_mapping.py               # MITRE ATT&CK T1566 technique mapper
│   ├── ingestion/
│   │   ├── eml_parser.py               # RFC 2822 + MIME .eml parser
│   │   └── dataset_loader.py           # Multi-source dataset loaders
│   ├── features/
│   │   ├── header_features.py          # Layer 1 — 12 header forensics features
│   │   ├── url_features.py             # Layer 2 — 29 URL lexical + WHOIS features
│   │   ├── url_cleaner.py              # URL deobfuscation (9 wrapper types)
│   │   ├── html_features.py            # Layer 3 — 11 HTML structural features
│   │   ├── text_features.py            # Layers 4+5 — embeddings + TF-IDF
│   │   ├── intelligence.py             # Layer 6 — 6 TI API integrations
│   │   ├── openai_analyzer.py          # Layer 8 — ChatGPT gpt-4.1-mini analysis
│   │   └── pipeline.py                 # Master FeaturePipeline (961 features)
│   ├── detection/
│   │   └── anomaly.py                  # Layer 7 — Isolation Forest zero-day detector
│   ├── models/
│   │   ├── trainer.py                  # Optuna + MLflow training
│   │   ├── evaluator.py                # Metrics, confusion matrix, stress test
│   │   ├── explainer.py                # SHAP + LIME dual explainability
│   │   ├── adversarial_tester.py       # Adversarial attack simulations
│   │   └── transformer_model.py        # DistilBERT fine-tuning (optional)
│   └── utils/
│       ├── config.py                   # PhishSentinelConfig dataclass
│       └── logger.py                   # loguru structured logging
│
├── tests/
│   ├── test_eml_parser.py              # 14 unit tests
│   ├── test_url_features.py            # 13 unit tests
│   ├── test_html_features.py           # 11 unit tests
│   └── test_pipeline.py               # Integration tests (50 synthetic emails)
│
├── data/
│   ├── raw/                            # Raw datasets (gitignored)
│   └── README.md                       # Dataset download instructions
│
├── models/                             # Trained .pkl artifacts (gitignored)
│
├── reports/
│   ├── metrics.json                    # Full per-model evaluation metrics
│   └── figures/                        # Confusion matrix PNGs
│
└── .github/workflows/ci.yml            # GitHub Actions CI/CD
```

> **Model files** are excluded from GitHub (`rf.pkl` = 428 MB exceeds GitHub's 100 MB limit).
> Run `python train.py` after cloning to regenerate all model artifacts locally.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.13+
- 16 GB RAM recommended
- NVIDIA GPU strongly recommended — reduces embedding stage from ~4–5 hours (CPU) to ~15 minutes
- See [GPU_SETUP.md](GPU_SETUP.md) for CUDA installation

### Installation

```bash
git clone https://github.com/CyberSec-Sagar-Security/PhishSentinel.git
cd PhishSentinel

python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/macOS

pip install -r requirements.txt

copy .env.example .env              # Windows
# cp .env.example .env              # Linux/macOS
# Edit .env and add your API keys
```

### Verify Installation

```bash
python install_and_verify.py --verify-only
# Expected: 5/5 checks pass
```

### Train

```bash
python download_datasets.py

# Fast — XGBoost only:
python train.py --data-dir data/processed --models xgboost --no-network --eval --save models

# Full — all 5 models + Optuna tuning:
python train.py --data-dir data/processed --models all --tune --no-network --eval --save models
```

### Launch

```bash
streamlit run app.py
# Open: http://localhost:8501
```

---

## ⚙️ ML Training Details

| Setting | Value |
|---|---|
| Hyperparameter search | Optuna, 50 trials per model |
| Cross-validation | 5-fold stratified K-Fold |
| Class balancing | SMOTE oversampling |
| Experiment tracking | MLflow |
| Embedding model | `all-MiniLM-L6-v2` (80 MB) |
| TF-IDF | max_features=500, ngram_range=(1,2) |
| Default threshold | 0.50 |
| GPU training time | ~45–60 min / 412K emails (RTX 2000 Ada, 8 GB VRAM) |
| CPU training time | ~8–12 hours / 412K emails |

---

## 🔌 API Key Configuration

```env
OPENAI_API_KEY=your_key        # platform.openai.com/api-keys
VIRUSTOTAL_API_KEY=your_key    # virustotal.com/gui/join-us (500 req/day free)
GOOGLE_SAFE_BROWSING_API_KEY=your_key  # console.cloud.google.com (free)
ABUSEIPDB_API_KEY=your_key     # abuseipdb.com/register (1,000 req/day free)
URLSCAN_API_KEY=your_key       # urlscan.io/user/signup (free tier)
IPQS_API_KEY=your_key          # ipqualityscore.com/create-account (free tier)
HF_TOKEN=your_token            # huggingface.co/settings/tokens (optional)
```

All features fall back gracefully to `-1` when keys are absent or rate-limited.

---

## 🎯 MITRE ATT&CK Coverage

| Technique ID | Name | Tactic |
|---|---|---|
| T1566 | Phishing | Initial Access |
| T1566.001 | Spearphishing Attachment | Initial Access |
| T1566.002 | Spearphishing Link | Initial Access |
| T1566.003 | Spearphishing via Service | Initial Access |
| T1036 | Masquerading | Defense Evasion |
| T1204 | User Execution | Execution |
| T1056 | Input Capture | Collection |
| T1078 | Valid Accounts | Persistence |
| T1027 | Obfuscated Files or Information | Defense Evasion |
| T1071.003 | Application Layer Protocol: Mail | Command and Control |

---

## 🔄 CI/CD

**Trigger:** Push + PR to `main` | **Python:** 3.13

1. Install dependencies
2. Smoke-import all 13 source modules
3. Bandit security scan
4. pytest test suite (38+ tests)

No API keys required in CI — all enrichment features degrade gracefully.

---

## ⚠️ Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Training data temporal gap | ~6-month F1 decay on newest attacks | Retrain quarterly |
| Modern HTML-rich legitimate email | Elevated false positive rate | More modern legitimate data needed |
| Targeted spear-phishing | May evade if brand not in training | ChatGPT AI + Isolation Forest |
| API rate limits | Intelligence enrichment throttled | Graceful fallback to -1 |
| Encrypted content (S/MIME) | Cannot analyse encrypted bodies | Header-only analysis |
| Image-only phishing | Minimal text/URL features | HTML image ratio feature |
| rf.pkl (428 MB) | Not deployable to HF Spaces free tier | Use LightGBM as primary (1.1 MB) |
| Adversarial noise (5%, all dims) | F1 drops 0.9505 → ~0.60 | Synthetic worst-case only |

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sagar B. Suryawanshi**
B.Sc. Information Technology → MSc Cybersecurity (Ireland)
Targeting SOC Analyst and Application Security roles in Ireland and the EU.

[![GitHub](https://img.shields.io/badge/GitHub-CyberSec--Sagar--Security-black?logo=github)](https://github.com/CyberSec-Sagar-Security?tab=repositories)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/sagar--suryawanshi/)

> *"The best firewall is a good detector."*
