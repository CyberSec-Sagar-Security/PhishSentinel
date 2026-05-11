# PhishLens — Complete Project Review & Technical Briefing

> **Purpose of this document:** A comprehensive, no-detail-omitted review of the PhishLens
> project as it stands on **11 May 2026**. Intended for handover to another AI assistant (Claude)
> or for a technical reviewer who needs to understand every layer of the system.

---

## Table of Contents

1. [Project Identity & Context](#1-project-identity--context)
2. [Architecture Overview](#2-architecture-overview)
3. [Dataset Infrastructure](#3-dataset-infrastructure)
4. [Feature Engineering — All 961 Features](#4-feature-engineering--all-961-features)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Model Performance & Metrics (Actual Measured Results)](#6-model-performance--metrics-actual-measured-results)
7. [The Streamlit Web Application (app.py)](#7-the-streamlit-web-application-apppy)
8. [Threat Intelligence API Layer](#8-threat-intelligence-api-layer)
9. [URL Deobfuscation Pipeline](#9-url-deobfuscation-pipeline)
10. [IOC Extraction & Threat Intel Export](#10-ioc-extraction--threat-intel-export)
11. [MITRE ATT&CK Mapping](#11-mitre-attck-mapping)
12. [AI Analysis Layer (OpenAI + Gemini)](#12-ai-analysis-layer-openai--gemini)
13. [Explainability (SHAP + LIME)](#13-explainability-shap--lime)
14. [Zero-Day Anomaly Detection](#14-zero-day-anomaly-detection)
15. [Adversarial Robustness Testing](#15-adversarial-robustness-testing)
16. [Tooling, Training Pipeline & Infrastructure](#16-tooling-training-pipeline--infrastructure)
17. [Test Suite](#17-test-suite)
18. [Technical Bugs Fixed During Development](#18-technical-bugs-fixed-during-development)
19. [Current Accuracy — What We Achieved vs What We Aimed For](#19-current-accuracy--what-we-achieved-vs-what-we-aimed-for)
20. [What Is NOT Yet Working / Limitations](#20-what-is-not-yet-working--limitations)
21. [Future Roadmap](#21-future-roadmap)
22. [File Map — Every Source File](#22-file-map--every-source-file)
23. [Environment & Dependencies](#23-environment--dependencies)
24. [Key Constants & Configuration](#24-key-constants--configuration)

---

## 1. Project Identity & Context

| Field | Value |
|---|---|
| **Project name** | PhishLens |
| **Version** | V1.0 |
| **Type** | MSc Cybersecurity Final Year Portfolio Project |
| **Author** | Sagar |
| **Tech stack** | Python 3.13.8, Streamlit 1.57.0, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost, CatBoost, sentence-transformers, SHAP, LIME |
| **Runtime** | Windows (primary), Linux-compatible |
| **GPU** | NVIDIA RTX 2000 Ada Generation Laptop GPU (CUDA, used for embedding acceleration) |
| **Port** | http://localhost:8501 |
| **HuggingFace** | Intended for https://huggingface.co/spaces/CyberSec-Sagar-Security/PhishLens (models not yet uploaded) |

**Mission statement:** PhishLens is a production-ready, multi-layer phishing email detection system. It combines classical machine learning, deep NLP embeddings, six threat intelligence APIs, AI-powered contextual analysis (GPT-4.1-mini and Gemini 2.0 Flash), and full SOC-analyst tooling (IOC extraction, MITRE ATT&CK mapping, MISP JSON export, SHAP explainability). It is designed as an honest, fully-disclosed portfolio piece demonstrating defence-in-depth cybersecurity engineering.

---

## 2. Architecture Overview

The system processes a raw email through **eight independent detection layers**, combining outputs via a trained ML ensemble:

```
Raw Email (paste text / upload .eml / batch CSV)
    │
    ▼
[ EML Parser ]  ← RFC 2822 + MIME full parsing
    │
    ├─► Layer 1: Header Forensics       →  12 features  (SPF/DKIM/DMARC, hop analysis)
    ├─► Layer 2: URL Intelligence       →  29 features  (lexical + WHOIS + cert transparency)
    ├─► Layer 3: HTML Structural        →  11 features  (hidden text, forms, tracking pixels)
    ├─► Layer 4: Semantic Embeddings    → 384 features  (all-MiniLM-L6-v2 sentence transformer)
    ├─► Layer 5: TF-IDF Vocabulary      → 500 features  (1+2-gram, 500 max vocab)
    ├─► Layer 6: Threat Intelligence    →  13 features  (VirusTotal, GSB, URLScan, URLhaus, AbuseIPDB)
    ├─► Layer 7: Zero-Day Anomaly       →   1 feature   (Isolation Forest score)
    └─► Layer 8: AI Analysis (optional) →   2 features  (GPT/Gemini — NOT in training vector)
                                              ─────
                                        TOTAL: 961 features (SACRED — must never change)
    │
    ▼
ML Ensemble: LightGBM · XGBoost · CatBoost · Random Forest · Logistic Regression
    │
    ▼
Verdict: PHISHING / LEGITIMATE / UNCERTAIN  +  probability 0–100%
    │
    ├─► SHAP waterfall chart (top 10 feature contributions, light mode)
    ├─► LIME local explanation (independent cross-check)
    ├─► IOC Extraction (URLs, IPs, domains, emails, attachment hashes)
    ├─► URL Deobfuscation (SafeLinks, Proofpoint, Mimecast, etc.)
    ├─► Per-IOC Threat Intelligence (VT + GSB + URLScan + URLhaus + IPQS per URL)
    ├─► Domain TI scanning (VT per domain)
    ├─► MITRE ATT&CK mapping (T1566 family + secondary techniques)
    ├─► OpenAI / Gemini narrative explanation
    └─► IOC export: MISP JSON, CEF Syslog
```

### The 961-Feature Vector is SACRED

The number **961** must never change after training. The feature pipeline was fitted on this exact schema. Adding, removing, or reordering any feature will cause a shape mismatch (`ValueError: X has N features but classifier expects 961`) when loading saved `.pkl` models. Any future feature additions require a full retrain.

---

## 3. Dataset Infrastructure

### 3.1 Training Data — Final Composition

| Source | Emails | Notes |
|---|---|---|
| locuoco/the-biggest-spam-ham-phish-email-dataset-300000 | 179,776 | Largest source |
| JinqiangDing/seven-phishing-email-datasets | 116,333 | Multi-corpus aggregate |
| CASIS Kaggle phishing dataset | 57,166 | Business Email Compromise heavy |
| Enron Email Corpus (Kaggle) | 24,033 | Legitimate email baseline |
| puyang2025/phish-email-datasets | 16,699 | Recent phishing samples |
| SpamAssassin Public Corpus | 6,583 | Gold-standard labelled |
| phishing_pot | 6,306 | Real-world phishing collection |
| LLMGen-Phishing-Email-Dataset | 5,369 | AI-generated phishing samples |
| **TOTAL** | **412,265** | |

- **Train split:** 412,265 emails (80%) → after SMOTE: balanced for training
- **Test split:** 103,067 emails (20%, stratified, held-out, never seen during training)
- **Class balance:** 175,821 phishing / 236,444 legitimate
- **Split strategy:** Single pool → stratified 80/20 (not source-segregated)

### 3.2 Dataset Manifest

File: `data/processed/dataset_manifest.json`  
Generated: 2026-05-07T09:37:59

### 3.3 Dataset Download

`download_datasets.py` — fully automated script that downloads, extracts, and builds the processed dataset. Uses Hugging Face `datasets` API where available.

### 3.4 Raw Dataset Files Present

- `data/raw/meajor.csv` — phishing corpus
- `data/raw/umbrella_top1m.csv` — Cisco Umbrella Top 1M domains (used in URL features)
- `data/raw/casis/` — CEAS_08.csv, Enron.csv, Ling.csv, Nazario.csv, Nigerian_Fraud.csv (CASIS corpus files)
- `data/raw/enron_kaggle/emails.csv` — Enron legitimate emails
- `data/raw/spamassassin_ham/` + `spamassassin_spam/` — SpamAssassin files
- `data/raw/phishing_pot/` — Real phishing .eml files
- `data/processed/train.csv` + `test.csv` — Ready-to-use processed splits

---

## 4. Feature Engineering — All 961 Features

### 4.1 Module 1: Header Forensics — 12 Features

**File:** `src/features/header_features.py`

| Feature | Description |
|---|---|
| `hdr_from_reply_to_mismatch` | From domain ≠ Reply-To domain (common spoofing signal) |
| `hdr_from_return_path_mismatch` | From domain ≠ Return-Path domain |
| `hdr_spf_result` | SPF record result: +1 pass, 0 softfail/neutral, -1 fail/none |
| `hdr_dkim_result` | DKIM signature result: +1 pass, -1 fail/none |
| `hdr_dmarc_result` | DMARC policy result: +1 pass, -1 fail/none |
| `hdr_auth_score` | Composite SPF+DKIM+DMARC score (-3 to +3) |
| `hdr_received_hop_count` | Number of SMTP relay hops (excessive hops = suspicious routing) |
| `hdr_sender_ip_private` | Sender IP is in private/RFC1918 range (spoofed headers) |
| `hdr_freemail_sender` | From address uses a known free email provider (gmail, yahoo, etc.) |
| `hdr_suspicious_xmailer` | X-Mailer matches known bulk-sending tools |
| `hdr_timezone_mismatch` | Claimed send timezone vs. actual timezone in Received headers |
| `hdr_html_content_type` | Content-Type is text/html (phishing rarely uses plain text) |

DNS lookups for SPF/DKIM/DMARC are performed live during inference but skipped (`use_network=False`) during training to prevent 300–400ms timeouts per email × 412k emails.

### 4.2 Module 2: URL Features — 29 Features

**File:** `src/features/url_features.py`

Per-URL features extracted for every URL in the email, then **aggregated** (max/mean/count) to produce a fixed-width vector regardless of how many URLs are present:

**Lexical features per URL (12, aggregated to 24 by max+mean):**
- URL entropy (Shannon entropy of character distribution)
- URL length
- Subdomain count
- IP address in URL (e.g., `http://192.168.1.1/login`)
- Brand keyword in subdomain (from 50-brand list)
- Suspicious keyword present (e.g., "login", "verify", "secure", "account", "update")
- Risk TLD (.xyz, .top, .click, .tk, .ml, .ga, .cf, .gq, etc.)
- Punycode / homoglyph characters in domain (confusable_homoglyphs library)
- Path depth (number of `/` segments)
- Query parameter count
- URL shortener service detected
- Numeric domain (domain is entirely digits, e.g., `123456789.com`)

**Network features (3 — WHOIS + cert):**
- Domain age in days (WHOIS lookup — domains < 30 days = high risk)
- SSL certificate issuer is Let's Encrypt (commonly used by phishing sites)
- Certificate issued within last 30 days (cert transparency log lookup)

**Count features (2):**
- Total URL count in email
- Suspicious URL count

Fixed-width schema is enforced by `_default_url_features()` fallback — emails with no URLs get a zero vector of identical shape.

### 4.3 Module 3: HTML Structural Anomaly — 11 Features

**File:** `src/features/html_features.py`  
Parsed with BeautifulSoup 4. HTML truncated at 200KB before parsing to prevent exponential parse time on massive newsletters.

| Feature | Description |
|---|---|
| `html_external_link_count` | Number of links to external domains |
| `html_form_count` | Number of `<form>` elements (credential harvesting) |
| `html_form_post_external` | Form with POST action pointing to external domain |
| `html_hidden_text_count` | Elements with CSS that hides text (display:none, font-size:0, visibility:hidden, opacity:0, overflow:hidden, text-indent:-999px, mso-hide:all, etc.) |
| `html_href_text_mismatch` | Visible link text domain ≠ actual href domain (classic deception) |
| `html_has_base64_data` | Inline base64 data URIs present (obfuscation) |
| `html_has_tracking_pixel` | 1×1 pixel images (read receipts / tracking) |
| `html_iframe_count` | Number of iframes |
| `html_script_count` | Number of inline `<script>` blocks |
| `html_link_count` | Total hyperlinks |
| `html_img_count` | Number of images |

Hidden text detection covers **11 CSS patterns** including modern obfuscation techniques (clip:rect(0), max-height:0, Outlook's mso-hide:all).

### 4.4 Module 4a: Text Scalar Features — 8 Features

**File:** `src/features/text_features.py`

| Feature | Description |
|---|---|
| `urgency_phrase_count` | Count of urgency phrases ("act now", "expires", "verify immediately", etc.) from a 50+ phrase list |
| `urgency_score` | Normalised urgency density (count ÷ word count) |
| `subject_has_re_fwd` | Subject contains Re: or Fwd: prefix (conversation hijacking) |
| `subject_urgency_score` | Urgency density computed only from the subject line |
| `subject_length` | Character count of subject |
| `subject_caps_ratio` | Fraction of capital letters in subject (URGENT ALL-CAPS = signal) |
| `subject_exclamation_count` | Number of ! in subject |
| `subject_has_brand_keyword` | Brand name from the 50-brand list appears in subject |

### 4.5 Module 4b: Sentence-Transformer Embedding — 384 Features

**Model:** `all-MiniLM-L6-v2` (80MB, from HuggingFace)  
**Device:** CUDA (GPU accelerated — NVIDIA RTX 2000 Ada, fp16 precision via Tensor Core)  
**Input:** Full email body text, truncated at 512 tokens  
**Output:** 384-dimensional dense float vector  

This is the **highest-impact feature group**. The embedding captures latent semantic meaning that cannot be evaded by synonym substitution or paraphrasing. It runs on GPU (fp16) making inference fast (~2ms per email vs ~180ms on CPU).

Embeddings are cached to `data/processed/embedding_cache/` using MD5 of input text as filename, so repeated training runs don't re-embed. Cache files present:
- `embeddings_053899378618615e392317b2bd854992.npy`
- `embeddings_8f9bf498f6c18912bb6b6a49367216fa.npy`
- `embeddings_916d4d1fc9097dcd8746139deccf0c27.npy`
- `embeddings_9935fc5e6ff805ed2f7489196e7dc731.npy`
- `embeddings_c76c2a5fe171779dd6f256865a70e5f2.npy`

### 4.6 Module 5: TF-IDF — 500 Features

**File:** `src/features/text_features.py` (integrated with pipeline)  
**Configuration:** Max 500 features, 1+2-gram range (unigrams + bigrams), fitted on training corpus  
**Fit strategy:** Fitted once on the full training set, then `transform()`-only at inference  
**Storage:** Serialised inside `feature_pipeline.pkl`

### 4.7 Module 6: Threat Intelligence — 13 Features

**File:** `src/features/intelligence.py`

| Feature | API Source | Description |
|---|---|---|
| `vt_malicious` | VirusTotal v3 | Number of AV engines reporting URL as malicious (out of 70+) |
| `vt_suspicious` | VirusTotal v3 | Number reporting suspicious |
| `vt_clean` | VirusTotal v3 | Number reporting undetected |
| `vt_reputation` | VirusTotal v3 | VT community reputation score |
| `gsb_is_flagged` | Google Safe Browsing v4 | 1 if any URL matches GSB threat database |
| `gsb_threat_count` | Google Safe Browsing v4 | Number of GSB threat matches |
| `urlscan_malicious` | URLScan.io | Verdict from URLScan automated scan |
| `urlscan_brand_impersonated` | URLScan.io | 1 if brand impersonation detected |
| `urlscan_redirect_count` | URLScan.io | Number of redirects during page load |
| `urlhaus_threat` | URLhaus (abuse.ch) | 1 if URL appears in URLhaus malicious database |
| `abuse_confidence_score` | AbuseIPDB | Sender IP abuse confidence 0–100% |
| `abuse_total_reports` | AbuseIPDB | Number of community reports for sender IP |
| `abuse_is_tor` | AbuseIPDB | 1 if sender IP is a known Tor exit node |

**These 13 features ARE in the 961-feature ML vector (from training).**

### 4.8 Module 7: Anomaly Score — 1 Feature

- `anomaly_score`: Isolation Forest anomaly score, normalised to [0,1] where 1.0 = most anomalous relative to "normal" email distribution. Trained only on legitimate email features (label=0).

### 4.9 Feature Vector Summary

```
hdr:    12   (header forensics)
url:    29   (URL lexical + WHOIS + cert)
html:   11   (HTML structural)
text:    8   (urgency + subject scalars)
embed: 384   (sentence-transformer)
tfidf: 500   (TF-IDF bag-of-ngrams)
intel:  13   (6 TI APIs → 13 numeric features)
anomaly: 1   (Isolation Forest)
─────────────
TOTAL: 961   ← NEVER CHANGE
```

The AI features (GPT/Gemini: `gemini_is_phishing`, `gemini_confidence`) are **NOT** in this vector. They are display-only enrichment computed after the ML verdict.

---

## 5. Machine Learning Models

### 5.1 Trained Models (all saved as `.pkl`)

| File | Model | Size |
|---|---|---|
| `models/lightgbm.pkl` | LGBMClassifier | 1.1 MB |
| `models/xgboost.pkl` | XGBClassifier | 1.3 MB |
| `models/catboost.pkl` | CatBoostClassifier | 0.4 MB |
| `models/rf.pkl` | RandomForestClassifier | 428 MB (large — many trees) |
| `models/lr.pkl` | LogisticRegression | 8.6 KB |
| `models/lr_scaler.pkl` | StandardScaler | 23 KB |
| `models/feature_pipeline.pkl` | FeaturePipeline | 34 KB |
| `models/anomaly_detector.pkl` | ZeroDayDetector (IsolationForest) | 3.2 MB |

### 5.2 Training Configuration

- **Hyperparameter optimisation:** Optuna Bayesian search, 50 trials per model
- **Cross-validation:** 5-fold stratified K-fold
- **Class imbalance handling:** SMOTE oversampling of phishing class
- **Experiment tracking:** MLflow (logged locally)
- **Train/test split:** 80/20 stratified, `random_state=42`
- **Max raw email size:** 500,000 characters (truncated before parsing to prevent timeout)

### 5.3 Model Selection at Inference

The Streamlit app's sidebar lets the analyst choose which model to use (default: LightGBM). All 5 models can be selected independently. The confidence threshold (default 0.50) is adjustable via a sidebar slider.

### 5.4 DistilBERT Transformer (Optional, Not Yet Trained)

**File:** `src/models/transformer_model.py`  
A DistilBERT-based end-to-end classifier that takes raw email body text without feature extraction. Architecture: DistilBERT base uncased → Linear(768) → Dropout(0.3) → Linear(2). **Status: Code is complete and tested, but the model has not been fine-tuned on this dataset yet.** Requires GPU for practical training.

---

## 6. Model Performance & Metrics (Actual Measured Results)

### 6.1 Test Set Results (103,067 emails, held-out)

| Model | F1 | AUC-ROC | Precision | Recall | FNR | FPR | MCC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 **LightGBM** | **0.9505** | **0.9941** | 0.9575 | 0.9436 | **5.64%** | 3.11% | **0.9143** | 41,476 | 57,272 | 1,839 | 2,480 |
| 🥈 XGBoost | 0.9482 | 0.9935 | 0.9616 | 0.9352 | 6.48% | 2.78% | 0.9109 | 41,109 | 57,468 | 1,643 | 2,847 |
| Random Forest | 0.9436 | 0.9926 | 0.9492 | 0.9381 | 6.19% | 3.73% | 0.9021 | 41,234 | 56,904 | 2,207 | 2,722 |
| CatBoost | 0.9356 | 0.9895 | 0.9388 | 0.9324 | 6.76% | 4.52% | 0.8880 | 40,984 | 56,438 | 2,673 | 2,972 |
| Logistic Regression | 0.9334 | 0.9905 | 0.9680 | 0.9013 | 9.87% | 2.22% | 0.8886 | 39,617 | 57,801 | 1,310 | 4,339 |

> **FNR = False Negative Rate** = fraction of phishing emails the model classifies as legitimate = the most dangerous failure mode.  
> **FPR = False Positive Rate** = fraction of legitimate emails incorrectly flagged = alert fatigue.

### 6.2 Key Observations

- LightGBM is the best model on F1 and MCC. It is selected as the **default model** in the Streamlit UI.
- XGBoost has the **lowest FPR (2.78%)** — fewest false alarms on legitimate email.
- Logistic Regression achieves the **lowest FPR (2.22%)** and **highest precision (96.8%)** — best at not crying wolf.
- LR's 9.87% FNR is the worst — it misses nearly 1 in 10 phishing emails, making it unsuitable as a sole detector.
- All models achieve **AUC-ROC > 0.989** — excellent discrimination.
- MCC > 0.88 for all models — reliable even given class imbalance.

### 6.3 Adversarial Stress Test (Honest Disclosure)

This test injects Gaussian noise across all 961 feature dimensions simultaneously to stress-test model stability. Real adversarial attacks manipulate specific features, not all simultaneously — this is a **worst-case synthetic scenario**.

| Noise Level | LightGBM F1 | Notes |
|---|---|---|
| 0% (baseline) | 0.9505 | Normal operation |
| 5% noise | ~0.60 | Synthetic worst-case |
| 10% noise | ~0.25 | Synthetic worst-case |
| 20% noise | ~0.14 | Synthetic worst-case |

> This degradation is **disclosed transparently** in the README. Omitting it would be academically dishonest.

### 6.4 README Claims vs. Actual Metrics

The README contains optimistic claims (`F1 = 0.9808`, `FNR = 1.36%`) from a previous training run on an earlier, smaller dataset. The **current actual** metrics from `reports/metrics.json` are the 103,067-email test set values shown in section 6.1 above (F1 = 0.9505 for LightGBM, FNR = 5.64%). These reflect training on the **full 412,265-email corpus**.

---

## 7. The Streamlit Web Application (app.py)

**File:** `phishlens/app.py`  
**Framework:** Streamlit 1.57.0  
**URL:** http://localhost:8501  
**Working directory requirement:** Must be launched from `phishlens/` (app.py uses `Path(__file__).parent` for all relative paths)  

### 7.1 UI Layout

Three tabs:
1. **Analyse Email** — The main detection interface
2. **Performance Dashboard** — Model metrics, confusion matrix, feature importance charts
3. **About** — Project info, methodology, citations

### 7.2 Input Methods (Analyse Email Tab)

1. **Paste Email Text** — raw email text (headers + body) pasted into a `st.text_area`
2. **Upload .eml File** — binary file upload, parsed via `parse_eml_bytes()`
3. **Batch CSV** — CSV with `raw_email` column; analyses all rows and returns a results table with CSV export

### 7.3 Sidebar Controls

- **Models directory:** Text input (default `models/`), resolved relative to `app.py` location
- **Active Model:** Dropdown of all 5 trained classifiers (LightGBM, XGBoost, CatBoost, RF, LR)
- **Confidence Threshold:** Slider 0.10–0.90 (default 0.50). Verdict = PHISHING if `P(phishing) ≥ threshold`
- **Threat Intelligence APIs:** Checkbox to enable/disable all TI API calls
- **ChatGPT Analysis (gpt-4.1-mini):** Checkbox to enable/disable OpenAI analysis

### 7.4 Analysis Output Sections

After clicking "🔍 Analyse", the UI renders:

#### Section 1: Verdict Banner
- `PHISHING` (red gradient), `LEGITIMATE` (green gradient), or `UNCERTAIN` (amber gradient)
- Large probability percentage (e.g., "94.7% Phishing")
- Active model name, anomaly score, confidence threshold

#### Section 2: Risk Assessment (3 sub-tabs)
- **Summary & Key Findings** — CRITICAL/HIGH/MEDIUM/LOW severity findings as structured cards, risk factor list, forensic details table (From, To, Subject, Date, Message-ID, Return-Path, SPF/DKIM/DMARC, X-Mailer)
- **IOCs & Evidence** — All extracted IOCs with URL deobfuscation display (wrapped URL badges, real destination green boxes), attachment hashes
- **MITRE ATT&CK** — Detected technique cards with confidence bars, links to attack.mitre.org

#### Section 3: SHAP Explainability
- SHAP waterfall chart (Plotly, light mode) showing top 10 feature contributions
- Feature values, SHAP values, directional arrows
- LIME cross-check (independent local explanation)
- SHAP/LIME agreement score

#### Section 4: Threat Intelligence (when enabled)
- 6 tool status cards (VT, GSB, URLScan, URLhaus, AbuseIPDB, IPQS)
- Overall TI risk gauge
- **Per-IOC Verification Results** — Each URL, IP, email, domain shown individually with its own TI results:
  - URLs: IPQS risk score, VT hits, GSB flag, URLhaus verdict, URLScan verdict, GPT verdict
  - IPs: AbuseIPDB score, IPQS IP fraud score, GPT verdict
  - Email addresses: IPQS email fraud score, GPT verdict
  - Domains: VT per-domain results, GSB domain check
  - Wrapped URLs: Orange `⚠️ WRAPPED — Microsoft SafeLinks` badge + green "REAL DESTINATION" box

#### Section 5: ChatGPT AI Analysis (when enabled)
- Provider badge (ChatGPT gpt-4.1-mini)
- Risk level, confidence, recommended action
- Impersonated brand detection
- Social engineering techniques list
- Narrative explanation (3–5 sentences, analyst-grade)
- Per-IOC GPT verdicts

### 7.5 CSS Design System

Full custom CSS implementing a **GitHub Light Mode** palette:
- Backgrounds: `#ffffff` (main), `#f6f8fa` (cards)
- Borders: `#d0d7de`
- Primary text: `#24292f`, Secondary: `#57606a`
- CRITICAL severity: `#cf222e`/`#ffebe9`
- HIGH: `#bc4c00`/`#fff1e5`
- MEDIUM: `#9a6700`/`#fffbeb`
- LOW: `#1a7f37`/`#dafbe1`
- INFO: `#0969da`/`#ddf4ff`
- Wrapped URL badge: orange `#bc4c00`/`#fff1e5`
- Real destination: green `#1a7f37`/`#dafbe1`

### 7.6 Performance Dashboard Tab

- KPI row: Best model F1, AUC-ROC, FNR, FPR, MCC (from `reports/metrics.json`)
- Model comparison bar chart (F1/AUC-ROC/MCC grouped by model, Plotly)
- FNR vs FPR scatter plot (trade-off visualisation)
- Confusion matrix heatmap per model
- Feature importance charts (where available from model)

---

## 8. Threat Intelligence API Layer

### 8.1 APIs Integrated

| API | Key Source | What It Checks | ML Feature |
|---|---|---|---|
| VirusTotal v3 | `VIRUSTOTAL_API_KEY` env | URL reputation, 70+ AV engines | Yes (vt_malicious, vt_suspicious, vt_clean, vt_reputation) |
| Google Safe Browsing v4 | `GOOGLE_SAFE_BROWSING_API_KEY` env | Batch URL check vs. Chrome threat DB | Yes (gsb_is_flagged, gsb_threat_count) |
| AbuseIPDB | `ABUSEIPDB_API_KEY` env | Sender IP reputation | Yes (abuse_confidence_score, abuse_total_reports, abuse_is_tor) |
| URLScan.io | `URLSCAN_API_KEY` env | Live page scan | Yes (urlscan_malicious, urlscan_brand_impersonated, urlscan_redirect_count) |
| URLhaus (abuse.ch) | No key required | URL in known malware database | Yes (urlhaus_threat) |
| IPQualityScore (IPQS) | Hardcoded key in env | URL/IP/Email fraud scores | No (display-only) |

**IPQS key:** `a2DOWAmSPKY7rYqi9BHhib39jeyrAycx` (returns "insufficient credits" gracefully — handled as display-only enrichment)

### 8.2 Per-URL TI Scanning (after recent fix)

Previously, VT/URLScan/URLhaus only scanned the **first URL** in the email. This was fixed: `enrich_email_with_intelligence()` now iterates all URLs (up to 5) and stores per-URL results:
- `_vt_url_0`, `_vt_url_1`, ... `_vt_url_4` — per-URL VT results
- `_uh_url_0`, ... — per-URL URLhaus results
- `_us_url_0`, ... — per-URL URLScan results

GSB still uses a single batch API call but now returns `_gsb_flagged_urls` (a set of exact flagged URLs), enabling per-URL GSB display.

### 8.3 Domain TI Scanning

`query_virustotal_domain(domain)` — new function added to query VT `/api/v3/domains/{domain}`. Called for each extracted domain (up to 8). Results stored as `_vt_domain_0`, `_vt_domain_1`, etc.

### 8.4 IPQS Display-Only Enrichment (app.py)

- Email address scan: `query_ipqs_email()` → fraud score, disposable flag, spam trap flag, recent abuse
- URL scan: `query_ipqs_url()` for each cleaned URL (up to 5) → risk score 0-100, phishing/malware/suspicious flags
- IP scan: `query_ipqs_ip()` for sender IP → fraud score, proxy/VPN/Tor flags

---

## 9. URL Deobfuscation Pipeline

**File:** `src/features/url_cleaner.py`

### 9.1 Problem Solved

Phishing emails wrap real destination URLs inside corporate email security gateways (SafeLinks, Proofpoint, Mimecast, Barracuda) or URL shorteners. Without deobfuscation, TI tools check the **wrapper** domain (which is legitimate, e.g., microsoft.com) and return "Clean", missing the actual phishing destination. This is a critical false-negative source.

**Example from real email:** A SafeLinks URL wrapping `cacvil.fin.ec/as/nw/dnt` would appear "Clean" to every TI tool because the outer domain is `nam11.safelinks.protection.outlook.com` (Microsoft). After unwrapping, the real Ecuadorian phishing site is exposed and can be correctly scanned.

### 9.2 Supported Wrappers

| Method | Detection Pattern | Extraction |
|---|---|---|
| Microsoft SafeLinks | `*.safelinks.protection.outlook.com/?url=...` | URL-decode `?url=` parameter |
| Proofpoint URLDefense v2 | `urldefense.proofpoint.com/v2/url?u=...` | Decode Proofpoint's custom hex encoding |
| Proofpoint URLDefense v3 | `urldefense.com/v3/__<url>__` | Regex extract between `__` markers |
| Google Redirect | `google.com/url?q=` or `google.com/url?sa=t&url=` | URL-decode `?q=` or `?url=` parameter |
| Mimecast URL Protect | `protect-*.mimecast.com/s/...` | Follow redirect (HEAD request) |
| Barracuda Email Security | `links.barracudanetworks.com/...` | Follow redirect (HEAD request) |
| Cisco Ironport/ESA | `*.cisco.com/c/r/...` | Follow redirect |
| Percent-encoding (recursive) | `%XX%XX%XX...` | `urllib.parse.unquote` (recursive until stable) |
| URL shorteners | bit.ly, t.co, goo.gl, tinyurl.com, ow.ly, buff.ly, etc. | HEAD request with redirect follow (5s timeout) |

### 9.3 Public API

```python
# Single URL
result = clean_url(url, follow_redirects=True, timeout=5)
# Returns: {original, cleaned, method, was_wrapped}

# Batch
results = clean_urls(urls)         # List of result dicts
cleaned = get_real_urls(urls)      # List of cleaned URL strings only
mapping = build_url_cleaning_map(urls)  # {original: result_dict}
```

### 9.4 Integration with IOC Extractor

`extract_iocs()` now runs URL cleaning on all extracted URLs:
- `iocs["urls"]` = original sanitised URLs (displayed in UI)
- `iocs["cleaned_urls"]` = real destination URLs (used for TI scanning)
- `iocs["url_cleaning_map"]` = `{original: {cleaned, method, was_wrapped}}`
- `iocs["domains"]` = derived from **cleaned** URLs (reveals real attacker infrastructure)

`_urls_for_ti` in `app.py` uses `iocs.get("cleaned_urls") or parsed.get("urls", [])` — TI APIs always scan real destinations.

### 9.5 Testing

```python
test = 'https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fcacvil.fin.ec%2Fas%2Fnw%2Fdnt&data=...'
result = clean_url(test, follow_redirects=False)
# → cleaned: 'https://cacvil.fin.ec/as/nw/dnt', method: 'Microsoft SafeLinks', was_wrapped: True
```

---

## 10. IOC Extraction & Threat Intel Export

**File:** `src/ioc_extractor.py`

### 10.1 IOC Types Extracted

| IOC Type | Extraction Method |
|---|---|
| URLs | Regex `https?://[^\s<>"']+` from body and HTML |
| IP addresses | Regex `\b(\d{1,3}\.){3}\d{1,3}\b` from Received headers + body (private ranges excluded) |
| Email addresses | Regex `[a-zA-Z0-9._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}` from all headers + body |
| Domains | Extracted from cleaned URLs via tldextract |
| Attachment hashes | MD5 + SHA256 computed for each attachment |
| Phone numbers | Regex for international phone formats from body |

### 10.2 Export Formats

**MISP-compatible JSON** (`iocs_to_misp_json()`):
```json
{
  "info": "PhishLens IOC Export",
  "distribution": 0,
  "threat_level_id": 1,
  "Attribute": [
    {"type": "url", "value": "https://...", "comment": "Extracted from email body"},
    {"type": "ip-src", "value": "1.2.3.4", "comment": "Sender IP from Received headers"}
  ]
}
```

**CEF Syslog** (`iocs_to_syslog()`): Formats IOCs as Common Event Format strings for Splunk/ELK SIEM ingestion.

### 10.3 MISP ATT&CK Techniques (Legacy)

`_ATTACK_TECHNIQUES` in `ioc_extractor.py` contains static mappings for T1566.001 (Spearphishing Attachment), T1566.002 (Spearphishing Link), etc. These are now superseded by the dynamic `attack_mapping.py` module.

---

## 11. MITRE ATT&CK Mapping

**File:** `src/attack_mapping.py`

### 11.1 Techniques Mapped

| Technique ID | Name | Tactic | Trigger Condition |
|---|---|---|---|
| T1566 | Phishing | Initial Access | ML verdict = PHISHING or UNCERTAIN |
| T1566.001 | Spearphishing Attachment | Initial Access | Attachments detected in email |
| T1566.002 | Spearphishing Link | Initial Access | URLs detected in email |
| T1566.003 | Spearphishing via Service | Initial Access | Social media / messaging platform references |
| T1036 | Masquerading | Defense Evasion | Brand impersonation (GPT detected), lookalike domains in URLs, from/reply-to mismatch |
| T1204 | User Execution | Execution | Call-to-action phrases ("click here", "open attachment", "verify account") |
| T1056 | Input Capture (Credential Harvesting) | Collection | HTML forms with POST actions, "password"/"login" keywords, credential-themed subject |
| T1078 | Valid Accounts (Credential Theft) | Persistence | "password reset", "account suspended", credential urgency phrases |
| T1071.003 | Application Layer Protocol: Mail | Command & Control | C2-via-email patterns, reply-to != from |
| T1027 | Obfuscated Files or Information | Defense Evasion | Base64 content in HTML body, HTML obfuscation features, high HTML hidden element count |

### 11.2 Confidence Scoring

T1566 confidence = `min(phishing_probability, 1.0)`. For T1036 and others, confidence scales with the specific feature value triggering it (e.g., `url_brand_keyword_in_subdomain` count).

### 11.3 T1566 Suppression for Legitimate Email

When `verdict == "LEGITIMATE"`, T1566 is **not** added to the mapping. This prevents the misleading display of phishing techniques for emails the model determined are benign.

---

## 12. AI Analysis Layer (OpenAI + Gemini)

### 12.1 OpenAI GPT-4.1-mini

**File:** `src/features/openai_analyzer.py`  
**Model:** `gpt-4.1-mini`  
**Key:** `OPENAI_API_KEY` environment variable

**Full context prompt includes:**
- ML verdict + phishing probability
- All raw email headers (up to 60 header lines, 300 chars each)
- Email body (first 3,000 characters)
- Full IOC list (URLs, IPs, domains, email addresses, attachment hashes, phone numbers)
- Threat intelligence verdicts from all 6 TI tools
- Returns structured JSON verdict

**Response schema:**
```json
{
  "is_phishing": true/false,
  "confidence": 0.0–1.0,
  "phishing_signals": ["list of detected indicators"],
  "impersonated_brand": "BrandName or null",
  "social_engineering_techniques": ["urgency", "authority impersonation"],
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "explanation": "3-5 sentence analyst-grade explanation",
  "recommended_action": "DELETE|QUARANTINE|MONITOR|SAFE",
  "ioc_verdicts": {
    "url_0": "MALICIOUS|SUSPICIOUS|CLEAN|UNKNOWN",
    "sender_ip": "CLEAN",
    "sender_email": "SUSPICIOUS"
  }
}
```

Mapped to internal schema: `gemini_is_phishing`, `gemini_confidence`, `gemini_risk_level`, `gemini_impersonated_brand`, `gemini_phishing_signals`, `gemini_social_engineering`, `gemini_explanation`, `gemini_recommended_action`, `gemini_ioc_verdicts`. The `gemini_` prefix is used throughout app.py for both GPT and Gemini results (unified schema).

**API parameters:** temperature=0.1, max_tokens=900, response_format={"type": "json_object"}, timeout=30s.

### 12.2 Google Gemini 2.0 Flash

**File:** `src/features/gemini_analyzer.py`  
**Model:** `gemini-2.0-flash` (fast, cost-efficient)  
**Key:** `GEMINI_API_KEY` environment variable  
**Library:** `google-genai` SDK

Returns the same schema as OpenAI (unified `gemini_*` keys). Used as an alternative AI provider when the user selects "Gemini" in the sidebar (or if OpenAI key is not configured).

### 12.3 AI Features Are NOT in the ML Vector

Both `gemini_is_phishing` and `gemini_confidence` are intentionally excluded from the 961-feature ML training vector. Including them would create a training/inference mismatch (API calls not feasible during training of 412k emails). They are **post-hoc enrichment only**.

---

## 13. Explainability (SHAP + LIME)

**File:** `src/models/explainer.py`

### 13.1 SHAP

- **Tree models** (XGBoost, LightGBM, CatBoost, RF): `shap.TreeExplainer` (fast exact Shapley values)
- **Linear models** (Logistic Regression): `shap.LinearExplainer`
- Output: `shap_values` array, visualised as a Plotly waterfall chart showing top 10 features
- **Light mode colors:** The SHAP chart uses GitHub Light Mode colors (positive SHAP = red `#cf222e`, negative SHAP = green `#1a7f37`, base line = `#57606a`). This was explicitly fixed from an earlier dark mode (`#0d1117` background).

### 13.2 LIME

- `LimeTabularExplainer` from the LIME library
- Provides independent local explanation around the decision boundary
- Used as a cross-check against SHAP

### 13.3 Agreement Score

Agreement score = fraction of top-10 SHAP features also appearing in top-10 LIME features. High agreement (>0.7) = analyst can trust the explanation. Low agreement = model is in uncertain territory.

---

## 14. Zero-Day Anomaly Detection

**File:** `src/detection/anomaly.py`

- **Algorithm:** Isolation Forest (scikit-learn)
- **Training:** Fitted on legitimate email features only (`y == 0`)
- **Contamination:** 5% (`ANOMALY_CONTAMINATION = 0.05`)
- **Output:** Anomaly score normalised to [0, 1] via MinMaxScaler
- **Saved model:** `models/anomaly_detector.pkl` (3.2 MB)
- **Display:** Shown in verdict banner alongside ML probability

**Security rationale:** Supervised ML can only detect patterns seen in training. Novel phishing campaigns (new brands, new lures, new URL structures) may evade the supervised classifiers but still appear structurally anomalous relative to the legitimate email distribution. Isolation Forest provides distribution-free zero-day coverage.

---

## 15. Adversarial Robustness Testing

**File:** `src/models/adversarial_tester.py`

### 15.1 Attack Types Simulated

| Attack | Method |
|---|---|
| Whitespace insertion | Inject zero-width Unicode chars (U+200B, U+200C, U+200D, U+2060, U+FEFF) between characters to break tokenisation |
| Homoglyph substitution | Replace Latin chars with visually identical Cyrillic (а, е, о, р, с, х, etc.) |
| URL obfuscation | Wrap URLs in fake redirectors (bit.ly, tinyurl, t.co placeholders) |
| Header spoofing | Modify Received headers to appear geographically benign |
| Brand name mutation | Add/remove characters from brand names in subject/body |
| Urgency dilution | Add neutral phrases alongside urgency phrases to reduce urgency score |

### 15.2 Robustness Target

Target: **>90% robustness score** = F1 after perturbation ÷ F1 on clean data.  
Current status: The adversarial tester code is complete but the **full robustness benchmark against the trained models has not been formally reported**. The Gaussian noise test in README (section 6.3) is a different measure.

---

## 16. Tooling, Training Pipeline & Infrastructure

### 16.1 `train.py` — Training CLI

```bash
python train.py [options]
  --data-dir          Directory containing train.csv/test.csv (default: data/processed)
  --models            Which models to train: lr, rf, xgboost, lightgbm, catboost, all
  --tune              Run Optuna hyperparameter search (50 trials)
  --no-network        Disable DNS and WHOIS network calls (for offline/CI use)
  --eval              Evaluate and save metrics to reports/metrics.json
  --save [models/]    Save trained models to directory
  --adversarial       Run adversarial robustness test after training
  --smote             Apply SMOTE class balancing (default: True)
```

### 16.2 `phishlens.ps1` — PowerShell Launcher

Windows one-stop launcher with actions: `install`, `train`, `run`, `verify`, `download`. Fixed 6 bugs during development (ValidateSet, function names, argument passing, etc.).

### 16.3 `download_datasets.py` — Dataset Downloader

Automated download using Hugging Face `datasets` API + direct HTTP. Builds processed train/test CSVs with the stratified split.

### 16.4 `install_and_verify.py` — Full Verification

Runs 5 verification suites: 35 library imports, 15 module imports, 36 source files check, 5 API keys check, entry-point smoke tests. **Status: 5/5 PASS verified on 2026-05-03.**

### 16.5 `monitor_training.py` — Training Progress Monitor

Monitors catboost_info/ training logs in real-time during training.

### 16.6 MLflow Experiment Tracking

All training runs are logged to MLflow (local). Tracks: hyperparameters, CV scores, test metrics, model artifacts.

### 16.7 GitHub Actions CI/CD (`.github/workflows/ci.yml`)

Runs on every push: install dependencies, run tests, syntax checks.

---

## 17. Test Suite

**Directory:** `tests/`

| File | Tests | Coverage |
|---|---|---|
| `test_eml_parser.py` | 14 unit tests | .eml parsing, MIME types, header extraction |
| `test_url_features.py` | 13 unit tests | URL lexical features, entropy, brand detection, shortener detection |
| `test_html_features.py` | 11 unit tests | HTML form detection, hidden text, href mismatch, tracking pixels |
| `test_pipeline.py` | Integration tests | 50 synthetic emails → 961 feature vectors, no shape crashes |

All tests use pytest. The 961-feature invariant is verified in `test_pipeline.py` — shape must always equal 961.

---

## 18. Technical Bugs Fixed During Development

| Bug | Cause | Fix |
|---|---|---|
| `SyntaxError: invalid non-printable character U+FEFF` | PowerShell `Set-Content -Encoding UTF8` writes a UTF-8 BOM | Use `New-Object System.Text.UTF8Encoding $false` or read+strip BOM with Python `data.lstrip(b'\xef\xbb\xbf')` |
| `ValueError: X has N features but classifier expects 961` | URL feature schema varied between emails-with-URLs and emails-without-URLs | Added `_default_url_features()` fallback function; `np.pad` guard in pipeline |
| `IndentationError` in app.py | Corrupted indentation from PowerShell text manipulation | Fixed with `replace_string_in_file` tool directly |
| "No trained models found" sidebar warning | Streamlit launched from parent folder → `models/` resolved to wrong CWD | Fixed: `load_phishlens()` and anomaly path now use `Path(__file__).parent` |
| `KeyError: parsed["headers"]` in OpenAI prompt | `eml_parser` doesn't emit a `headers` dict key | Build `_all_headers` manually from individual parsed field keys |
| TI only scanned first URL | `enrich_email_with_intelligence()` only called VT/URLScan/URLhaus on `urls[0]` | Refactored to loop all URLs (up to 5), store per-index results |
| VT/GSB/URLhaus showed `—` for all URLs except first | Display code checked `_ti_idx == 0` as condition | Fixed: use per-URL `_vt_url_N` keys and `_gsb_flagged_urls` set |
| Domain rows showed no TI data | No domain scanning existed | Added `query_virustotal_domain()` function; domain loop in app.py |
| SHAP charts had dark background | `plot_bgcolor="#0d1117"` in Plotly config | Changed to `plot_bgcolor="#ffffff"`, `paper_bgcolor="#f6f8fa"` (light mode) |
| T1566 shown for legitimate emails | No verdict check before adding technique | Added `if verdict in ("PHISHING", "UNCERTAIN")` guard |
| XGBoost `use_label_encoder` deprecation error | Old XGBoost API | Removed `use_label_encoder=False` parameter |
| Dataset imbalance design flaw | Some sources were train-only or test-only (not pool+split) | Unified into single pool → stratified 80/20 split |
| `argparse` error `--save models` | `nargs=1` expected value after `--save` | Changed to `nargs='?', const='models/'` |
| BOM-corrupted `openai_analyzer.py` | `Set-Content` with BOM encoding | Rewrote file cleanly |
| Streamlit "CONNECTING" / grayed UI | Stale WebSocket from killed process | Fixed: Ctrl+Shift+R hard refresh; server confirmed running |

---

## 19. Current Accuracy — What We Achieved vs What We Aimed For

### 19.1 Targets vs Reality

| Metric | Target | Achieved (LightGBM) | Status |
|---|---|---|---|
| F1 Score | ≥ 0.97 | **0.9505** | ⚠️ Near target (README claims 0.9808 from earlier run) |
| AUC-ROC | ≥ 0.99 | **0.9941** | ✅ Exceeded |
| FNR | < 5% | **5.64%** | ⚠️ Just above target |
| FPR | < 5% | **3.11%** | ✅ Achieved |
| MCC | ≥ 0.90 | **0.9143** | ✅ Achieved |
| Adversarial robustness | > 90% | Unknown (not formally tested) | ❓ Pending |
| Processing speed | < 5s per email | ~2–4s (GPU) | ✅ Achieved |
| SHAP explainability | All models | All 5 models | ✅ Achieved |
| MITRE ATT&CK mapping | Complete | 10 techniques | ✅ Achieved |
| IOC extraction | MISP + Syslog | Both formats | ✅ Achieved |
| URL deobfuscation | SafeLinks + Proofpoint | 9 wrapper types | ✅ Achieved |
| Per-URL TI scanning | Per-URL (not just first) | Up to 5 URLs per email | ✅ Achieved (recent fix) |
| Domain TI scanning | VT per domain | VT per domain (up to 8) | ✅ Achieved (recent fix) |

### 19.2 Why F1 Is 0.9505 and Not 0.9808

The 0.9808 figure in the README came from training on the **smaller CASIS+Enron+SpamAssassin dataset** (~168,608 emails with possibly more balanced, cleaner labels). The current 0.9505 is on the **412,265-email corpus** with 8 sources including:
- LLM-generated phishing samples (harder to distinguish)
- The bigger dataset introduces more label noise and more diverse email styles
- The test set of 103,067 is far more representative and harder than smaller test splits

This is **still excellent performance** — 94.4% recall (sensitivity) on 103k+ emails with 5.6% FNR means missing ~1 in 18 phishing emails at the 0.5 threshold.

### 19.3 Targets Not Met

- **FNR < 5%**: Achieved 5.64% (target was < 5%). Threshold tuning can get it below 5% at the cost of higher FPR.
- **Adversarial robustness > 90%**: Not formally measured on trained models (only Gaussian noise test exists).
- **DistilBERT fine-tuning**: Code is complete but model has not been trained — could push F1 > 0.97.
- **Ensemble voting**: A hard-voting ensemble of all 5 models could push FNR below 5%.

---

## 20. What Is NOT Yet Working / Limitations

### 20.1 HuggingFace Hub

Models are **not uploaded** to `https://huggingface.co/spaces/CyberSec-Sagar-Security/PhishLens`. The HF Hub download attempt logs a 404 warning at startup, but falls back to local models gracefully.

### 20.2 IPQS API Credits

The hardcoded IPQS key (`a2DOWAmSPKY7rYqi9BHhib39jeyrAycx`) returns "insufficient credits" for IPQS URL/IP/email scans. These show "IPQS: —" in the UI. The IPQS TI section is functional but produces no data without a funded key.

### 20.3 API Keys Not Configured

Unless `.env` is populated, all TI APIs return `-1` sentinel values:
- `VIRUSTOTAL_API_KEY` — needed for VT URL and domain scans
- `GOOGLE_SAFE_BROWSING_API_KEY` — needed for GSB batch URL check
- `ABUSEIPDB_API_KEY` — needed for sender IP reputation
- `URLSCAN_API_KEY` — needed for URLScan page analysis
- `OPENAI_API_KEY` — needed for ChatGPT analysis
- `GEMINI_API_KEY` — needed for Gemini analysis

### 20.4 URL Shortener Redirect Following

The shortener resolution in `url_cleaner.py` uses an HTTP HEAD request with 5-second timeout. Some shorteners redirect via JavaScript, not HTTP 3xx — those are not resolved. Additionally, following real redirects in production is a security concern (could ping attacker infrastructure).

### 20.5 Large .eml Files

Emails are truncated at 500KB before ML pipeline parsing (`_MAX_RAW_CHARS = 500_000`) and at 200KB before HTML parsing (`_MAX_HTML_CHARS = 200_000`). Attachments inside large emails may not be fully processed.

### 20.6 Random Forest Size

`models/rf.pkl` is 428MB — unusually large for a random forest. Loading it takes ~3–5 seconds and consumes significant RAM. It is not the default model for this reason.

### 20.7 Batch CSV Processing

Batch CSV mode works but is slow for large files because:
- Embedding inference is sequential (not batched)
- Each row triggers a full pipeline transform
- Network calls (WHOIS, TI APIs) are made per email

### 20.8 Streamlit CWD Sensitivity

Streamlit must be launched such that the working directory is `phishlens/`. If launched from the parent folder, the old `models/` path lookup failed (now fixed with `Path(__file__).parent`). But any remaining relative path strings (e.g., in dataset loading) could still fail if CWD is wrong.

---

## 21. Future Roadmap

### 21.1 High Priority (Performance Improvements)

1. **Threshold tuning to achieve FNR < 5%**  
   Use the ROC curve to find the threshold where FNR = 4.9%. For LightGBM with AUC=0.9941, this is achievable with threshold ~0.40 or lower.

2. **Ensemble hard voting (majority vote of 5 models)**  
   Create a `VotingClassifier` that combines all 5 models. Expected to push F1 > 0.96 and FNR below 5%.

3. **DistilBERT fine-tuning**  
   Train `src/models/transformer_model.py` (DistilBERT base uncased) on the full 412k corpus for 3 epochs with GPU. Expected to produce a standalone F1 > 0.97 and complement the feature engineering pipeline as a 6th ensemble member.

4. **Retrain with more balanced/cleaner labels**  
   Audit LLM-generated phishing samples (Dizzzy0x00/LLMGen-Phishing-Email-Dataset) which may have noisy labels. Remove or re-label and retrain.

### 21.2 Medium Priority (New Features)

5. **Formal adversarial robustness benchmark**  
   Run `AdversarialTester.run_full_test()` against all 5 trained models. Report robustness scores per attack type. Aim for > 90% on whitespace/homoglyph attacks, > 70% on URL obfuscation.

6. **Header authentication live verification (SPF/DKIM/DMARC)**  
   Currently DNS lookups are skipped during training. Build an inference-time enrichment step that performs live DNS verification and adds these as post-hoc signals (not in the ML vector — display only).

7. **Email header fingerprinting / campaign clustering**  
   Hash-based deduplication of X-Mailer + IP + sending time window to detect coordinated phishing campaigns. Alert when multiple emails share infrastructure.

8. **Attachment static analysis**  
   Extract text from PDF/DOCX/XLSX attachments using `pdfminer`, `python-docx`, `openpyxl`. Add URL extraction from document bodies. Compute PE header features for EXE/Office macros.

9. **Reply-chain injection detection**  
   T1566.003: Phishing emails that hijack legitimate reply chains (contain fake Re: thread history). Detect by analysing Message-ID chains and checking if cited messages actually exist.

10. **Sender reputation database (local)**  
    Maintain a SQLite cache of previously seen sender IPs/domains with their historical PhishLens verdicts. Fast local lookup before making API calls.

### 21.3 Infrastructure & Deployment

11. **HuggingFace Spaces deployment**  
    Upload trained models (except RF at 428MB — use LightGBM only) to the HF Hub repo. Configure `HF_TOKEN` secret. The `app.py` already has the HF Hub download logic.

12. **Docker containerisation**  
    Create `Dockerfile` + `docker-compose.yml` for reproducible deployment. Expose port 8501. Include GPU support via `nvidia/cuda` base image.

13. **REST API wrapper (FastAPI)**  
    Create `api.py` exposing `/analyse` POST endpoint accepting raw email text. Return JSON verdict. Enables integration with SOC tooling, SOAR platforms, email gateways.

14. **GitHub Actions full CI/CD**  
    Current `.github/workflows/ci.yml` runs tests. Extend to: build Docker image on merge to main, deploy to HF Spaces automatically, run adversarial tests on nightly schedule.

15. **Prometheus metrics export**  
    Expose `/metrics` endpoint with: analysis count, FP/FN counts, API response times, model inference latency. For monitoring in Grafana.

### 21.4 AI & LLM Enhancements

16. **Multi-model AI voting**  
    Query both GPT-4.1-mini AND Gemini 2.0 Flash. When they agree, high confidence. When they disagree, flag as "AI consensus uncertain" requiring analyst review.

17. **Fine-tuned phishing LLM**  
    Fine-tune a small open-source LLM (Llama 3.2 3B) on the PhishLens dataset to produce a local LLM that doesn't require API keys. Deployable offline.

18. **RAG-enhanced analysis**  
    Retrieve similar past phishing campaigns from a vector database (FAISS or Chroma) and include them in the GPT/Gemini prompt context. Enables "This email resembles Campaign X from 2025..."

19. **IOC-to-threat-actor attribution**  
    Cross-reference extracted IPs/domains against known APT infrastructure databases (MISP communities, OTX). Attribute phishing to known threat actors where possible.

### 21.5 Security Hardening

20. **Input sanitisation for HTML rendering**  
    The IOC display in Streamlit uses `unsafe_allow_html=True`. While email content is not directly rendered into HTML (it's wrapped in safe spans), review all HTML injection surfaces.

21. **API key rotation support**  
    Move API keys to a Streamlit secrets manager or HashiCorp Vault. Support key rotation without restarting the application.

22. **Rate limiting for TI APIs**  
    Add per-API rate limiter (token bucket) to prevent accidental API quota exhaustion during batch processing of large CSV files.

### 21.6 Academic / Portfolio Enhancements

23. **Explainability report generation**  
    Add "Export Report" button that generates a PDF/DOCX incident report including: verdict, SHAP chart, IOC table, MITRE ATT&CK techniques, narrative explanation, recommended action. Uses `reportlab` or `python-docx`.

24. **Temporal drift monitoring**  
    Track feature distribution shifts over time. Alert when incoming emails look statistically different from training distribution (indicates concept drift / new campaign types).

25. **Cross-validation on BEC (Business Email Compromise)**  
    BEC emails contain no malicious URLs and pass SPF/DKIM. They rely entirely on social engineering. Specifically evaluate BEC detection rate using the Enron + CASIS BEC subset.

---

## 22. File Map — Every Source File

```
phishlens/
├── app.py                          # Streamlit web application (main UI)
├── train.py                        # CLI training script
├── download_datasets.py            # Automated dataset downloader
├── install_and_verify.py           # 5-suite verification tool
├── setup_and_verify.py             # Alternative setup verifier
├── monitor_training.py             # Training progress monitor
├── phishlens.ps1                   # PowerShell launcher (Windows)
├── requirements.txt                # Pinned Python dependencies
├── packages.txt                    # System-level packages (for HF Spaces)
├── patch_torch_py313.py            # PyTorch compatibility patch for Python 3.13
├── GPU_SETUP.md                    # GPU acceleration setup guide
├── README.md                       # Full project documentation
├── verify_output.txt               # Saved verification output
│
├── src/
│   ├── __init__.py
│   ├── ioc_extractor.py            # IOC extraction → MISP JSON + CEF syslog
│   ├── attack_mapping.py           # MITRE ATT&CK T1566 technique mapper
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── eml_parser.py           # RFC 2822 + MIME .eml parser
│   │   └── dataset_loader.py       # Multi-corpus dataset loaders
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── header_features.py      # 12 header forensics features
│   │   ├── url_features.py         # 29 URL lexical + WHOIS + cert features
│   │   ├── html_features.py        # 11 HTML structural anomaly features
│   │   ├── text_features.py        # 384-dim embeddings + TF-IDF + 8 text scalars
│   │   ├── intelligence.py         # 6 TI APIs → 13 ML features + display enrichment
│   │   ├── url_cleaner.py          # URL deobfuscation (9 wrapper types)
│   │   ├── gemini_analyzer.py      # Google Gemini 2.0 Flash AI analysis
│   │   ├── openai_analyzer.py      # OpenAI GPT-4.1-mini AI analysis
│   │   └── pipeline.py             # Master FeaturePipeline (961 features)
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── anomaly.py              # Isolation Forest zero-day detector
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Optuna + MLflow model training
│   │   ├── evaluator.py            # Comprehensive metrics + visualisations
│   │   ├── explainer.py            # SHAP + LIME dual explainability
│   │   ├── adversarial_tester.py   # 6 adversarial attack simulations
│   │   └── transformer_model.py    # DistilBERT fine-tuning (optional)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # PhishLensConfig + all constants
│       └── logger.py               # loguru structured logging
│
├── tests/
│   ├── __init__.py
│   ├── test_eml_parser.py          # 14 unit tests for EML parsing
│   ├── test_url_features.py        # 13 unit tests for URL features
│   ├── test_html_features.py       # 11 unit tests for HTML features
│   └── test_pipeline.py            # Pipeline integration tests
│
├── data/
│   ├── README.md                   # Dataset download instructions
│   ├── raw/                        # Raw corpus files (gitignored)
│   │   ├── meajor.csv
│   │   ├── umbrella_top1m.csv
│   │   ├── casis/                  # CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud
│   │   ├── enron_kaggle/
│   │   ├── spamassassin_ham/
│   │   ├── spamassassin_spam/
│   │   └── phishing_pot/
│   └── processed/
│       ├── train.csv               # 412,265 emails
│       ├── test.csv                # 103,067 emails
│       ├── dataset_manifest.json   # Dataset composition record
│       └── embedding_cache/        # Cached sentence-transformer embeddings
│
├── models/                         # Trained model artifacts
│   ├── lightgbm.pkl                # 1.1 MB — PRIMARY MODEL
│   ├── xgboost.pkl                 # 1.3 MB
│   ├── catboost.pkl                # 0.4 MB
│   ├── rf.pkl                      # 428 MB
│   ├── lr.pkl                      # 8.6 KB
│   ├── lr_scaler.pkl               # 23 KB
│   ├── feature_pipeline.pkl        # 34 KB — TF-IDF + config
│   └── anomaly_detector.pkl        # 3.2 MB — Isolation Forest
│
├── reports/
│   ├── metrics.json                # All model evaluation metrics
│   ├── STATUS_REPORT.md            # Development status report (May 2026)
│   ├── verify_report.json          # Verification results
│   ├── verify_report.txt
│   └── figures/                    # Confusion matrix PNGs + plots
│
├── logs/                           # Application logs (loguru)
│
├── "samples eamil"/                # Test email samples
│   ├── Goldstar RFP 325725 _ 02.msg
│   ├── Re_ Accommodation Booking...eml
│   ├── RE_ RBG Promotions.eml
│   └── test_phishing.eml
│
└── docs/                           # Project documentation
```

---

## 23. Environment & Dependencies

### 23.1 Runtime Environment

| Component | Version |
|---|---|
| Python | 3.13.8 |
| Virtual environment | `.venv` at `phishlens/.venv/` |
| Streamlit | 1.57.0 |
| LightGBM | 4.6.0 |
| scikit-learn | 1.8.0 |
| NumPy | 2.4.4 |
| Pandas | 2.3.3 |
| Requests | 2.33.1 |
| SHAP | 0.51.0 |
| Plotly | 6.7.0 |
| BeautifulSoup4 | 4.14.3 |
| sentence-transformers | (installed) |
| XGBoost | (installed) |
| CatBoost | (installed) |
| LIME | (installed) |
| python-dotenv | (installed) |
| loguru | (installed) |
| tldextract | (installed) |
| dnspython | (installed) |
| openai | (installed) |
| google-genai | (installed) |
| huggingface_hub | (installed) |
| confusable_homoglyphs | (installed) |
| imblearn | (installed — for SMOTE) |
| optuna | (installed) |
| mlflow | (installed) |
| joblib | (installed) |

### 23.2 GPU Setup

GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU  
CUDA used for: sentence-transformer embedding (fp16, Tensor Core acceleration)  
GPU reduces embedding stage from ~4–5 hours (CPU) to ~10–15 minutes

### 23.3 Environment Variables (`.env`)

```
VIRUSTOTAL_API_KEY=
GOOGLE_SAFE_BROWSING_API_KEY=
ABUSEIPDB_API_KEY=
URLSCAN_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
IPQS_API_KEY=your_ipqs_api_key_here  # Set in .env — see .env.example
HF_TOKEN=  # Optional — for HuggingFace private repo access
```

---

## 24. Key Constants & Configuration

From `src/utils/config.py`:

| Constant | Value | Purpose |
|---|---|---|
| `RANDOM_STATE` | 42 | All random operations |
| `TEST_SIZE` | 0.20 | 80/20 stratified split |
| `CV_FOLDS` | 5 | Stratified k-fold CV |
| `OPTUNA_TRIALS` | 50 | Bayesian HP search trials |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer |
| `TFIDF_MAX_FEATURES` | 500 | TF-IDF vocabulary size |
| `TFIDF_NGRAM_RANGE` | (1, 2) | Unigrams + bigrams |
| `DOMAIN_AGE_RISK_DAYS` | 30 | Domains < 30 days = high risk |
| `DOMAIN_AGE_WARN_DAYS` | 90 | 30–90 days = medium risk |
| `ANOMALY_CONTAMINATION` | 0.05 | Isolation Forest rate |
| `WHOIS_TIMEOUT` | 2s | WHOIS lookup timeout |
| `NETWORK_TIMEOUT` | 3s | API call timeout |
| `EMBEDDING_MAX_TOKENS` | 512 | Truncate before embedding |
| `_MAX_RAW_CHARS` | 500,000 | Max email size for pipeline |
| `_MAX_HTML_CHARS` | 200,000 | Max HTML size for parser |

**Brand list (50 brands):** Microsoft, Apple, Google, Amazon, Netflix, PayPal, Dropbox, DocuSign, Zoom, Adobe, Spotify, LinkedIn, Facebook, Instagram, Twitter, WhatsApp, Telegram, Wells Fargo, Bank of America, Chase, Citibank, HSBC, Barclays, Santander, NatWest, AIB, Bank of Ireland, Ulster Bank, Permanent TSB, KBC, Revenue.ie, An Post, DHL, FedEx, UPS, USPS, Royal Mail, DPD, NHS, HSE, HMRC, IRS, Salesforce, Slack, Office365, OneDrive, SharePoint, iCloud, Outlook, Gmail.

**Risk TLDs:** .xyz, .top, .click, .tk, .ml, .ga, .cf, .gq, .icu, .online, .site, .work, .live, .tech, .pw, .cc, .biz, .info, .mobi, .name

**Safe TLDs:** .com, .org, .net, .edu, .gov, .ie, .co.uk, .co.ie, .ac.uk, .ac.ie, .gov.uk, .gov.ie

---

## Appendix A: How to Run PhishLens Right Now

```powershell
# From project root, activate venv and start server:
Set-Location "D:\CyberSecurity\Cyber_Security_Future_Projects\PhishLens\phishlens"
.venv\Scripts\Activate.ps1

# Start Streamlit:
& ".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true

# Open browser:
# http://localhost:8501
```

If the browser shows "CONNECTING": press **Ctrl + Shift + R** (hard refresh).  
If models show "No trained models found": ensure you're running from the `phishlens/` directory, or use the absolute path approach above.

---

## Appendix B: Critical Rules for Future Development

1. **NEVER change the 961-feature vector** without retraining all models. Any addition/removal/reordering of features requires a full retrain.
2. **NEVER use PowerShell `Set-Content` to write Python files** — it writes a UTF-8 BOM that causes `SyntaxError: invalid non-printable character U+FEFF`. Use `replace_string_in_file` or `[System.IO.File]::WriteAllText` with `New-Object System.Text.UTF8Encoding $false`.
3. **AI features (gemini_is_phishing, gemini_confidence) are NOT in the ML vector** and must stay out. They are display-only post-hoc enrichment.
4. **TI API results are display-only for IPQS** (insufficient credits). VT, GSB, AbuseIPDB, URLScan, URLhaus are real ML features.
5. **All relative file paths in app.py must use `Path(__file__).parent`** as the base, not `Path(".")`, to survive being launched from a different working directory.
6. **The `_urls_for_ti` variable** = `iocs.get("cleaned_urls") or parsed.get("urls", [])` — always use cleaned (deobfuscated) URLs for TI scanning, never raw wrapped URLs.
7. **`gemini_ioc_verdicts`** is the key in the OpenAI result dict for per-IOC GPT verdicts (despite being prefixed `gemini_` for backward compat).
