"""
PhishLens Central Configuration Module.

All hyperparameter grids, feature engineering thresholds, brand lists,
risk TLD lists, and API endpoint constants are centralised here.
Modify this file — not scattered magic numbers — to tune PhishLens behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Dataset & Training Constants
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20       # 80/20 stratified split
CV_FOLDS: int = 5             # Stratified k-fold cross-validation
OPTUNA_TRIALS: int = 50       # Bayesian hyperparameter search trials
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # 80MB; load once via cache
TFIDF_MAX_FEATURES: int = 500
TFIDF_NGRAM_RANGE: tuple = (1, 2)


# ---------------------------------------------------------------------------
# Feature Engineering Thresholds
# ---------------------------------------------------------------------------

DOMAIN_AGE_RISK_DAYS: int = 30       # Domains < 30 days old → risk score 1.0
DOMAIN_AGE_WARN_DAYS: int = 90       # 30–90 days → risk score 0.5
MIN_URL_ENTROPY: float = 3.5         # Below this = low-entropy, likely benign
CERT_LETS_ENCRYPT_RISK: float = 0.6  # LE cert alone is not conclusive
ANOMALY_CONTAMINATION: float = 0.05  # Isolation Forest contamination rate
WHOIS_TIMEOUT: int = 2               # Seconds before WHOIS fallback to -1
NETWORK_TIMEOUT: int = 3             # Seconds for crt.sh / API calls
EMBEDDING_MAX_TOKENS: int = 512      # Truncate body before embedding


# ---------------------------------------------------------------------------
# Top 50 Spoofed Brands (used in brand impersonation feature)
# ---------------------------------------------------------------------------

BRAND_LIST: List[str] = [
    # Global tech & e-commerce
    "microsoft", "apple", "google", "amazon", "netflix", "paypal",
    "dropbox", "docusign", "zoom", "adobe", "spotify", "linkedin",
    "facebook", "instagram", "twitter", "whatsapp", "telegram",
    # Financial — global
    "wellsfargo", "bankofamerica", "chase", "citibank", "hsbc",
    "barclays", "santander", "natwest",
    # Financial — Ireland-specific (SOC relevance for Irish roles)
    "aib", "bankofi", "bankofireland", "ulsterbank", "permanenttsb",
    "kbc", "revenuecie", "revenue", "anpost",
    # Logistics & shipping
    "dhl", "fedex", "ups", "usps", "anpost", "royalmail", "dpd",
    # Healthcare & government
    "nhs", "hse", "hmrc", "irs", "gov",
    # Cloud / SaaS
    "salesforce", "slack", "office365", "onedrive", "sharepoint",
    "icloud", "outlook", "gmail",
]

# ---------------------------------------------------------------------------
# Risk TLD List
# ---------------------------------------------------------------------------

RISK_TLD_LIST: List[str] = [
    ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf",
    ".gq", ".icu", ".online", ".site", ".work", ".live", ".tech",
    ".pw", ".cc", ".biz", ".info", ".mobi", ".name",
]

SAFE_TLD_LIST: List[str] = [
    ".com", ".org", ".net", ".edu", ".gov", ".ie", ".co.uk",
    ".co.ie", ".ac.uk", ".ac.ie", ".gov.uk", ".gov.ie",
]

# ---------------------------------------------------------------------------
# Known URL Shortener Domains
# ---------------------------------------------------------------------------

URL_SHORTENER_DOMAINS: List[str] = [
    "bit.ly", "tinyurl.com", "ow.ly", "t.co", "goo.gl", "rebrand.ly",
    "buff.ly", "adf.ly", "short.link", "cutt.ly", "is.gd", "v.gd",
    "tiny.cc", "bl.ink", "soo.gd", "s2r.co", "clck.ru", "tr.im",
]

# ---------------------------------------------------------------------------
# Suspicious URL Keywords
# ---------------------------------------------------------------------------

SUSPICIOUS_URL_KEYWORDS: List[str] = [
    "login", "verify", "secure", "update", "confirm", "account",
    "banking", "webscr", "cmd=", "token=", "password", "credential",
    "signin", "logon", "auth", "reset", "recover", "unlock",
    "suspended", "validate", "authorize",
]

# ---------------------------------------------------------------------------
# Urgency / Social-Engineering Phrases (body text scoring)
# ---------------------------------------------------------------------------

URGENCY_PHRASES: List[str] = [
    "urgent", "immediately", "verify now", "account suspended",
    "unusual activity", "click here", "limited time", "you have been selected",
    "congratulations", "security alert", "action required",
    "your account will be closed", "within 24 hours", "expires today",
    "final notice", "important notice", "immediate action",
    "your password has been compromised", "verify your identity",
    "update your information", "confirm your details",
    # Modern phishing lures:
    "sign-in attempt", "signin attempt", "unusual sign-in",
    "one-time password", "one time password", "enter your otp",
    "delivery failed", "parcel could not be delivered", "package on hold",
    "claim your prize", "you have won", "you are a winner",
    "invoice attached", "payment overdue", "payment declined",
    "your account has been locked", "we detected suspicious",
    "confirm your email", "validate your account",
    "your subscription has expired", "reactivate your account",
    "refund pending", "tax refund", "wire transfer",
]

# ---------------------------------------------------------------------------
# Suspicious X-Mailer strings (bulk-sender fingerprints)
# ---------------------------------------------------------------------------

SUSPICIOUS_XMAILER_PATTERNS: List[str] = [
    # Confirmed bulk / mass-mailing tools — safe to flag
    "phpmailer", "sendblaster", "gmass", "massmailer", "bulkmail",
    # ESP platforms sometimes abused for phishing delivery
    "mailchimp", "sendgrid", "brevo", "constantcontact",
    # Note: 'postfix' / 'exim' are MTA names, NOT suspicious on their own.
    # Note: do NOT include "" or "unknown" — empty X-Mailer is NORMAL for
    #       Gmail, Outlook Web, Apple Mail, and Yahoo Mail.
]

# ---------------------------------------------------------------------------
# Known-Abuse Registrars (domain registration risk signal)
# ---------------------------------------------------------------------------

ABUSE_REGISTRARS: List[str] = [
    "namecheap", "godaddy", "tucows", "public domain registry",
    "pdricann", "internet domain service", "alibaba cloud",
    "west263", "bizcn", "hichina",
]

# ---------------------------------------------------------------------------
# Freemail Domains (reply-to freemail = social engineering signal)
# ---------------------------------------------------------------------------

FREEMAIL_DOMAINS: List[str] = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "protonmail.com", "mail.com", "icloud.com",
    "yandex.com", "tutanota.com", "gmx.com", "live.com",
]

# ---------------------------------------------------------------------------
# MITRE ATT&CK Technique Mapping
# ---------------------------------------------------------------------------

ATTACK_TECHNIQUE_MAP: Dict[str, Dict[str, str]] = {
    # Keys = exact feature names from the ML feature vector
    "url_url_shortener_max": {
        "technique_id": "T1566.002",
        "technique_name": "Phishing: Spearphishing Link",
        "tactic": "Initial Access",
    },
    "url_domain_age_risk": {
        "technique_id": "T1583.001",
        "technique_name": "Acquire Infrastructure: Domains",
        "tactic": "Resource Development",
    },
    "url_punycode_detected_max": {
        "technique_id": "T1036.007",
        "technique_name": "Masquerading: Double File Extension",
        "tactic": "Defense Evasion",
    },
    "html_external_form_action": {
        "technique_id": "T1056.003",
        "technique_name": "Input Capture: Web Portal Capture",
        "tactic": "Collection",
    },
    "url_cert_brand_mismatch": {
        "technique_id": "T1036",
        "technique_name": "Masquerading",
        "tactic": "Defense Evasion",
    },
    "hdr_received_geo_anomaly": {
        "technique_id": "T1071.003",
        "technique_name": "Application Layer Protocol: Mail Protocols",
        "tactic": "Command and Control",
    },
    "url_is_ip_address_max": {
        "technique_id": "T1583.005",
        "technique_name": "Acquire Infrastructure: Botnet",
        "tactic": "Resource Development",
    },
    "hdr_from_reply_to_mismatch": {
        "technique_id": "T1656",
        "technique_name": "Impersonation",
        "tactic": "Defense Evasion",
    },
}

# ---------------------------------------------------------------------------
# Model Hyperparameter Grids (Optuna search space)
# ---------------------------------------------------------------------------

XGBOOST_PARAM_GRID: Dict[str, Any] = {
    "learning_rate": (0.01, 0.3),      # log-uniform
    "max_depth": (3, 10),              # int
    "n_estimators": (100, 1000),       # int
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),       # int
    "gamma": (0.0, 0.5),
}

RF_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": (100, 500),        # int
    "max_depth": (5, 30),              # int; None = unlimited → use high int
    "min_samples_split": (2, 20),      # int
    "max_features": ["sqrt", "log2"],
}

LR_PARAM_GRID: Dict[str, Any] = {
    "C": (0.001, 100.0),               # log-uniform
    "solver": ["lbfgs", "saga"],
    "max_iter": [500, 1000, 2000],
}

CATBOOST_PARAM_GRID: Dict[str, Any] = {
    "iterations": (200, 1000),
    "learning_rate": (0.01, 0.3),
    "depth": (4, 10),
    "l2_leaf_reg": (1, 10),
}

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

API_ENDPOINTS: Dict[str, str] = {
    "virustotal_url": "https://www.virustotal.com/api/v3/urls/{url_id}",
    "virustotal_submit": "https://www.virustotal.com/api/v3/urls",
    "google_safe_browsing": "https://safebrowsing.googleapis.com/v4/threatMatches:find",
    "abuseipdb_check": "https://api.abuseipdb.com/api/v2/check",
    "urlscan_search": "https://urlscan.io/api/v1/search/",
    "urlscan_submit": "https://urlscan.io/api/v1/scan/",
    "urlscan_result": "https://urlscan.io/api/v1/result/{uuid}/",
    "crtsh": "https://crt.sh/?q={domain}&output=json",
    "urlhaus_lookup": "https://urlhaus-api.abuse.ch/v1/url/",
}

# ---------------------------------------------------------------------------
# Config Dataclass (passed around the codebase)
# ---------------------------------------------------------------------------


@dataclass
class PhishLensConfig:
    """Central configuration object for PhishLens.

    Instantiate once and pass to all modules that need thresholds or params.
    """

    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    optuna_trials: int = OPTUNA_TRIALS
    embedding_model: str = EMBEDDING_MODEL
    tfidf_max_features: int = TFIDF_MAX_FEATURES
    tfidf_ngram_range: tuple = field(default_factory=lambda: TFIDF_NGRAM_RANGE)
    domain_age_risk_days: int = DOMAIN_AGE_RISK_DAYS
    domain_age_warn_days: int = DOMAIN_AGE_WARN_DAYS
    min_url_entropy: float = MIN_URL_ENTROPY
    anomaly_contamination: float = ANOMALY_CONTAMINATION
    whois_timeout: int = WHOIS_TIMEOUT
    network_timeout: int = NETWORK_TIMEOUT
    embedding_max_tokens: int = EMBEDDING_MAX_TOKENS
    brand_list: List[str] = field(default_factory=lambda: BRAND_LIST)
    risk_tld_list: List[str] = field(default_factory=lambda: RISK_TLD_LIST)
    safe_tld_list: List[str] = field(default_factory=lambda: SAFE_TLD_LIST)
    url_shortener_domains: List[str] = field(default_factory=lambda: URL_SHORTENER_DOMAINS)
    suspicious_url_keywords: List[str] = field(default_factory=lambda: SUSPICIOUS_URL_KEYWORDS)
    urgency_phrases: List[str] = field(default_factory=lambda: URGENCY_PHRASES)
    freemail_domains: List[str] = field(default_factory=lambda: FREEMAIL_DOMAINS)
    abuse_registrars: List[str] = field(default_factory=lambda: ABUSE_REGISTRARS)
    prediction_threshold: float = 0.5

    def __repr__(self) -> str:
        return (
            f"PhishLensConfig("
            f"random_state={self.random_state}, "
            f"cv_folds={self.cv_folds}, "
            f"embedding_model='{self.embedding_model}', "
            f"optuna_trials={self.optuna_trials})"
        )


# Singleton default config — import this directly where no customisation needed
DEFAULT_CONFIG = PhishLensConfig()
