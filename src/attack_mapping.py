"""
PhishLens MITRE ATT&CK Technique Mapper.

Maps detected phishing indicators to MITRE ATT&CK Enterprise framework
techniques. This transforms PhishLens output from a binary verdict into
structured threat intelligence that maps to the adversary kill chain.

Primary technique: T1566 — Phishing (all phishing emails)
Sub-techniques:
  - T1566.001 — Spearphishing Attachment (emails with malicious attachments)
  - T1566.002 — Spearphishing Link (emails with malicious URLs)
  - T1566.003 — Spearphishing via Service (via social media / messaging)

Secondary techniques (based on detected features):
  - T1036 — Masquerading (brand impersonation, lookalike domains)
  - T1204 — User Execution (calls to action: click link, open attachment)
  - T1056 — Input Capture (credential harvesting forms)
  - T1078 — Valid Accounts (credential theft)
  - T1071.003 — Application Layer Protocol: Mail (email C2 communications)
  - T1027 — Obfuscated Files or Information (base64 content, HTML obfuscation)

Security rationale: ATT&CK mapping enables:
  1. Automated threat classification for SOC triage
  2. Integration with threat intelligence platforms (MISP, OpenCTI)
  3. Compliance reporting (NIST CSF, ISO 27001 requirement mapping)
  4. Adversary technique trend analysis over time
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.utils.config import DEFAULT_CONFIG, ATTACK_TECHNIQUE_MAP
from src.utils.logger import get_logger

log = get_logger(__name__)


def map_attack_techniques(
    features: Dict,
    iocs: Dict,
    gemini_result: Optional[Dict] = None,
    phishing_probability: float = 0.5,
    verdict: str = "UNCERTAIN",
) -> List[Dict]:
    """Map extracted email features to MITRE ATT&CK techniques.

    Args:
        features: Dict of feature names → values from the feature pipeline.
        iocs: IOC dict from ioc_extractor.extract_iocs().
        gemini_result: Optional AI analysis dict for additional signals.
        phishing_probability: ML model probability (0–1).
        verdict: "PHISHING", "LEGITIMATE", or "UNCERTAIN".

    Returns:
        List of ATT&CK technique dicts, each with:
          - technique_id: MITRE ATT&CK technique ID (e.g., "T1566.002")
          - technique_name: Human-readable technique name
          - tactic: ATT&CK tactic (e.g., "Initial Access")
          - confidence: Float 0–1 for technique detection confidence
          - evidence: List of feature names that triggered this mapping
    """
    techniques: List[Dict] = []

    # ---- T1566: Phishing (only when ML verdict is PHISHING or UNCERTAIN) --
    # For LEGITIMATE emails, suppress T1566 entirely — it is misleading to
    # map phishing techniques when the model determined this is not phishing.
    if verdict in ("PHISHING", "UNCERTAIN"):
        t1566_conf = round(min(phishing_probability, 1.0), 2)
        techniques.append({
            "technique_id": "T1566",
            "technique_name": "Phishing",
            "tactic": "Initial Access",
            "confidence": t1566_conf,
            "evidence": [f"PhishLens ML verdict: {phishing_probability:.1%} phishing probability"],
            "mitre_url": "https://attack.mitre.org/techniques/T1566/",
        })

    # ---- T1566.001: Spearphishing Attachment ----------------------------
    attachment_count = features.get("parsed_attachments_count", 0) or len(iocs.get("attachment_hashes", []))
    if attachment_count > 0:
        techniques.append({
            "technique_id": "T1566.001",
            "technique_name": "Spearphishing Attachment",
            "tactic": "Initial Access",
            "confidence": 0.85,
            "evidence": [f"attachment_count={attachment_count}"],
            "mitre_url": "https://attack.mitre.org/techniques/T1566/001/",
        })

    # ---- T1566.002: Spearphishing Link ---------------------------------
    url_count = len(iocs.get("urls", []))
    if url_count > 0:
        techniques.append({
            "technique_id": "T1566.002",
            "technique_name": "Spearphishing Link",
            "tactic": "Initial Access",
            "confidence": min(0.5 + 0.1 * url_count, 0.95),
            "evidence": [f"url_count={url_count}"],
            "mitre_url": "https://attack.mitre.org/techniques/T1566/002/",
        })

    # ---- T1036: Masquerading (brand impersonation) ----------------------
    brand_evidence = []

    # SHAP / Gemini brand signals
    if gemini_result and gemini_result.get("gemini_impersonated_brand"):
        brand_evidence.append(f"gemini_brand={gemini_result['gemini_impersonated_brand']}")

    # URL features: brand in subdomain
    if features.get("url_brand_in_subdomain_max", 0) > 0:
        brand_evidence.append("brand_in_subdomain=True")

    # Cert mismatch
    if features.get("url_cert_brand_mismatch_max", 0) > 0:
        brand_evidence.append("cert_brand_mismatch=True")

    # Domain spoofing
    if features.get("url_punycode_detected_max", 0) > 0:
        brand_evidence.append("punycode_domain=True")

    if brand_evidence:
        techniques.append({
            "technique_id": "T1036",
            "technique_name": "Masquerading",
            "tactic": "Defense Evasion",
            "confidence": 0.80,
            "evidence": brand_evidence,
            "mitre_url": "https://attack.mitre.org/techniques/T1036/",
        })

    # ---- T1204: User Execution (urgency-based social engineering) -------
    urgency = features.get("txt_urgency_score_normalised", 0.0) or 0.0
    if float(urgency) > 0.3:
        techniques.append({
            "technique_id": "T1204",
            "technique_name": "User Execution",
            "tactic": "Execution",
            "confidence": min(float(urgency), 0.9),
            "evidence": [f"urgency_score={urgency:.3f}"],
            "mitre_url": "https://attack.mitre.org/techniques/T1204/",
        })

    # ---- T1056: Input Capture (credential harvesting forms) ------------
    if features.get("html_external_form_action", 0) > 0:
        techniques.append({
            "technique_id": "T1056",
            "technique_name": "Input Capture",
            "tactic": "Collection",
            "confidence": 0.75,
            "evidence": ["external_form_action=True"],
            "mitre_url": "https://attack.mitre.org/techniques/T1056/",
        })

    # ---- T1027: Obfuscated Files / Information -------------------------
    obfuscation_evidence = []
    if features.get("html_base64_content_count", 0) > 0:
        obfuscation_evidence.append("base64_html_content=True")
    if features.get("html_hidden_text_count", 0) > 0:
        obfuscation_evidence.append("hidden_text=True")
    if features.get("html_javascript_count", 0) > 2:
        obfuscation_evidence.append("javascript_obfuscation=True")
    if features.get("url_url_entropy_max", 0) > 4.5:
        obfuscation_evidence.append("high_url_entropy=True")

    if obfuscation_evidence:
        techniques.append({
            "technique_id": "T1027",
            "technique_name": "Obfuscated Files or Information",
            "tactic": "Defense Evasion",
            "confidence": 0.70,
            "evidence": obfuscation_evidence,
            "mitre_url": "https://attack.mitre.org/techniques/T1027/",
        })

    # ---- T1078: Valid Accounts (credential theft phishing) -------------
    keywords_count = features.get("url_suspicious_keywords_in_url_max", 0) or 0
    if float(keywords_count) > 0:
        techniques.append({
            "technique_id": "T1078",
            "technique_name": "Valid Accounts",
            "tactic": "Persistence",
            "confidence": 0.60,
            "evidence": [f"suspicious_url_keywords={keywords_count}"],
            "mitre_url": "https://attack.mitre.org/techniques/T1078/",
        })

    # ---- Authentication bypass / SPF-DKIM-DMARC failures ---------------
    auth_evidence = []
    if float(features.get("hdr_spf_result", 0) or 0) < 0:
        auth_evidence.append("spf_fail=True")
    if float(features.get("hdr_dkim_result", 0) or 0) < 0:
        auth_evidence.append("dkim_fail=True")
    if float(features.get("hdr_dmarc_result", 0) or 0) < 0:
        auth_evidence.append("dmarc_fail=True")

    if auth_evidence:
        techniques.append({
            "technique_id": "T1071.003",
            "technique_name": "Application Layer Protocol: Mail Protocols",
            "tactic": "Command and Control",
            "confidence": 0.65,
            "evidence": auth_evidence,
            "mitre_url": "https://attack.mitre.org/techniques/T1071/003/",
        })

    # ---- T1598: Phishing for Information (form + suspicious URL) --------
    if (features.get("html_external_form_action", 0) or 0) > 0 and (
        float(features.get("url_suspicious_keywords_in_url_max", 0) or 0) > 0
    ):
        if not any(t["technique_id"] == "T1598" for t in techniques):
            techniques.append({
                "technique_id": "T1598",
                "technique_name": "Phishing for Information",
                "tactic": "Reconnaissance",
                "confidence": 0.72,
                "evidence": ["external_form_action=True", "suspicious_url_keywords=True"],
                "mitre_url": "https://attack.mitre.org/techniques/T1598/",
            })

    # ---- T1539: Steal Web Session Cookie (form + urgency) ---------------
    if (features.get("html_external_form_action", 0) or 0) > 0 and (
        float(features.get("txt_urgency_score_normalised", 0) or 0) > 0.5
    ):
        if not any(t["technique_id"] == "T1539" for t in techniques):
            techniques.append({
                "technique_id": "T1539",
                "technique_name": "Steal Web Session Cookie",
                "tactic": "Credential Access",
                "confidence": 0.68,
                "evidence": ["external_form_action=True", "high_urgency=True"],
                "mitre_url": "https://attack.mitre.org/techniques/T1539/",
            })

    # ---- ATTACK_TECHNIQUE_MAP: config-driven feature→technique mapping ---
    for feature_name, tech_info in ATTACK_TECHNIQUE_MAP.items():
        feat_val = features.get(feature_name, 0)
        if feat_val and float(feat_val) > 0:
            tech_id = tech_info.get("technique_id", "") if isinstance(tech_info, dict) else str(tech_info)
            if not any(t["technique_id"] == tech_id for t in techniques):
                techniques.append({
                    "technique_id": tech_id,
                    "technique_name": tech_info.get("technique_name", _technique_name_lookup(tech_id)) if isinstance(tech_info, dict) else _technique_name_lookup(tech_id),
                    "tactic": tech_info.get("tactic", _technique_tactic_lookup(tech_id)) if isinstance(tech_info, dict) else _technique_tactic_lookup(tech_id),
                    "confidence": 0.65,
                    "evidence": [f"{feature_name}={feat_val}"],
                    "mitre_url": f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}/",
                })

    # ── Verdict-based confidence calibration ─────────────────────────────
    # Many features (having a URL, using HTML, base64 encoding) appear in
    # perfectly legitimate business email. Calibrate technique confidence to
    # reflect the actual ML verdict so the ATT&CK map is proportionate.
    _PHISH_THRESHOLD = 0.65

    if verdict == "LEGITIMATE":
        # Drop direct phishing-entry techniques — they are false signals for
        # legitimate email and would mislead SOC analysts.
        _phish_entry_ids = {"T1566", "T1566.001", "T1566.002", "T1566.003"}
        techniques = [t for t in techniques if t["technique_id"] not in _phish_entry_ids]
        # Scale remaining technique confidences down to reflect the low
        # phishing probability.  Max cap: 30%.
        scale = min(0.30, max(0.05, phishing_probability) * 3.0)
        for t in techniques:
            t["confidence"] = round(t["confidence"] * scale, 2)
        # Remove near-zero entries — they add noise, not value.
        techniques = [t for t in techniques if t["confidence"] >= 0.05]

    elif verdict == "UNCERTAIN":
        # Scale proportionately to how far the probability is from the threshold.
        scale = min(1.0, max(0.45, phishing_probability / _PHISH_THRESHOLD))
        for t in techniques:
            t["confidence"] = round(min(t["confidence"] * scale, 0.80), 2)
        techniques = [t for t in techniques if t["confidence"] >= 0.05]

    # For PHISHING verdict: keep all techniques at their full computed confidence.

    log.debug(f"Mapped {len(techniques)} ATT&CK techniques (verdict={verdict}, prob={phishing_probability:.2f})")
    return techniques


def format_attack_mapping_report(techniques: List[Dict]) -> str:
    """Format the ATT&CK mapping as a readable text report.

    Args:
        techniques: Output of map_attack_techniques().

    Returns:
        Multi-line string report suitable for display in Streamlit or terminal.
    """
    if not techniques:
        return "No ATT&CK techniques mapped (email classified as legitimate)."

    lines = ["MITRE ATT&CK Technique Mapping\n" + "=" * 40]
    for t in techniques:
        conf_bar = "█" * int(t["confidence"] * 10)
        lines.append(
            f"\n[{t['technique_id']}] {t['technique_name']}\n"
            f"  Tactic:     {t['tactic']}\n"
            f"  Confidence: {conf_bar} {t['confidence']:.0%}\n"
            f"  Evidence:   {', '.join(t['evidence'])}\n"
            f"  Reference:  {t['mitre_url']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_TECHNIQUE_NAMES = {
    "T1566": "Phishing",
    "T1566.001": "Spearphishing Attachment",
    "T1566.002": "Spearphishing Link",
    "T1566.003": "Spearphishing via Service",
    "T1036": "Masquerading",
    "T1204": "User Execution",
    "T1056": "Input Capture",
    "T1078": "Valid Accounts",
    "T1071.003": "Application Layer Protocol: Mail Protocols",
    "T1027": "Obfuscated Files or Information",
    "T1598": "Phishing for Information",
    "T1539": "Steal Web Session Cookie",
}

_TECHNIQUE_TACTICS = {
    "T1566": "Initial Access",
    "T1566.001": "Initial Access",
    "T1566.002": "Initial Access",
    "T1566.003": "Initial Access",
    "T1036": "Defense Evasion",
    "T1204": "Execution",
    "T1056": "Collection",
    "T1078": "Persistence",
    "T1071.003": "Command and Control",
    "T1027": "Defense Evasion",
    "T1598": "Reconnaissance",
    "T1539": "Credential Access",
}


def _technique_name_lookup(technique_id: str) -> str:
    return _TECHNIQUE_NAMES.get(technique_id, "Unknown Technique")


def _technique_tactic_lookup(technique_id: str) -> str:
    return _TECHNIQUE_TACTICS.get(technique_id, "Unknown Tactic")
