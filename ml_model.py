import math
import os
import pickle
import re
import tempfile
import time
from urllib.parse import urlparse

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from PIL import Image, ImageStat
except Exception:
    Image = None
    ImageStat = None

try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.options import Options
except Exception:
    webdriver = None
    TimeoutException = Exception
    WebDriverException = Exception
    Options = None

try:
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
except Exception:
    AutoImageProcessor = None
    AutoModel = None
    AutoModelForImageClassification = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


URL_PATTERN = re.compile(
    r"((?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?)"
)

PHISHING_KEYWORDS = {
    "urgent",
    "immediately",
    "verify",
    "suspend",
    "restricted",
    "password",
    "click here",
    "action required",
    "confirm your account",
    "login now",
    "security alert",
    "invoice attached",
    "verify your identity",
    "account locked",
    "payment failed",
    "update payment",
    "wire transfer",
    "gift card",
    "crypto",
    "unauthorized",
    "reset password",
    "confirm now",
}

SUSPICIOUS_URL_KEYWORDS = {
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "billing",
    "wallet",
    "bank",
    "signin",
    "unlock",
    "auth",
    "otp",
    "password",
    "recover",
    "webscr",
}

PHISHING_PROTOTYPES = [
    "Urgent action required to verify your account now.",
    "Your account will be suspended unless you click this link immediately.",
    "Security alert, confirm password and billing details now.",
    "Reset your bank login with the attached secure verification link.",
]

SAFE_PROTOTYPES = [
    "Team meeting agenda for tomorrow attached.",
    "Your order has shipped and is expected this week.",
    "Monthly report summary and project updates.",
    "Friendly reminder about your scheduled appointment.",
]

_bert_bundle = None
_mobilenet_bundle = None
_url_model_bundle = None


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _safe_logit(p):
    p = _clamp01(p)
    p = min(1.0 - 1e-6, max(1e-6, p))
    return math.log(p / (1.0 - p))


def _stacked_fused_risk(nlp_risk, vision_risk, has_successful_vision):
    """Logistic-regression style stacking of NLP and Vision risks.

    Coefficients are calibrated heuristics (not fit from labels yet), designed to
    amplify confidence when NLP and vision are both high.
    """
    nlp_risk = _clamp01(nlp_risk)
    vision_risk = _clamp01(vision_risk)

    x_nlp = _safe_logit(nlp_risk)

    if has_successful_vision:
        x_vision = _safe_logit(vision_risk)
        interaction = nlp_risk * vision_risk
        z = (
            -0.40
            + 1.20 * x_nlp
            + 1.00 * x_vision
            + 1.20 * interaction
        )
    else:
        # If no successful screenshots, fall back to NLP-only stacked estimate.
        z = -0.45 + 1.30 * x_nlp

    return _clamp01(_sigmoid(z))


def _shannon_entropy(text):
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    total = float(len(text))
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _extract_precomputed_like_features(url, feature_columns):
    parsed = urlparse(url)
    raw_host = (parsed.netloc or "").split("@")[-1]
    host = raw_host.split(":")[0].lower()
    path = parsed.path or ""
    query = parsed.query or ""
    url_lower = url.lower()

    try:
        port = parsed.port
    except ValueError:
        port = None

    host_labels = [token for token in host.split(".") if token]
    tld_len = len(host_labels[-1]) if host_labels else 0
    query_vars = query.count("&") + 1 if query else 0
    is_ip = 1 if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host) else 0

    url_digit_count = sum(ch.isdigit() for ch in url)
    host_digit_count = sum(ch.isdigit() for ch in host)
    query_digit_count = sum(ch.isdigit() for ch in query)

    url_letter_count = sum(ch.isalpha() for ch in url)
    host_letter_count = sum(ch.isalpha() for ch in host)
    query_letter_count = sum(ch.isalpha() for ch in query)

    symbol_count_url = sum(not ch.isalnum() for ch in url)
    symbol_count_domain = sum(not ch.isalnum() for ch in host)

    sensitive_hits = sum(1 for token in SUSPICIOUS_URL_KEYWORDS if token in url_lower)
    executable = 1 if any(path.lower().endswith(ext) for ext in [".exe", ".bat", ".cmd", ".scr", ".js", ".vbs", ".jar", ".ps1", ".msi"]) else 0

    values = {
        "querylength": len(query),
        "domainlength": len(host),
        "pathlength": len(path),
        "urllen": len(url),
        "tld": tld_len,
        "isporteighty": 1 if port == 80 else 0,
        "numberofdotsinurl": url.count("."),
        "isipaddressindomainname": is_ip,
        "url_digitcount": url_digit_count,
        "host_digitcount": host_digit_count,
        "query_digitcount": query_digit_count,
        "url_letter_count": url_letter_count,
        "host_letter_count": host_letter_count,
        "query_lettercount": query_letter_count,
        "numberrate_url": url_digit_count / max(len(url), 1),
        "numberrate_domain": host_digit_count / max(len(host), 1),
        "symbolcount_url": symbol_count_url,
        "symbolcount_domain": symbol_count_domain,
        "entropy_url": _shannon_entropy(url_lower),
        "entropy_domain": _shannon_entropy(host),
        "url_sensitiveword": sensitive_hits,
        "urlqueries_variable": query_vars,
        "executable": executable,
    }

    return [float(values.get(col.lower(), 0.0)) for col in feature_columns]


def _get_url_model_bundle():
    global _url_model_bundle
    if _url_model_bundle is not None:
        return _url_model_bundle

    model_path = os.path.join("trained_models", "url_classifier.pkl")
    if not os.path.exists(model_path):
        _url_model_bundle = None
        return None

    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        # Backward compatibility with old tuple format.
        if isinstance(payload, tuple) and len(payload) == 3:
            clf, scaler, extractor = payload
            payload = {
                "clf": clf,
                "scaler": scaler,
                "extractor": extractor,
                "mode": "legacy_extractor",
                "feature_columns": None,
            }

        if not isinstance(payload, dict):
            _url_model_bundle = None
            return None

        if "clf" not in payload or "scaler" not in payload:
            _url_model_bundle = None
            return None

        _url_model_bundle = payload
        return _url_model_bundle
    except Exception:
        _url_model_bundle = None
        return None


def _model_url_probability(url):
    bundle = _get_url_model_bundle()
    if bundle is None:
        return None

    try:
        mode = bundle.get("mode", "legacy_extractor")
        if mode == "legacy_extractor" and bundle.get("extractor") is not None:
            feature_row = bundle["extractor"].extract_features(url)
        elif mode == "precomputed" and bundle.get("feature_columns"):
            feature_row = _extract_precomputed_like_features(url, bundle["feature_columns"])
        else:
            return None

        scaled = bundle["scaler"].transform([feature_row])
        clf = bundle["clf"]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(scaled)[0]
            if len(probs) >= 2:
                return float(probs[1])
            return float(probs[0])
        return None
    except Exception:
        return None


def _get_bert_bundle():
    global _bert_bundle
    if AutoTokenizer is None or torch is None or F is None:
        return None
    if _bert_bundle is None:
        classifier_model = None
        encoder_model = None
        tokenizer = None

        trained_cls_path = os.path.join("trained_models", "bert_email_classifier")

        if AutoModelForSequenceClassification is not None and os.path.isdir(trained_cls_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(trained_cls_path)
                classifier_model = AutoModelForSequenceClassification.from_pretrained(trained_cls_path)
                classifier_model.eval()
            except Exception:
                classifier_model = None
                tokenizer = None

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        if AutoModel is not None and classifier_model is None:
            encoder_model = AutoModel.from_pretrained("bert-base-uncased")
            encoder_model.eval()
            phishing_center = _embed_many(PHISHING_PROTOTYPES, tokenizer, encoder_model).mean(dim=0)
            safe_center = _embed_many(SAFE_PROTOTYPES, tokenizer, encoder_model).mean(dim=0)
            phishing_center = F.normalize(phishing_center, dim=0)
            safe_center = F.normalize(safe_center, dim=0)
        else:
            phishing_center = None
            safe_center = None

        _bert_bundle = {
            "tokenizer": tokenizer,
            "encoder_model": encoder_model,
            "classifier_model": classifier_model,
            "phishing_center": phishing_center,
            "safe_center": safe_center,
        }
    return _bert_bundle


def _get_mobilenet_bundle():
    global _mobilenet_bundle
    if AutoImageProcessor is None or AutoModelForImageClassification is None or torch is None or F is None:
        return None
    if _mobilenet_bundle is None:
        trained_model_path = os.path.join("trained_models", "mobilenet_vision_classifier")
        if os.path.isdir(trained_model_path):
            processor = AutoImageProcessor.from_pretrained(trained_model_path)
            model = AutoModelForImageClassification.from_pretrained(trained_model_path)
        else:
            model_name = "google/mobilenet_v2_1.0_224"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        _mobilenet_bundle = {"processor": processor, "model": model}
    return _mobilenet_bundle


def _embed_many(texts, tokenizer, model):
    if torch is None or F is None:
        return None
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        outputs = model(**encoded)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_vectors, dim=1)


def _bert_text_risk(email_text):
    bundle = _get_bert_bundle()
    if bundle is None:
        heuristic = min(1.0, sum(1 for k in PHISHING_KEYWORDS if k in email_text.lower()) * 0.14)
        return heuristic, 0.0, 0.0

    classifier_model = bundle.get("classifier_model")
    if classifier_model is not None:
        with torch.no_grad():
            encoded = bundle["tokenizer"](
                [email_text],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            logits = classifier_model(**encoded).logits
            probs = F.softmax(logits, dim=-1)[0]

        # Trained classifier label convention: 0=legitimate, 1=phishing.
        phishing_prob = float(probs[1].item()) if probs.numel() > 1 else float(probs[0].item())
        safe_prob = float(probs[0].item()) if probs.numel() > 1 else float(1.0 - phishing_prob)
        return max(0.0, min(1.0, phishing_prob)), phishing_prob, safe_prob

    encoder_model = bundle.get("encoder_model")
    if encoder_model is None or bundle.get("phishing_center") is None or bundle.get("safe_center") is None:
        heuristic = min(1.0, sum(1 for k in PHISHING_KEYWORDS if k in email_text.lower()) * 0.14)
        return heuristic, 0.0, 0.0

    embeddings = _embed_many([email_text], bundle["tokenizer"], encoder_model)
    if embeddings is None:
        heuristic = min(1.0, sum(1 for k in PHISHING_KEYWORDS if k in email_text.lower()) * 0.14)
        return heuristic, 0.0, 0.0
    vector = embeddings[0]

    phishing_sim = torch.dot(vector, bundle["phishing_center"]).item()
    safe_sim = torch.dot(vector, bundle["safe_center"]).item()
    margin = phishing_sim - safe_sim

    probability = 1.0 / (1.0 + math.exp(-5.0 * margin))
    return max(0.0, min(1.0, probability)), phishing_sim, safe_sim


def _extract_urls(text):
    matches = URL_PATTERN.findall(text)
    seen = set()
    urls = []
    for url in matches:
        clean = url.strip("\n\r\t .,;\")'")
        normalized = clean if clean.startswith(("http://", "https://")) else f"https://{clean}"
        if normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def _url_risk(url):
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    full = f"{host}{path}"

    risk = 0.0
    reasons = []

    if re.search(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        risk += 0.45
        reasons.append("Direct IP address in URL")
    if "@" in url:
        risk += 0.25
        reasons.append("URL contains @ redirection pattern")
    if "xn--" in host:
        risk += 0.30
        reasons.append("Punycode domain detected")
    if any(token in full for token in SUSPICIOUS_URL_KEYWORDS):
        risk += 0.25
        reasons.append("Credential-themed URL keywords")
    if len(host) > 45:
        risk += 0.15
        reasons.append("Unusually long domain")

    model_prob = _model_url_probability(url)
    if model_prob is not None:
        # Blend learned classifier output with structural heuristic to improve robustness.
        risk = min(1.0, 0.65 * model_prob + 0.35 * risk)
        reasons.append(f"Model URL risk: {int(round(model_prob * 100))}%")

    return min(1.0, risk), reasons


def _start_driver():
    if webdriver is None or Options is None:
        raise WebDriverException("Selenium is not installed")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(options=options)


def _capture_screenshots(urls, output_dir, max_urls=None):
    captures = []
    target_urls = urls if max_urls is None else urls[:max_urls]
    if not target_urls:
        return captures

    def _wait_for_render(driver, timeout_seconds=12):
        """Wait until DOM is complete and page body has visible content."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                ready_state = driver.execute_script("return document.readyState")
                body_exists = driver.execute_script("return !!document.body")
                body_text_len = driver.execute_script(
                    "return document.body && document.body.innerText ? document.body.innerText.trim().length : 0"
                )
                if ready_state == "complete" and body_exists and body_text_len > 20:
                    return True
            except Exception:
                pass
            time.sleep(0.35)
        return False

    def _is_likely_blank_screenshot(image_path):
        if not image_path or not os.path.exists(image_path):
            return True
        if Image is None or ImageStat is None:
            return False
        try:
            gray = Image.open(image_path).convert("L")
            stats = ImageStat.Stat(gray)
            mean = stats.mean[0]
            variance = stats.var[0]
            # Near-uniform pages (all white/all dark) are likely failed or pre-render captures.
            return variance < 12 and (mean > 235 or mean < 20)
        except Exception:
            return False

    def _scroll_page_for_lazy_content(driver):
        """Scroll through the page to trigger lazy-loaded UI/assets."""
        try:
            page_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
            if not page_height or page_height <= 0:
                return
            steps = [0.25, 0.55, 0.85, 0.45, 0.0]
            for ratio in steps:
                y = int(page_height * ratio)
                driver.execute_script("window.scrollTo(0, arguments[0]);", y)
                time.sleep(0.4)
        except Exception:
            pass

    try:
        driver = _start_driver()
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(8)
    except WebDriverException as exc:
        for url in target_urls:
            captures.append(
                {
                    "url": url,
                    "screenshot_path": None,
                    "page_title": "",
                    "error": f"Selenium setup failed: {exc.__class__.__name__}",
                }
            )
        return captures

    for idx, url in enumerate(target_urls):
        shot_path = os.path.join(output_dir, f"shot_{idx + 1}.png")
        try:
            driver.get(url)
            _wait_for_render(driver, timeout_seconds=12)
            # Give async resources (fonts/images/client scripts) a moment to paint.
            time.sleep(1.0)
            driver.save_screenshot(shot_path)
            retry_note = ""

            if _is_likely_blank_screenshot(shot_path):
                try:
                    driver.refresh()
                    _wait_for_render(driver, timeout_seconds=12)
                    time.sleep(1.3)
                    driver.save_screenshot(shot_path)
                    if _is_likely_blank_screenshot(shot_path):
                        _scroll_page_for_lazy_content(driver)
                        _wait_for_render(driver, timeout_seconds=8)
                        time.sleep(1.0)
                        driver.save_screenshot(shot_path)
                        if _is_likely_blank_screenshot(shot_path):
                            retry_note = "Capture still appears blank after retries"
                except Exception:
                    retry_note = "Retry after blank capture failed"

            page_text_hint = ""
            try:
                page_text_hint = (
                    driver.execute_script(
                        "return (document.body && document.body.innerText) ? document.body.innerText.slice(0, 1200) : '';"
                    )
                    or ""
                )
            except Exception:
                page_text_hint = ""

            captures.append(
                {
                    "url": url,
                    "screenshot_path": shot_path,
                    "page_title": (driver.title or "").strip(),
                    "page_text_hint": str(page_text_hint).strip(),
                    "error": retry_note,
                }
            )
        except TimeoutException:
            captures.append(
                {
                    "url": url,
                    "screenshot_path": None,
                    "page_title": "",
                    "error": "Page load timeout",
                }
            )
        except WebDriverException as exc:
            captures.append(
                {
                    "url": url,
                    "screenshot_path": None,
                    "page_title": "",
                    "error": f"Screenshot failed: {exc.__class__.__name__}",
                }
            )

    driver.quit()
    return captures


def _vision_risk_for_capture(capture):
    if not capture["screenshot_path"] or not os.path.exists(capture["screenshot_path"]):
        return {
            "vision_risk": 0.35,
            "vision_label": "No screenshot",
            "vision_confidence": 0.0,
            "vision_notes": capture["error"] or "Screenshot unavailable",
        }

    bundle = _get_mobilenet_bundle()
    if Image is None:
        return {
            "vision_risk": 0.35,
            "vision_label": "Pillow missing",
            "vision_confidence": 0.0,
            "vision_notes": "Pillow is not installed",
        }

    image = Image.open(capture["screenshot_path"]).convert("RGB")

    title_text = str(capture.get("page_title", "") or "").lower()
    error_text = str(capture.get("error", "") or "").lower()
    page_text_hint = str(capture.get("page_text_hint", "") or "").lower()
    combined_text = f"{title_text} {error_text} {page_text_hint}"

    warning_markers = [
        "your connection is not private",
        "privacy error",
        "net::err_cert",
        "cert_common_name_invalid",
        "err_ssl",
        "certificate",
        "back to safety",
    ]
    if any(marker in combined_text for marker in warning_markers):
        return {
            "vision_risk": 1.0,
            "vision_label": "SSL/Privacy warning interstitial",
            "vision_confidence": 100.0,
            "vision_notes": "Browser certificate/privacy warning detected",
        }

    if bundle is not None and torch is not None and F is not None:
        with torch.no_grad():
            inputs = bundle["processor"](images=image, return_tensors="pt")
            logits = bundle["model"](**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        label = bundle["model"].config.id2label[top_idx]
    else:
        confidence = 0.0
        label = "MobileNetV2 unavailable"

    # A very low-variance screenshot often indicates blank/error pages used in evasive phishing flows.
    gray_stats = ImageStat.Stat(image.convert("L")) if ImageStat is not None else None
    variance = gray_stats.var[0] if gray_stats is not None else 0.0
    flat_page_penalty = 0.40 if variance < 20 else 0.0

    model_uncertainty = 1.0 - confidence
    confidence_penalty = min(0.55, model_uncertainty * 0.7)
    label_penalty = 0.20 if any(token in label.lower() for token in ["web", "monitor", "screen", "menu"]) else 0.0

    risk = min(1.0, 0.15 + confidence_penalty + flat_page_penalty + label_penalty)
    note = f"Top class: {label} ({confidence:.2f})"
    return {
        "vision_risk": risk,
        "vision_label": label,
        "vision_confidence": round(confidence * 100, 2),
        "vision_notes": note,
    }


def _calc_url_structure_risk_score(urls, url_risk_scores):
    if not url_risk_scores:
        return 0
    avg_risk = sum(url_risk_scores) / len(url_risk_scores)
    return int(round(avg_risk * 100))


def _calc_vision_similarity(captures):
    if not captures or Image is None:
        return 50
    if len(captures) < 2:
        screenshot_path = captures[0].get("screenshot_path") if captures else None
        if screenshot_path and os.path.exists(screenshot_path):
            return 95
        return 0
    img1_path = captures[0].get("screenshot_path")
    img2_path = captures[1].get("screenshot_path")
    if not img1_path or not img2_path or not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return 0
    try:
        img1 = Image.open(img1_path).convert("L").resize((64, 64))
        img2 = Image.open(img2_path).convert("L").resize((64, 64))
        data1 = list(img1.getdata())
        data2 = list(img2.getdata())
        diff = sum(abs(a - b) for a, b in zip(data1, data2))
        max_diff = 255 * len(data1)
        similarity = max(0, 100 - int(round(100.0 * diff / max_diff)))
        return similarity
    except Exception:
        return 50


def _risk_band(score):
    if score > 50:
        return "DANGER", "#ff4b4b"
    if score >= 25:
        return "SUSPICIOUS", "#ffd43b"
    return "SAFE", "#00ff00"


def _classify_email(risk_score, nlp_risk):
    if risk_score > 50:
        return "DANGER"
    if risk_score >= 25:
        return "SUSPICIOUS"
    return "SAFE"


def _assessment_from_percent(score_percent):
    label, _ = _risk_band(float(score_percent))
    return label


def _generate_nlp_findings(nlp_risk, keyword_hits, url_risk_score):
    findings = []
    if nlp_risk >= 0.65:
        findings.append("Suspicious E-mail language patterns")
        findings.append("High Sentiment: FEAR")
    elif nlp_risk >= 0.35:
        findings.append("Suspicious E-mail language patterns")
        findings.append("Sentiment: CAUTION")
    else:
        findings.append("Normal E-mail language patterns")
    if url_risk_score > 0:
        findings.append(f"Suspicious URL Structure Score: {url_risk_score}%")
    return findings


def analyze_email_with_ai(email_text):
    time.sleep(0.5)
    if not email_text or not email_text.strip():
        return default_safe()

    text_lower = email_text.lower()
    bert_prob, phishing_sim, safe_sim = _bert_text_risk(email_text)
    keyword_hits = [token for token in PHISHING_KEYWORDS if token in text_lower]
    keyword_risk = min(1.0, 0.16 * len(keyword_hits))
    nlp_risk = min(1.0, 0.70 * bert_prob + 0.30 * keyword_risk)

    urls = _extract_urls(email_text)
    url_objects = []
    url_risk_scores = []

    output_dir = os.path.join(tempfile.gettempdir(), "phishing_detector_v3")
    os.makedirs(output_dir, exist_ok=True)

    capture_map = {}
    captures = _capture_screenshots(urls, output_dir=output_dir, max_urls=None)
    for cap in captures:
        capture_map[cap["url"]] = cap

    vision_risks = []
    for url in urls:
        base_risk, reasons = _url_risk(url)
        capture = capture_map.get(
            url,
            {"url": url, "screenshot_path": None, "page_title": "", "error": "Not captured"},
        )
        vision_result = _vision_risk_for_capture(capture)

        combined_url_risk = min(1.0, 0.3 * base_risk + 0.7 * vision_result["vision_risk"])
        url_risk_scores.append(combined_url_risk)
        vision_risks.append(vision_result["vision_risk"])

        status = "DANGER" if combined_url_risk > 0.50 else ("SUSPICIOUS" if combined_url_risk >= 0.25 else "SAFE")
        short_url = url if len(url) <= 72 else f"{url[:69]}..."
        full_url = url

        url_objects.append(
            {
                "url": short_url,
                "full_url": full_url,
                "status": status,
                "url_risk_score": round(combined_url_risk * 100, 2),
                "screenshot_path": capture["screenshot_path"],
                "page_title": capture["page_title"],
                "vision_risk_score": round(vision_result["vision_risk"] * 100, 2),
                "vision_label": vision_result["vision_label"],
                "vision_confidence": vision_result["vision_confidence"],
                "vision_notes": vision_result["vision_notes"],
                "url_reasons": reasons,
            }
        )

    avg_url_risk = sum(url_risk_scores) / len(url_risk_scores) if url_risk_scores else 0.05
    avg_vision_risk = sum(vision_risks) / len(vision_risks) if vision_risks else 0.5
    url_structure_risk_score = _calc_url_structure_risk_score(urls, url_risk_scores)

    # Check if vision data is actually available (successful captures).
    has_successful_vision = any(cap.get("screenshot_path") and os.path.exists(cap.get("screenshot_path")) for cap in captures)
    
    # Logistic-regression style stacking across model outputs.
    unified_risk = _stacked_fused_risk(
        nlp_risk=nlp_risk,
        vision_risk=avg_vision_risk,
        has_successful_vision=has_successful_vision,
    )
    
    risk_score = int(round(unified_risk * 100))
    risk_score = min(100, max(0, risk_score))

    bert_nlp_score = round(bert_prob * 100, 2)
    url_analyzer_score = round(avg_url_risk * 100, 2)
    mobilenet_vision_score = round(avg_vision_risk * 100, 2)

    classification = _classify_email(risk_score, nlp_risk)
    nlp_findings = _generate_nlp_findings(nlp_risk, keyword_hits, url_structure_risk_score)
    vision_similarity = _calc_vision_similarity(list(capture_map.values()))

    risk_level, color = _risk_band(risk_score)
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "color": color,
        "classification": classification,
        "stacked_scores": {
            "bert_nlp_risk_score": bert_nlp_score,
            "mobilenet_vision_risk_score": mobilenet_vision_score,
            "url_analyzer_risk_score": url_analyzer_score,
            "fused_risk_score": risk_score,
        },
        "nlp_summary": {
            "bert_nlp_risk_score": bert_nlp_score,
            "assessment": _assessment_from_percent(bert_nlp_score),
        },
        "url_summary": {
            "url_analyzer_risk_score": url_analyzer_score,
            "assessment": _assessment_from_percent(url_analyzer_score),
            "checked_urls": len(urls),
        },
        "nlp_findings": nlp_findings,
        "extracted_urls": url_objects,
        "has_vision_data": has_successful_vision,
        "vision_summary": {
            "avg_vision_risk": mobilenet_vision_score,
            "assessment": _assessment_from_percent(mobilenet_vision_score),
            "checked_urls": len(captures),
            "similarity": vision_similarity,
        },
    }


def default_safe():
    return {
        "risk_score": 0,
        "risk_level": "SAFE",
        "color": "#00ff00",
        "classification": "SAFE",
        "stacked_scores": {
            "bert_nlp_risk_score": 0.0,
            "mobilenet_vision_risk_score": 0.0,
            "url_analyzer_risk_score": 0.0,
            "fused_risk_score": 0,
        },
        "nlp_summary": {
            "bert_nlp_risk_score": 0.0,
            "assessment": "SAFE",
        },
        "url_summary": {
            "url_analyzer_risk_score": 0.0,
            "assessment": "SAFE",
            "checked_urls": 0,
        },
        "nlp_findings": ["Normal E-mail language patterns"],
        "extracted_urls": [],
        "vision_summary": {"avg_vision_risk": 0.0, "assessment": "SAFE", "checked_urls": 0, "similarity": 100},
    }