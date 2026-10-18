import re
from preprocess import get_body_text, get_urls, get_headers


IP_URL_RE = re.compile(r'https?://(\d{1,3}\.){3}\d{1,3}')   # http(s):// followed by an IP


URGENCY_WORDS = [ "urgent", "immediately", "verify", "confirm", "act now",
                 "expire", "expires", "within 24 hours", "as soon as possible",
                 "important", "attention", "action required" ]   # e.g. urgent, immediately, verify, confirm, expire, act now
THREAT_WORDS  = [ "suspended", "locked", "terminated", "closed", "disabled",
                 "restricted", "unauthorised", "unauthorized", "breach",
                 "compromised", "fraud", "unusual activity" ]   # e.g. suspended, locked, terminated, restricted, unauthorised

def get_domain(address):
    match = re.search(r'@([\w.-]+)', str(address))
    return match.group(1).lower() if match else ""

def extract_features(message):
    """Turn one email into a dict of behavioural features."""
    body = get_body_text(message)
    urls = get_urls(message)
    headers = get_headers(message)
    text = body.lower()

    features = {}
    
    from_domain = get_domain(headers["from"])
    return_domain = get_domain(headers["return_path"])
    features["sender_domain_mismatch"] = int(
        bool(from_domain) and bool(return_domain) and from_domain != return_domain
    )

    # --- mechanical features ---
    features["url_count"] = len(urls)
    features["email_length"] = len(body.split())          # word count
    letters = [ch for ch in body if ch.isalpha()]
    uppers = [ch for ch in letters if ch.isupper()]
    features["capital_ratio"] = len(uppers) / len(letters) if letters else 0.0
    features["special_char_count"] = sum(body.count(ch) for ch in "!$%")

    
    # 1. suspicious_url: 1 if ANY url is an IP-address link OR contains '@', else 0.
    #    Loop over `urls`, test each with IP_URL_RE and the '@' check.
    features["suspicious_url"] = int(any(
    IP_URL_RE.match(u) or "@" in u for u in urls
))

    # 2. urgency_score / threat_score: count how many words from each list appear in `text`.
    #    Decide: count occurrences or just presence? raw or normalised by length? Record it.
    features["urgency_score"] = sum(1 for w in URGENCY_WORDS if w in text)
    features["threat_score"]  = sum(1 for w in THREAT_WORDS if w in text)

    # 3. sender_domain_mismatch: extract the domain from headers["from"] and from
    #    headers["return_path"], compare them, return 1 if they differ, else 0.
    

    return features


if __name__ == "__main__":
    from preprocess import load_emails
    records = load_emails()
    for target_label, name in [(1, "PHISHING"), (0, "LEGITIMATE")]:
        msg = next(m for m, label in records if label == target_label)
        print(f"\n--- {name} ---")
        print(extract_features(msg))