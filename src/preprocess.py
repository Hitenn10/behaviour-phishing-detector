import mailbox
import re
from email import message_from_bytes
from pathlib import Path
from bs4 import BeautifulSoup

URL_RE = re.compile(r'https?://[^\s<>"\')]+', re.IGNORECASE)

def load_emails():
    records = []  # each item: (email.message.Message, label)
    skipped = 0

    # --- Phishing: every Nazario mbox in the folder (label 1) ---
    nazario_dir = Path("data/raw/nazario")
    for mbox_path in sorted(nazario_dir.glob("*.mbox")):
        box = mailbox.mbox(str(mbox_path))
        for message in box:
            records.append((message, 1))
        print(f"  loaded {mbox_path.name}")   # shows which files were read

    # --- Legitimate: SpamAssassin ham folders (label 0) ---
    ham_dirs = ["data/raw/spamassassin/easy_ham",
                "data/raw/spamassassin/hard_ham"]
    for ham_dir in ham_dirs:
        for path in Path(ham_dir).iterdir():
            if not path.is_file():
                continue
            if path.name == "cmds":      # skip the non-email index file
                continue
            try:
                message = message_from_bytes(path.read_bytes())
                records.append((message, 0))
            except Exception:
                skipped += 1

    phishing = sum(1 for _, label in records if label == 1)
    legit = sum(1 for _, label in records if label == 0)
    print(f"Phishing (Nazario):            {phishing}")
    print(f"Legitimate (SpamAssassin ham): {legit}")
    print(f"Total:                         {len(records)}")
    print(f"Skipped (unparseable):         {skipped}")

    return records
def _decode(payload, part):
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, AttributeError):
        return payload.decode("utf-8", errors="replace")


def get_body_text(message):
    """Return the email body as plain text (HTML stripped, best-effort decode)."""
    plain_parts = []
    html_parts = []

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type not in ("text/plain", "text/html"):
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            decoded = _decode(payload, part)
            if content_type == "text/plain":
                plain_parts.append(decoded)
            else:
                html_parts.append(decoded)
    else:
        payload = message.get_payload(decode=True)
        if payload:
            decoded = _decode(payload, message)
            if message.get_content_type() == "text/html":
                html_parts.append(decoded)
            else:
                plain_parts.append(decoded)

    # Prefer plain text when the sender provided both alternatives
    chosen = plain_parts if plain_parts else html_parts
    text = "\n".join(chosen)
    return BeautifulSoup(text, "lxml").get_text(separator=" ").strip()

def get_urls(message):
    """Extract URLs from the email: hrefs from HTML parts, bare URLs from plain text."""
    urls = []

    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        decoded = _decode(payload, part)

        if content_type == "text/html":
            soup = BeautifulSoup(decoded, "lxml")
            for tag in soup.find_all("a", href=True):
                urls.append(tag["href"])
            # bare URLs can also sit in HTML text, outside anchors
            urls.extend(URL_RE.findall(soup.get_text(separator=" ")))
        else:
            urls.extend(URL_RE.findall(decoded))

    # keep only http/https links; drop mailto:, javascript:, anchors
    urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]
    return urls

def get_headers(message):
    return {
        "from": str(message.get("From", "")),
        "return_path": str(message.get("Return-Path", "")),
        "reply_to": str(message.get("Reply-To", "")),
        "subject": str(message.get("Subject", "")),
    }

if __name__ == "__main__":
    records = load_emails()

    for target_label, name in [(1, "PHISHING"), (0, "LEGITIMATE")]:
        msg = next(m for m, label in records if label == target_label)
        print(f"\n--- {name} ---")
        print("Headers:", get_headers(msg))
        print("URLs:", get_urls(msg)[:5])
        print("Body (first 200 chars):", get_body_text(msg)[:200])