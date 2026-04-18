import os
import re
import pdfplumber

def save_uploaded_file(uploaded_file, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def extract_text_from_pdf_or_txt(file_path):
    if file_path.lower().endswith('.pdf'):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ""

def load_job_description(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_skills(text, skill_keywords):
    text_lower = text.lower()
    found = []
    for skill in skill_keywords:
        if not skill:
            continue
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def extract_skills_from_jd(text):
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "with", "you", "your", "we", "our", "will", "this",
        "these", "those", "such", "plus", "preferred", "required", "requirements",
        "skills", "skill", "qualifications", "qualification", "experience",
        "proficient", "proficiency", "knowledge", "familiar", "ability", "abilities"
    }

    def normalize_skill(phrase):
        words = phrase.split()
        normalized = []
        for word in words:
            if word.isupper() or any(ch.isdigit() for ch in word) or any(ch in word for ch in ["+", "#"]):
                normalized.append(word)
            else:
                normalized.append(word.title())
        return " ".join(normalized).strip()

    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    focus_lines = []
    trigger_terms = [
        "skills", "requirements", "qualifications", "must have", "experience with",
        "proficient", "knowledge of", "familiar with"
    ]

    for line in lines:
        lowered = line.lower()
        if any(term in lowered for term in trigger_terms):
            focus_lines.append(line)
        if re.match(r"^[-*\u2022]\s+", line):
            focus_lines.append(line)

    if not focus_lines:
        focus_lines = lines

    candidates = []
    for line in focus_lines:
        line = re.sub(r"^[-*\u2022]\s+", "", line)
        parts = re.split(r"[;,/|]", line)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part = re.sub(
                r"(?i)\b(requirements?|skills?|qualifications?|experience with|proficient in|knowledge of|familiar with|must have)\b[:\-]?\s*",
                "",
                part
            )
            tokens = re.findall(r"[A-Za-z0-9\+\#\.\-]+", part)
            cleaned_tokens = []
            for token in tokens:
                token_lower = token.lower()
                if token_lower in stopwords:
                    continue
                if len(token) < 2:
                    continue
                cleaned_tokens.append(token)

            if not cleaned_tokens:
                continue

            if len(cleaned_tokens) > 4:
                cleaned_tokens = cleaned_tokens[:4]

            phrase = " ".join(cleaned_tokens)
            candidates.append(phrase)

    skills = []
    seen = set()
    for phrase in candidates:
        normalized = normalize_skill(phrase)
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            skills.append(normalized)

    return skills

def load_all_resumes(folder):
    # Load resumes as dict with text & filename
    resumes = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        text = extract_text_from_pdf_or_txt(path)
        if text:
            resumes.append({"filename": file, "text": text})
    return resumes
