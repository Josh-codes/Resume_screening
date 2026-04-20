import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pdfplumber

from app.utils_ner import extract_skill_mentions, normalize_skill_name

logger = logging.getLogger(__name__)

REQUIREMENT_WEIGHTS: Dict[str, int] = {
    "must-have": 5,
    "required": 3,
    "preferred": 2,
    "nice-to-have": 1,
}

REQUIREMENT_HINTS = {
    "must-have": ["must have", "must-have", "required", "requirements"],
    "preferred": ["preferred", "nice to have", "nice-to-have", "bonus", "optional"],
}


def extract_text_from_pdf_or_txt(file_path: str) -> str:
    """Extract text from PDF or TXT with basic error handling."""
    try:
        if file_path.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as exc:  # pragma: no cover - best-effort extraction
        logger.info("Failed to extract text from %s: %s", file_path, exc)
    return ""


def _infer_requirement_from_context(context: str) -> str:
    lowered = context.lower()
    for label, hints in REQUIREMENT_HINTS.items():
        if any(hint in lowered for hint in hints):
            return label
    return "required"


def build_jd_skill_profile(text: str) -> Dict[str, object]:
    """
    Extract skills from JD and assign importance weights.

    Returns dict with jd_skills list, total_weight, and missing buckets.
    """
    if not text or not text.strip():
        return {"jd_skills": [], "total_weight": 0}

    mentions = extract_skill_mentions(text)
    jd_skills: List[Dict[str, object]] = []
    total_weight = 0
    for mention in mentions:
        contexts = mention.get("context", [])
        requirement = "required"
        for context in contexts:
            requirement = _infer_requirement_from_context(context)
            if requirement != "required":
                break
        weight = REQUIREMENT_WEIGHTS.get(requirement, 3)
        jd_skills.append({
            "skill": mention["skill"],
            "requirement": requirement,
            "weight": weight,
            "mentions": mention.get("mentions", 0),
        })
        total_weight += weight

    return {"jd_skills": jd_skills, "total_weight": total_weight}


def extract_resume_skills(text: str, jd_skills: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Extract skills from resume text using NER with fallback."""
    return extract_skill_mentions(text, jd_skills)


def _parse_years(text: str) -> List[int]:
    matches = re.findall(r"\b(\d{1,2})\+?\s*(?:years|yrs)\b", text.lower())
    return [int(m) for m in matches]


def extract_experience_info(text: str) -> Dict[str, object]:
    """
    Extract total years and basic per-skill experience mentions.
    """
    years = _parse_years(text)
    total_years = max(years) if years else 0

    by_skill: Dict[str, int] = {}
    for match in re.finditer(
        r"\b(\d{1,2})\+?\s*(?:years|yrs)\s+(?:of\s+)?([A-Za-z][A-Za-z0-9\+\#\. ]{1,30})",
        text,
        flags=re.IGNORECASE,
    ):
        count = int(match.group(1))
        skill = normalize_skill_name(match.group(2))
        if skill:
            by_skill[skill] = max(by_skill.get(skill, 0), count)

    level = classify_experience_level(total_years)
    return {"total_years": total_years, "level": level, "by_skill": by_skill}


def classify_experience_level(years: int) -> str:
    if years >= 5:
        return "Senior"
    if years >= 2:
        return "Mid"
    if years > 0:
        return "Entry"
    return "Unknown"


def experience_match_score(jd_exp: Dict[str, object], resume_exp: Dict[str, object]) -> int:
    """Compute 0-100 alignment score between JD and resume levels."""
    jd_level = jd_exp.get("level") if jd_exp else "Unknown"
    resume_level = resume_exp.get("level") if resume_exp else "Unknown"
    if jd_level == "Unknown" or resume_level == "Unknown":
        return 50

    levels = {"Entry": 1, "Mid": 2, "Senior": 3}
    jd_value = levels.get(jd_level, 0)
    resume_value = levels.get(resume_level, 0)
    diff = resume_value - jd_value

    if diff >= 0:
        return 95 if diff == 0 else 85
    if diff == -1:
        return 60
    return 35
