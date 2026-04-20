import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:  # pragma: no cover - optional dependency
    spacy = None
    PhraseMatcher = None

SKILL_SYNONYMS: Dict[str, str] = {
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure": "Azure",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "py": "Python",
    "python": "Python",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "ai": "Artificial Intelligence",
    "nlp": "NLP",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "sql": "SQL",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "c++": "C++",
    "c#": "C#",
}

SKILL_LEXICON: List[str] = sorted(set(SKILL_SYNONYMS.values()))


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy model once, return None if unavailable."""
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("spaCy model en_core_web_sm not available; falling back to regex.")
        return None


def _normalize_key(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = cleaned.replace("/", " ").replace("_", " ")
    cleaned = cleaned.replace("-", " ")
    return cleaned


def normalize_skill_name(text: str) -> str:
    """Normalize skill text to canonical representation when possible."""
    if not text:
        return ""
    key = _normalize_key(text)
    if key in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[key]
    # Preserve mixed tokens like C++, C#, Node.js
    if re.search(r"[A-Za-z]\d|[\+\#\.]", text):
        return text.strip()
    return text.strip().title()


def _build_phrase_matcher(nlp, extra_terms: Optional[List[str]] = None):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms = set(SKILL_LEXICON)
    if extra_terms:
        terms.update([t for t in extra_terms if t])
    patterns = [nlp.make_doc(t) for t in terms]
    matcher.add("SKILL", patterns)
    return matcher


def _extract_candidates_from_doc(doc, matcher) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        candidates.append({"text": span.text, "context": span.sent.text})

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if len(chunk_text) < 2:
            continue
        if re.search(r"[\+\#\.]", chunk_text) or chunk_text.isupper():
            candidates.append({"text": chunk_text, "context": chunk.sent.text})
    return candidates


def extract_skill_mentions(text: str, known_skills: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """
    Extract skills with contexts and confidence using spaCy when available.

    Returns a list of dicts with keys: skill, mentions, context, confidence.
    """
    if not text or not text.strip():
        return []

    nlp = get_nlp()
    if nlp is None:
        return _fallback_extract(text, known_skills)

    matcher = _build_phrase_matcher(nlp, known_skills)
    doc = nlp(text)
    candidates = _extract_candidates_from_doc(doc, matcher)

    aggregated: Dict[str, Dict[str, object]] = {}
    for item in candidates:
        normalized = normalize_skill_name(item["text"])
        if not normalized:
            continue
        entry = aggregated.setdefault(
            normalized,
            {"skill": normalized, "mentions": 0, "context": [], "confidence": 0.7},
        )
        entry["mentions"] += 1
        if item["context"] not in entry["context"]:
            entry["context"].append(item["context"])

    results: List[Dict[str, object]] = []
    for entry in aggregated.values():
        base = 0.9 if entry["skill"] in SKILL_LEXICON else 0.75
        boost = min(entry["mentions"] * 0.02, 0.08)
        entry["confidence"] = round(min(base + boost, 0.98), 2)
        results.append(entry)

    return sorted(results, key=lambda x: (-x["mentions"], x["skill"]))


def _fallback_extract(text: str, known_skills: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Regex-based fallback when spaCy is unavailable."""
    candidates = known_skills or SKILL_LEXICON
    text_lower = text.lower()
    aggregated: Dict[str, Dict[str, object]] = {}
    for skill in candidates:
        if not skill:
            continue
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        matches = re.findall(pattern, text_lower)
        if not matches:
            continue
        normalized = normalize_skill_name(skill)
        aggregated[normalized] = {
            "skill": normalized,
            "mentions": len(matches),
            "context": [],
            "confidence": 0.6,
        }
    return list(aggregated.values())
