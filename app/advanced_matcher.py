import logging
from functools import lru_cache
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, util

from app.enhanced_utils import experience_match_score
from app.utils_ner import normalize_skill_name

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_sbert_model() -> SentenceTransformer:
    """Load SBERT model once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def _encode_texts(texts: List[str], model: SentenceTransformer):
    return model.encode(texts, convert_to_tensor=True)


def semantic_skill_matches(
    jd_skills: List[Dict[str, object]],
    resume_skills: List[Dict[str, object]],
    model: SentenceTransformer,
    threshold: float = 0.75,
) -> List[Dict[str, object]]:
    """
    Match JD skills to resume skills using embeddings.
    """
    jd_names = [normalize_skill_name(skill["skill"]) for skill in jd_skills]
    resume_names = [normalize_skill_name(skill["skill"]) for skill in resume_skills]

    if not jd_names or not resume_names:
        return []

    jd_embeddings = _encode_texts(jd_names, model)
    resume_embeddings = _encode_texts(resume_names, model)
    similarity_matrix = util.cos_sim(jd_embeddings, resume_embeddings)

    matches: List[Dict[str, object]] = []
    for i, jd_skill in enumerate(jd_names):
        best_idx = int(similarity_matrix[i].argmax().item())
        best_score = float(similarity_matrix[i][best_idx].item())
        matched = best_score >= threshold
        matches.append({
            "jd_skill": jd_skill,
            "resume_mention": resume_names[best_idx],
            "similarity": round(best_score, 2),
            "matched": matched,
            "reason": "Semantic similarity"
            if matched
            else "Below similarity threshold",
        })
    return matches


def compute_weighted_match_pct(
    jd_skills: List[Dict[str, object]], matches: List[Dict[str, object]]
) -> Dict[str, object]:
    """Compute weighted match percentage and missing buckets."""
    matched_set = {m["jd_skill"].lower() for m in matches if m.get("matched")}
    total_weight = sum(int(s.get("weight", 0)) for s in jd_skills)
    matched_weight = 0
    missing_critical: List[str] = []
    missing_preferred: List[str] = []

    for skill in jd_skills:
        name = str(skill.get("skill", ""))
        weight = int(skill.get("weight", 0))
        requirement = skill.get("requirement", "required")
        if name.lower() in matched_set:
            matched_weight += weight
        else:
            if requirement in ("must-have", "required"):
                missing_critical.append(name)
            else:
                missing_preferred.append(name)

    pct = (matched_weight / total_weight * 100) if total_weight else 0
    return {
        "matched_weight": matched_weight,
        "total_weight": total_weight,
        "weighted_match_pct": round(pct, 2),
        "missing_critical": missing_critical,
        "missing_preferred": missing_preferred,
    }


def _build_reasoning(semantic_score: float, skill_pct: float, exp_score: float) -> str:
    return (
        f"Strong semantic match ({semantic_score:.0f}%) with "
        f"skill coverage at {skill_pct:.0f}%, "
        f"and experience alignment {exp_score:.0f}%."
    )


def match_resumes_advanced(
    job_description: str,
    resumes: List[Dict[str, object]],
    jd_skill_profile: Dict[str, object],
    jd_experience: Dict[str, object],
    weights: Dict[str, float],
    model: Optional[SentenceTransformer] = None,
) -> List[Dict[str, object]]:
    """
    Multi-criteria ranking with semantic similarity, skill match, and experience.
    """
    model = model or get_sbert_model()
    results: List[Dict[str, object]] = []

    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    jd_skills = jd_skill_profile.get("jd_skills", [])

    for resume in resumes:
        resume_text = resume.get("text", "")
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        semantic_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100

        matches = semantic_skill_matches(jd_skills, resume.get("skills", []), model)
        weighted_match = compute_weighted_match_pct(jd_skills, matches)

        exp_score = experience_match_score(jd_experience, resume.get("experience", {}))

        final_score = (
            semantic_score * weights["semantic"]
            + weighted_match["weighted_match_pct"] * weights["skills"]
            + exp_score * weights["experience"]
        )

        results.append({
            "filename": resume.get("filename"),
            "final_score": round(final_score, 2),
            "breakdown": {
                "semantic_score": round(semantic_score, 2),
                "semantic_weight": weights["semantic"],
                "semantic_contribution": round(semantic_score * weights["semantic"], 2),
                "skills_match_pct": weighted_match["weighted_match_pct"],
                "skills_weight": weights["skills"],
                "skills_contribution": round(
                    weighted_match["weighted_match_pct"] * weights["skills"], 2
                ),
                "experience_match": exp_score,
                "experience_weight": weights["experience"],
                "experience_contribution": round(exp_score * weights["experience"], 2),
            },
            "matches": matches,
            "skill_weighted": weighted_match,
            "reasoning": _build_reasoning(
                semantic_score, weighted_match["weighted_match_pct"], exp_score
            ),
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
