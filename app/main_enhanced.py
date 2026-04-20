import os
import shutil
import uuid
import logging
from typing import Dict, List

import pandas as pd
import streamlit as st

from app.advanced_matcher import get_sbert_model, match_resumes_advanced
from app.enhanced_utils import (
    build_jd_skill_profile,
    extract_experience_info,
    extract_resume_skills,
    extract_text_from_pdf_or_txt,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI-Powered Resume Screener (Enhanced)")

resume_dir = "uploaded_resumes"
jd_dir = "uploaded_jd"
os.makedirs(resume_dir, exist_ok=True)
os.makedirs(jd_dir, exist_ok=True)


def save_uploaded_file_with_unique_name(uploaded_file, folder):
    os.makedirs(folder, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    path = os.path.join(folder, unique_filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


with st.sidebar:
    st.subheader("Scoring Weights")
    semantic_weight = st.slider("Semantic Similarity", 0, 100, 60)
    skills_weight = st.slider("Skills Match", 0, 100, 30)
    experience_weight = st.slider("Experience Match", 0, 100, 10)

    total = max(semantic_weight + skills_weight + experience_weight, 1)
    weights = {
        "semantic": semantic_weight / total,
        "skills": skills_weight / total,
        "experience": experience_weight / total,
    }
    st.caption(
        f"Normalized weights: semantic={weights['semantic']:.2f}, "
        f"skills={weights['skills']:.2f}, experience={weights['experience']:.2f}"
    )

jd_file = st.file_uploader("📄 Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])
resume_files = st.file_uploader(
    "📂 Upload Resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True
)


def _cache_key(path: str) -> str:
    stats = os.stat(path)
    return f"{path}:{stats.st_size}:{stats.st_mtime}"


def prepare_data(jd_path: str, resumes_paths: List[str]):
    job_desc = extract_text_from_pdf_or_txt(jd_path)
    jd_skill_profile = build_jd_skill_profile(job_desc)
    jd_skills = [s["skill"] for s in jd_skill_profile.get("jd_skills", [])]
    jd_experience = extract_experience_info(job_desc)

    resumes_data = []
    skill_cache: Dict[str, List[Dict[str, object]]] = st.session_state.setdefault(
        "resume_skill_cache", {}
    )
    exp_cache: Dict[str, Dict[str, object]] = st.session_state.setdefault(
        "resume_exp_cache", {}
    )

    for path in resumes_paths:
        text = extract_text_from_pdf_or_txt(path)
        key = _cache_key(path)
        if key in skill_cache:
            skills_found = skill_cache[key]
        else:
            skills_found = extract_resume_skills(text, jd_skills)
            skill_cache[key] = skills_found

        if key in exp_cache:
            exp_info = exp_cache[key]
        else:
            exp_info = extract_experience_info(text)
            exp_cache[key] = exp_info

        resumes_data.append(
            {
                "filename": os.path.basename(path),
                "text": text,
                "skills": skills_found,
                "experience": exp_info,
            }
        )

    return job_desc, jd_skill_profile, jd_experience, resumes_data


if jd_file and resume_files:
    jd_path = save_uploaded_file_with_unique_name(jd_file, jd_dir)
    resume_paths = [save_uploaded_file_with_unique_name(f, resume_dir) for f in resume_files]

    job_desc, jd_skill_profile, jd_experience, resumes_data = prepare_data(
        jd_path, resume_paths
    )

    if not jd_skill_profile.get("jd_skills"):
        st.warning("No skills were extracted from the job description. Missing skills will be empty.")

    with st.spinner("Matching resumes using SBERT..."):
        results = match_resumes_advanced(
            job_desc,
            resumes_data,
            jd_skill_profile,
            jd_experience,
            weights,
            model=get_sbert_model(),
        )

    st.subheader("🔍 Matching Results (Composite Score):")
    data_for_csv = []
    for r in results:
        breakdown = r["breakdown"]
        weighted = r["skill_weighted"]
        st.markdown(f"**{r['filename']}** — Final Score: `{r['final_score']}%`")
        st.write(
            "Semantic: {0:.1f}% | Skills: {1:.1f}% | Experience: {2:.1f}%".format(
                breakdown["semantic_score"],
                breakdown["skills_match_pct"],
                breakdown["experience_match"],
            )
        )

        chart_data = pd.DataFrame(
            {
                "Contribution": [
                    breakdown["semantic_contribution"],
                    breakdown["skills_contribution"],
                    breakdown["experience_contribution"],
                ]
            },
            index=["Semantic", "Skills", "Experience"],
        )
        st.bar_chart(chart_data, height=130, use_container_width=True)

        st.write("Missing critical skills:", ", ".join(weighted["missing_critical"]) or "None")
        st.write("Missing preferred skills:", ", ".join(weighted["missing_preferred"]) or "None")

        with st.expander("Why this score?"):
            st.write(r["reasoning"])
            st.write("Skills matched:")
            st.write(
                ", ".join(
                    [
                        f"{m['jd_skill']} ↔ {m['resume_mention']} ({m['similarity']})"
                        for m in r["matches"]
                        if m.get("matched")
                    ]
                )
                or "None"
            )

        st.markdown("---")

        data_for_csv.append(
            {
                "Filename": r["filename"],
                "Final Score (%)": r["final_score"],
                "Semantic Score": breakdown["semantic_score"],
                "Skills Match (%)": breakdown["skills_match_pct"],
                "Experience Match": breakdown["experience_match"],
                "Semantic Contribution": breakdown["semantic_contribution"],
                "Skills Contribution": breakdown["skills_contribution"],
                "Experience Contribution": breakdown["experience_contribution"],
                "Missing Critical": ", ".join(weighted["missing_critical"]),
                "Missing Preferred": ", ".join(weighted["missing_preferred"]),
            }
        )

    df = pd.DataFrame(data_for_csv)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Match Report as CSV",
        data=csv_data,
        file_name="resume_match_report.csv",
        mime="text/csv",
    )

elif not jd_file or not resume_files:
    st.info("⬆️ Please upload both job description and resumes to begin matching.")

if st.button("🧹 Clear Uploaded Files"):
    shutil.rmtree(resume_dir, ignore_errors=True)
    shutil.rmtree(jd_dir, ignore_errors=True)
    os.makedirs(resume_dir, exist_ok=True)
    os.makedirs(jd_dir, exist_ok=True)

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.success("Uploaded files cleared.")
    st.rerun()
