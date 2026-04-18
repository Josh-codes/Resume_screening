# app/main.py

import streamlit as st
import os
import shutil
import uuid
import pandas as pd
from app.utils import (
    load_all_resumes,
    extract_skills,
    extract_text_from_pdf_or_txt,
    extract_skills_from_jd
)
from app.matcher import match_resumes

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI-Powered Resume Screener with SBERT + Skill Insights")

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


jd_file = st.file_uploader("📄 Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])
resume_files = st.file_uploader("📂 Upload Resumes (PDF or TXT)", type=[
                                "pdf", "txt"], accept_multiple_files=True)


def prepare_data(jd_path, resumes_paths):
    job_desc = extract_text_from_pdf_or_txt(jd_path)
    jd_skills = extract_skills_from_jd(job_desc)
    resumes_data = []
    for path in resumes_paths:
        text = extract_text_from_pdf_or_txt(path)
        skills_found = extract_skills(text, jd_skills)
        resumes_data.append({
            "filename": os.path.basename(path),
            "text": text,
            "skills": skills_found
        })
    return job_desc, jd_skills, resumes_data


if jd_file and resume_files:
    jd_path = save_uploaded_file_with_unique_name(jd_file, jd_dir)
    resume_paths = []
    for f in resume_files:
        p = save_uploaded_file_with_unique_name(f, resume_dir)
        resume_paths.append(p)

    job_desc, jd_skills, resumes_data = prepare_data(jd_path, resume_paths)

    if not jd_skills:
        st.warning("No skills were extracted from the job description. Missing skills will be empty.")

    with st.spinner("Matching resumes using SBERT..."):
        results = match_resumes(job_desc, resumes_data)

    st.subheader("🔍 Matching Results (Semantic Similarity):")
    data_for_csv = []
    for r in results:
        missing_skills = list(set(jd_skills) - set(r.get('skills', [])))
        st.markdown(f"**{r['filename']}** — Score: `{r['score']}%`")
        st.write(f"Skills Found: {', '.join(r.get('skills', [])) or 'None'}")
        if missing_skills:
            st.write(f"❌ Missing Skills: {', '.join(missing_skills)}")
        st.markdown("---")

        # Prepare row for CSV export
        data_for_csv.append({
            "Filename": r['filename'],
            "Match Score (%)": r['score'],
            "Skills Found": ", ".join(r.get('skills', [])),
            "Missing Skills": ", ".join(missing_skills)
        })

    df = pd.DataFrame(data_for_csv)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Match Report as CSV",
        data=csv_data,
        file_name="resume_match_report.csv",
        mime="text/csv"
    )

elif not jd_file or not resume_files:
    st.info("⬆️ Please upload both job description and resumes to begin matching.")

# Clear button to remove uploaded files + reset UI state
if st.button("🧹 Clear Uploaded Files"):
    shutil.rmtree(resume_dir, ignore_errors=True)
    shutil.rmtree(jd_dir, ignore_errors=True)
    os.makedirs(resume_dir, exist_ok=True)
    os.makedirs(jd_dir, exist_ok=True)

    # Reset file uploader session states
    for key in st.session_state.keys():
        del st.session_state[key]

    st.success("Uploaded files cleared.")
    st.rerun()
