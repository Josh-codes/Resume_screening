from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def match_resumes(job_description, resumes):
    results = []

    jd_embedding = model.encode(job_description, convert_to_tensor=True)

    for resume in resumes:
        resume_embedding = model.encode(resume["text"], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        results.append({
            "filename": resume["filename"],
            "score": round(similarity * 100, 2),
            "skills": resume.get("skills", [])
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
    