import os
import zipfile

base_dir = "sample_data"
resumes_dir = os.path.join(base_dir, "sample_resumes")
job_desc_dir = os.path.join(base_dir, "job_descriptions")

os.makedirs(resumes_dir, exist_ok=True)
os.makedirs(job_desc_dir, exist_ok=True)

resume_texts = {
    "resume_john_doe.txt": """John Doe
Software Engineer with 5 years experience in Python, AWS, Docker, and Kubernetes.
Skilled in Machine Learning and Data Analysis.
Contact: john.doe@example.com""",
    "resume_jane_smith.txt": """Jane Smith
Data Scientist with expertise in Python, SQL, Machine Learning, and Data Visualization.
Experience with TensorFlow and AWS.
Contact: jane.smith@example.com""",
    "resume_alex_kumar.txt": """Alex Kumar
DevOps Engineer experienced in CI/CD pipelines, Docker, Kubernetes, and AWS.
Familiar with Jenkins and Git.
Contact: alex.kumar@example.com""",
    "resume_rita_sharma.txt": """Rita Sharma
Frontend Developer skilled in JavaScript, React, HTML, and CSS.
Worked on multiple web applications and UI/UX design.
Contact: rita.sharma@example.com"""
}

job_desc_texts = {
    "job_software_engineer.txt": """We are looking for a Software Engineer skilled in Python, AWS, Docker, and Kubernetes.
Experience with Machine Learning is a plus.
Responsibilities include developing scalable backend systems.""",
    "job_data_scientist.txt": """Seeking a Data Scientist proficient in Python, SQL, and Machine Learning.
Experience with data visualization tools and TensorFlow is required.
You will work on analyzing large datasets to drive business decisions."""
}

for filename, content in resume_texts.items():
    with open(os.path.join(resumes_dir, filename), "w") as f:
        f.write(content)

for filename, content in job_desc_texts.items():
    with open(os.path.join(job_desc_dir, filename), "w") as f:
        f.write(content)

zip_path = "sample_data.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, _, files in os.walk(resumes_dir):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=f"sample_resumes/{file}")
    for root, _, files in os.walk(job_desc_dir):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=f"job_descriptions/{file}")

print(f"Created sample zip archive at {zip_path}")
