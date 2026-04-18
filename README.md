#  AI Resume Screener

An AI-powered resume screening tool that intelligently matches candidate resumes with job descriptions using semantic similarity techniques. This project leverages modern NLP models like Sentence-BERT (SBERT) to go beyond keyword matching and understand the context of skills and experience.

---

##  Features

- Upload multiple resumes and a job description via an interactive Streamlit interface
- Uses Sentence-BERT to calculate semantic similarity between resumes and job descriptions
- Scores and ranks resumes based on relevance
- Displays real-time results with a clean and user-friendly UI

---

##  Tech Stack

- **Language**: Python
- **Framework**: Streamlit
- **NLP Model**: Sentence-BERT (SBERT)
- **Libraries**: `sentence-transformers`, `scikit-learn`, `pdfminer.six`, `pandas`


## ⚙️ Getting Started

Follow these steps to run the project locally on your machine.

```bash
# Clone the repository
git clone https://github.com/pragati9998/AI_Resume_Screener.git
cd AI_Resume_Screener

# Create a virtual environment
python -m venv venv

# Activate the virtual environment (choose the command for your OS)

# Linux/macOS:
source venv/bin/activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run app/main.py






