# Resume Analyzer + Job Matcher (Streamlit App with Realistic ATS Scoring)

import streamlit as st
import fitz  # PyMuPDF
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Caching models ---
@st.cache_resource
def load_models():
    return spacy.load("en_core_web_sm"), SentenceTransformer('all-MiniLM-L6-v2')

nlp, embedder = load_models()

# --- Common Skills ---
@st.cache_data
def get_common_skills():
    return set("""
    python sql power bi excel pandas numpy data analysis machine learning deep learning communication teamwork
    tensorflow keras matplotlib scikit-learn tableau r flask nlp problem solving java c++ c# scala javascript typescript
    seaborn scipy dplyr tidyverse statsmodels lookml qlikview ggplot2 dash plotly looker xgboost lightgbm catboost pytorch
    supervised learning unsupervised learning reinforcement learning model evaluation cross-validation hyperparameter tuning
    model deployment cnn rnn transformers autoencoders bert gpt llm text mining spacy nltk text classification named entity recognition
    sentiment analysis language modeling topic modeling tf-idf word2vec mysql postgresql mssql oracle nosql mongodb redshift bigquery
    hive snowflake dynamodb database design joins window functions aws azure gcp docker kubernetes lambda sagemaker ec2 cloud storage
    mlops ci/cd jenkins airflow databricks terraform hadoop spark pyspark kafka flink hdfs etl data pipeline data warehouse
    django fastapi streamlit gradio html css api development data cleaning data wrangling feature engineering data preprocessing
    data modeling data governance data validation schema design git github bitbucket jira notion postman linux bash shell scripting
    uipath automation anywhere blue prism workflow automation robotic process automation orchestrator reframework selectors
    citrix automation mainframe automation leadership collaboration time management critical thinking adaptability attention to detail
    presentation skills stakeholder management business acumen project management
    """.split())

common_skills = get_common_skills()

# --- Helper Functions ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_skills_local(text):
    doc = nlp(text.lower())
    extracted = set()
    for chunk in doc.noun_chunks:
        skill = chunk.text.strip()
        if skill in common_skills:
            extracted.add(skill)
    for token in doc:
        if token.text.strip() in common_skills:
            extracted.add(token.text.strip())
    return extracted

def calculate_ats_score(resume_text, jd_required_skills, job_title=None, semantic_score=None):
    resume_text_lower = resume_text.lower()
    resume_skills = extract_skills_local(resume_text)

    # 1. Keyword Match Score (40%)
    if jd_required_skills:
        keyword_score = round(len(resume_skills & jd_required_skills) / len(jd_required_skills) * 100, 2)
    else:
        keyword_score = 0

    # 2. Section Score (15%)
    required_sections = ['experience', 'skills', 'education', 'certification', 'projects', 'summary', 'objective']
    section_hits = sum(1 for section in required_sections if section in resume_text_lower)
    section_score = round((section_hits / len(required_sections)) * 100, 2)

    # 3. Job Title Score (15%)
    if job_title:
        job_words = set(re.findall(r'\w+', job_title.lower()))
        resume_words = set(re.findall(r'\w+', resume_text_lower))
        title_score = round(len(job_words & resume_words) / len(job_words) * 100, 2) if job_words else 0
    else:
        title_score = 0

    # 4. Semantic Score (30%)
    sem_score = semantic_score if semantic_score is not None else 0

    # Final Score
    ats_score = (
        0.4 * keyword_score +
        0.15 * section_score +
        0.15 * title_score +
        0.3 * sem_score
    )
    return round(ats_score, 2)

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Matcher", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; border-radius: 12px; padding: 20px; }
    .block-container { padding: 2rem 2rem 0rem 2rem; }
    .stButton>button { border-radius: 10px; background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Resume Analyzer + Job Matcher")
st.markdown("Upload your resume and compare it against multiple job descriptions for best fit.")

resume_file = st.file_uploader("📎 Upload Resume (PDF only)", type=["pdf"])
jd_input = st.text_area("📝 Paste Job Descriptions (format: Title | Company | Description, one per line)", height=250)

if resume_file and jd_input:
    with st.spinner("🔍 Analyzing Resume and Matching Jobs..."):
        resume_text = extract_text_from_pdf(resume_file)
        resume_embed = embedder.encode([resume_text])[0]
        resume_skills = extract_skills_local(resume_text)

        # Parse JDs
        job_descriptions = []
        for line in jd_input.strip().splitlines():
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) == 3:
                job_descriptions.append({"title": parts[0], "company": parts[1], "description": parts[2]})

        jd_texts = [jd["description"] for jd in job_descriptions]
        jd_embeds = embedder.encode(jd_texts, show_progress_bar=False)
        similarity = cosine_similarity([resume_embed], jd_embeds)[0]
        job_skills_list = [extract_skills_local(jd["description"]) for jd in job_descriptions]

        results = []
        for i, jd in enumerate(job_descriptions):
            matched = resume_skills.intersection(job_skills_list[i])
            missing = job_skills_list[i].difference(resume_skills)
            skill_score = round(len(matched) / len(job_skills_list[i]) * 100, 2) if job_skills_list[i] else 0
            sem_score = similarity[i] * 100
            ats_score = calculate_ats_score(resume_text, job_skills_list[i], jd["title"], sem_score)

            results.append({
                "title": jd["title"],
                "company": jd["company"],
                "matched_skills": matched,
                "missing_skills": missing,
                "semantic_score": round(sem_score, 2),
                "skill_score": skill_score,
                "final_score": round(0.7 * sem_score + 0.3 * skill_score, 2),
                "ats_score": ats_score,
                "description": jd["description"][:200] + "..."
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        st.success("✅ Resume Analysis Complete")

        for res in results[:5]:
            st.markdown(f"### {res['title']} at {res['company']}")
            st.markdown(f"**Match Score**: {res['final_score']}%")
            st.markdown(f"- **ATS Score**: {res['ats_score']}%")
            st.markdown(f"- **Semantic Score**: {res['semantic_score']}%")
            st.markdown(f"- **Skill Score**: {res['skill_score']}%")
            st.markdown(f"- **Matched Skills**: {', '.join(res['matched_skills']) if res['matched_skills'] else 'None'}")
            st.markdown(f"- **Missing Skills**: {', '.join(res['missing_skills']) if res['missing_skills'] else 'None'}")
            st.markdown(f"**Description**: {res['description']}")
            st.markdown("---")
