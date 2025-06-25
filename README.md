# Resume-Analyzer-Job-MatcherHere's a professional and visually appealing `README.md` for your **Resume Analyzer + Job Matcher** project:

---

````markdown
# 📄 Resume Analyzer + Job Matcher 🎯

A powerful and intelligent **Streamlit web app** that analyzes your resume, matches it against multiple job descriptions, and calculates an **ATS (Applicant Tracking System) score** — similar to what real hiring systems use!

## 🔍 What it Does

✅ Upload your **PDF resume**  
✅ Paste multiple **job descriptions** (Title | Company | Description)  
✅ Get:
- Top 5 job matches based on **semantic similarity** and **skills overlap**
- Resume's **ATS Score**
- Matched and missing skills
- Individual **semantic** and **skill** scores

---

## 📸 Preview

![App Screenshot](https://streamlit.io/images/brand/streamlit-mark-color.png) <!-- Replace with your own screenshot or GIF -->

---

## 🚀 Try It Live

👉 [Live Demo on Streamlit Cloud](https://share.streamlit.io/your-username/resume-analyzer/main/resume_matcher.py)

---

## 🧠 How It Works

- **Resume Parsing**: Extracts clean text from PDF using `PyMuPDF`
- **Skill Extraction**: Uses `spaCy` NLP and a curated dictionary of 250+ IT/Data/ML skills
- **Semantic Matching**: Computes similarity using `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`)
- **ATS Scoring**: Realistic score using:
  - Keywords (40%)
  - Resume sections (15%)
  - Job title relevance (15%)
  - Semantic relevance (30%)

---

## 📦 Tech Stack

| Category         | Tools Used                          |
|------------------|-------------------------------------|
| 🧠 NLP/ML        | spaCy, Sentence Transformers         |
| 📊 Similarity    | sklearn cosine similarity           |
| 📎 PDF Parsing   | PyMuPDF                             |
| 🌐 Frontend UI   | Streamlit                           |
| 🗂 Deployment     | Streamlit Community Cloud (Free)    |

---

## 🛠️ Setup Instructions

1. **Clone the Repo**
```bash
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run resume_matcher.py
```

---

## 📁 Folder Structure

```
resume-analyzer/
│
├── resume_matcher.py         # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # UI config (optional)
└── README.md
```

---

## 💡 Examples for Job Description Input

```
Data Scientist | ABC Corp | We are looking for someone skilled in Python, ML, and SQL.
ML Engineer | XYZ Inc | Responsibilities include building models and deploying them using TensorFlow.
RPA Analyst | Deloitte | Automate processes using UiPath and Python with deep understanding of workflows.
```

---

## 📈 Future Enhancements

* Resume summary using OpenAI LLMs (GPT)
* CSV download of matched jobs
* Cover letter generator
* Job scraping from LinkedIn/Naukri

---

## 👨‍💻 Author

**Mohd Humaid**
[LinkedIn](https://www.linkedin.com/in/mohdhumaid/) | [GitHub](https://github.com/mohdhumaid)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

```

---

Let me know if you'd like:
- A **custom badge** for ATS score or deployment status
- A **demo video/GIF** embed
- Help uploading this to GitHub with assets and `.streamlit/config.toml`
```
