import PyPDF2
import os
import re
import json
import io
import traceback
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from markupsafe import Markup
from groq import Groq

GROQ_KEYS = []
if os.getenv("GROQ_API_KEY"):
    GROQ_KEYS.append(os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.3-70b-versatile"
groq_clients = [Groq(api_key=key) for key in GROQ_KEYS if key]

import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
google_client = None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    google_client = genai.GenerativeModel('gemini-pro')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print(f"Warning: sklearn/numpy could not be loaded: {e}")
    TfidfVectorizer = None
    cosine_similarity = None

try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    print(f"Warning: spaCy could not be loaded: {e}")
    nlp = None

try:

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    print("Warning: sentence-transformers not found. Please run 'python -m pip install sentence-transformers'")
    st_model = None

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not found. Using PyPDF2 as fallback. Install with: pip install pymupdf")

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not found. Using PyPDF2 as fallback. Install with: pip install pymupdf")

_skills_extractor = None

def _get_skills_extractor():
    global _skills_extractor
    if _skills_extractor is None:
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "local_skills_db",
                os.path.join(os.path.dirname(__file__), "skills_db.py")
            )
            if spec and spec.loader:
                local_skills_db = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_skills_db)
                _skills_extractor = local_skills_db
        except Exception as e:
            print(f"[warn] skills_db import failed: {e}")
            _skills_extractor = False
    return _skills_extractor


def _extract_local_skills(resume_text: str):
    extractor = _get_skills_extractor()
    if extractor and hasattr(extractor, "extract_skills_from_text"):
        try:
            return extractor.extract_skills_from_text(resume_text)
        except Exception as e:
            print(f"[warn] local skill extraction failed: {e}")
    return {}


def analyze_resume_with_groq(resume_text, job_description):
    """Primary AI analysis using Groq."""
    if not groq_clients:
        return None

    prompt = (
        "You are an expert HR AI assistant with deep knowledge of technical recruiting.\n"
        "Analyze the following candidate's resume against the job description.\n"
        "Be thorough, specific, and honest.\n\n"
        "Job Description:\n" + str(job_description) + "\n\n"
        "Candidate Resume:\n" + str(resume_text) + "\n\n"
        "Respond ONLY in this JSON format (using short keys to save tokens):\n"
        "{\n"
        "    \"s\": <integer 0-100 score>,\n"
        "    \"sb\": {\"sk\": <0-100>,\"ex\": <0-100>,\"ed\": <0-100>,\"of\": <0-100>},\n"
        "    \"bd\": \"2-3 sentence summary\",\n"
        "    \"ey\": \"total years\",\n"
        "    \"f\": [\"Frontend skills\"],\n"
        "    \"b\": [\"Backend skills\"],\n"
        "    \"d\": [\"Database skills\"],\n"
        "    \"do\": [\"DevOps skills\"],\n"
        "    \"c\": [\"Cloud skills\"],\n"
        "    \"os\": [\"Other tech/soft skills\"],\n"
        "    \"fr\": <1-10 rating>,\n"
        "    \"br\": <1-10 rating>,\n"
        "    \"dr\": <1-10 rating>,\n"
        "    \"dor\": <1-10 rating>,\n"
        "    \"cr\": <1-10 rating>,\n"
        "    \"mk\": [\"found keywords\"],\n"
        "    \"ms\": [\"missing keywords\"],\n"
        "    \"es\": [\"all extracted skills\"],\n"
        "    \"r\": \"detailed reasoning\",\n"
        "    \"st\": [\"strengths\"],\n"
        "    \"g\": [\"gaps\"],\n"
        "    \"ea\": \"experience summary\",\n"
        "    \"re\": \"Strong Hire | Hire | Maybe | Reject\"\n"
        "}"
    )

    for i, client in enumerate(groq_clients):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1024
            )
            result = json.loads(response.choices[0].message.content)
            result['ai_source'] = 'Groq (Llama 3.3 70B)'
            return result
        except Exception as e:
            print(f"Groq API Error with key {i+1}: {e}")
            if i == len(groq_clients) - 1:
                print("All Groq keys failed. Trying Google fallback...")
                return None

def analyze_resume_with_google(resume_text, job_description):
    """Fallback AI analysis using Google Gemini."""
    if not google_client:
        return None

    prompt = (
        "Analyze this candidate for the role. Respond ONLY in JSON:\n\n"
        "Job Description:\n" + str(job_description) + "\n\n"
        "Resume:\n" + str(resume_text) + "\n\n"
        "Respond ONLY in this JSON format (using short keys to save tokens):\n"
        "{\n"
        "    \"s\": <integer 0-100 score>,\n"
        "    \"sb\": {\"sk\": <0-100>,\"ex\": <0-100>,\"ed\": <0-100>,\"of\": <0-100>},\n"
        "    \"bd\": \"2-3 sentence summary\",\n"
        "    \"ey\": \"total years\",\n"
        "    \"f\": [\"Frontend skills\"],\n"
        "    \"b\": [\"Backend skills\"],\n"
        "    \"d\": [\"Database skills\"],\n"
        "    \"do\": [\"DevOps skills\"],\n"
        "    \"c\": [\"Cloud skills\"],\n"
        "    \"os\": [\"Other tech/soft skills\"],\n"
        "    \"fr\": <1-10 rating>,\n"
        "    \"br\": <1-10 rating>,\n"
        "    \"dr\": <1-10 rating>,\n"
        "    \"dor\": <1-10 rating>,\n"
        "    \"cr\": <1-10 rating>,\n"
        "    \"mk\": [\"found keywords\"],\n"
        "    \"ms\": [\"missing keywords\"],\n"
        "    \"es\": [\"all extracted skills\"],\n"
        "    \"r\": \"detailed reasoning\",\n"
        "    \"st\": [\"strengths\"],\n"
        "    \"g\": [\"gaps\"],\n"
        "    \"ea\": \"experience summary\",\n"
        "    \"re\": \"Strong Hire | Hire | Maybe | Reject\"\n"
        "}"
    )

    try:
        response = google_client.generate_content(prompt)
        text = response.text
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]

        result = json.loads(text.strip())
        result['ai_source'] = 'Google (Gemini 1.5 Flash)'
        return result
    except Exception as e:
        print(f"Google API Error: {e}. Falling back to local scoring...")
        return None


def redact_bias(text):
    if not nlp:
        return text
    doc = nlp(text)
    words = []
    for token in doc:
        if token.ent_type_ in ['PERSON', 'NORP', 'GPE']:
            words.append('[REDACTED]' + token.whitespace_)
        else:
            words.append(token.text_with_ws)
    return "".join(words)

_skills_extractor = None

def _get_skills_extractor():
    global _skills_extractor
    if _skills_extractor is None:
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "local_skills_db",
                os.path.join(os.path.dirname(__file__), "skills_db.py")
            )
            if spec and spec.loader:
                local_skills_db = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_skills_db)
                _skills_extractor = local_skills_db
        except Exception as e:
            print(f"[warn] skills_db import failed: {e}")
            _skills_extractor = False
    return _skills_extractor


def _extract_local_skills(resume_text: str):
    extractor = _get_skills_extractor()
    if extractor and hasattr(extractor, "extract_skills_from_text"):
        try:
            return extractor.extract_skills_from_text(resume_text)
        except Exception as e:
            print(f"[warn] local skill extraction failed: {e}")
    return {}


def extract_keywords_from_jd(jd_text):
    if ',' in jd_text and all(len(kw.strip().split()) <= 4 for kw in jd_text.split(',')):
        return list(set([kw.strip() for kw in jd_text.split(',') if kw.strip()]))
    else:
        if nlp:
            doc = nlp(jd_text)
            return list(set([chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]))
        else:
            return list(set([word for word in jd_text.split() if len(word) > 4]))

def check_keyword_matches(resume_text, keywords):
    resume_text_lower = resume_text.lower()
    matched = []
    missing = []
    for kw in keywords:
        if kw.lower() in resume_text_lower:
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing

def extract_text_from_pdf_stream(file_stream):
    text = []
    
    if PYMUPDF_AVAILABLE:
        try:
            file_stream.seek(0)
            doc = fitz.open(stream=file_stream.read(), filetype="pdf")
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return " ".join(text)
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}. Falling back to PyPDF2...")
            file_stream.seek(0)

    try:
        file_stream.seek(0)
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return " ".join(text)
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
        return ""

def extract_years_of_experience(resume_text):
    total_years = 0.0
    year_pattern = re.compile(
        r"(?P<years>\d+(?:\.\d+)?)(?:\+|-)?\s*(?:years?|yrs?\.?)?\s*(?:of\s*)?experience|"
        r"(?P<start_month>[a-zA-Z]{3,9})?\s*(?P<start_year>\d{4})\s*[-to]+\s*(?P<end_month>[a-zA-Z]{3,9})?\s*(?P<end_year>\d{4})\s*(?:experience)?",
        re.IGNORECASE
    )
    for match in year_pattern.finditer(resume_text):
        if match.group("years"):
            total_years += float(match.group("years"))
        elif match.group("start_year") and match.group("end_year"):
            start_year = int(match.group("start_year"))
            end_year = int(match.group("end_year"))
            start_month = match.group("start_month") if match.group("start_month") else "Jan"
            end_month = match.group("end_month") if match.group("end_month") else "Dec"
            try:
                start_date = datetime.strptime(f"{start_month} {start_year}", "%b %Y")
                end_date = datetime.strptime(f"{end_month} {end_year}", "%b %Y")
                total_years += (end_date - start_date).days / 365.25
            except ValueError:
                pass
    return total_years

def score_resume_with_embeddings(resume_text, job_description):
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    if TfidfVectorizer:
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            vectors = tfidf.fit_transform([resume_text, job_description])
            cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]
        except ValueError:
            cosine_sim = 0.0
    else:
        # Pure python fallback if sklearn is blocked
        matched, missing = check_keyword_matches(resume_text, job_description.split())
        total_len = len(matched) + len(missing)
        cosine_sim = len(matched) / total_len if total_len > 0 else 0.0

    if st_model:
        resume_embedding = st_model.encode([resume_text])
        jd_embedding = st_model.encode([job_description])
        embedding_similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
    elif nlp:
        resume_embedding = nlp(resume_text).vector
        jd_embedding = nlp(job_description).vector
        embedding_similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    else:
        embedding_similarity = 0.0

    return (cosine_sim * 0.4) + (embedding_similarity * 0.6)

def process_resumes(job_description, uploaded_files):
    candidates = []
    jd_keywords = extract_keywords_from_jd(job_description)

    for file in uploaded_files:
        if file.filename == '':
            continue

        filename = file.filename
        file_stream = io.BytesIO(file.read())

        try:
            raw_resume_text = extract_text_from_pdf_stream(file_stream)
            if not raw_resume_text or not raw_resume_text.strip():
                print(f"Warning: No text extracted from {filename}")
                continue
            resume_text = redact_bias(raw_resume_text)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            traceback.print_exc()
            continue

        local_matched, local_missing = check_keyword_matches(resume_text, jd_keywords)

        local_skills = _extract_local_skills(resume_text)

        ai_result = analyze_resume_with_groq(resume_text, job_description)
        if not ai_result:
            ai_result = analyze_resume_with_google(resume_text, job_description)

        if ai_result:
            display_score = ai_result.get('s', ai_result.get('score', 0))
            score_breakdown = ai_result.get('sb', ai_result.get('score_breakdown', {}))
            explanation = ai_result.get('r', ai_result.get('reasoning', 'No explanation provided.'))
            strengths = ai_result.get('st', ai_result.get('strengths', []))
            gaps = ai_result.get('g', ai_result.get('gaps', []))
            recommendation = ai_result.get('re', ai_result.get('recommendation', 'N/A'))
            experience_assessment = ai_result.get('ea', ai_result.get('experience_assessment', 'N/A'))
            ai_source = ai_result.get('ai_source', 'Unknown')

            score = display_score / 100.0 * 10.0

            brief_description = ai_result.get('bd', ai_result.get('brief_description', 'No description available.'))
            experience_years = ai_result.get('ey', ai_result.get('experience_years', 'N/A'))
            frontend = list({s.lower(): s for s in ai_result.get('f', ai_result.get('frontend', [])) + local_skills.get('frontend', [])}.values())
            backend = list({s.lower(): s for s in ai_result.get('b', ai_result.get('backend', [])) + local_skills.get('backend', [])}.values())
            database = list({s.lower(): s for s in ai_result.get('d', ai_result.get('database', [])) + local_skills.get('database', [])}.values())
            devops = list({s.lower(): s for s in ai_result.get('do', ai_result.get('devops', [])) + local_skills.get('devops', [])}.values())
            cloud = list({s.lower(): s for s in ai_result.get('c', ai_result.get('cloud', [])) + local_skills.get('cloud', [])}.values())
            other_skills = list({s.lower(): s for s in ai_result.get('os', ai_result.get('other_skills', []))}.values())

            try:
                from auto_categorizer import auto_categorize_skills
                all_ai_skills = ai_result.get('es', ai_result.get('extracted_skills', []))
                already_categorized = set(frontend + backend + database + devops + cloud + other_skills)
                uncategorized = [s for s in all_ai_skills if s not in already_categorized]
                if uncategorized:
                    auto_cats = auto_categorize_skills(uncategorized)
                    frontend.extend(auto_cats.get('frontend', []))
                    backend.extend(auto_cats.get('backend', []))
                    database.extend(auto_cats.get('database', []))
                    devops.extend(auto_cats.get('devops', []))
                    cloud.extend(auto_cats.get('cloud', []))
                    other_skills.extend(auto_cats.get('other', []))
                    
                    frontend = list({s.lower(): s for s in frontend}.values())
                    backend = list({s.lower(): s for s in backend}.values())
                    database = list({s.lower(): s for s in database}.values())
                    devops = list({s.lower(): s for s in devops}.values())
                    cloud = list({s.lower(): s for s in cloud}.values())
                    other_skills = list({s.lower(): s for s in other_skills}.values())
            except Exception as e:
                pass
            frontend_rating = ai_result.get('fr', ai_result.get('frontend_rating', 0))
            backend_rating = ai_result.get('br', ai_result.get('backend_rating', 0))
            database_rating = ai_result.get('dr', ai_result.get('database_rating', 0))
            devops_rating = ai_result.get('dor', ai_result.get('devops_rating', 0))
            cloud_rating = ai_result.get('cr', ai_result.get('cloud_rating', 0))

            ai_matched = ai_result.get('mk', ai_result.get('matched_keywords', []))
            ai_missing = ai_result.get('ms', ai_result.get('missing_keywords', []))
            ai_extracted = ai_result.get('es', ai_result.get('extracted_skills', []))

            merged_matched = list({kw.lower(): kw for kw in local_matched + ai_matched}.values())
            merged_missing = list({kw.lower(): kw for kw in local_missing + ai_missing}.values())
            merged_matched_lower = {kw.lower() for kw in merged_matched}
            merged_missing = [kw for kw in merged_missing if kw.lower() not in merged_matched_lower]
        else:
            base_score = score_resume_with_embeddings(resume_text, job_description)
            years = extract_years_of_experience(resume_text)

            exp_score = 1 if years >= 5 else (0.5 if years >= 3 else 0)
            base_score += exp_score

            score = base_score * 5.0

            display_score = min(max(int(score * 10), 0), 100)
            explanation = f"Score based on {len(local_matched)} matched keywords. Local NLP only."
            strengths = []
            gaps = local_missing
            recommendation = 'N/A'
            experience_assessment = f"{years} years"
            brief_description = 'N/A'
            experience_years = experience_assessment
            frontend = local_skills.get('frontend', [])
            backend = local_skills.get('backend', [])
            database = local_skills.get('database', [])
            devops = local_skills.get('devops', [])
            cloud = local_skills.get('cloud', [])
            other_skills = []
            try:
                from auto_categorizer import auto_categorize_skills
                raw_extracted = local_skills.get('extracted_skills', [])
                if raw_extracted:
                    already = set(frontend + backend + database + devops + cloud)
                    remaining = [s for s in raw_extracted if s not in already]
                    if remaining:
                        cats = auto_categorize_skills(remaining)
                        frontend.extend(cats.get('frontend', []))
                        backend.extend(cats.get('backend', []))
                        database.extend(cats.get('database', []))
                        devops.extend(cats.get('devops', []))
                        cloud.extend(cats.get('cloud', []))
                        other_skills = cats.get('other', [])
            except Exception:
                pass
            frontend_rating = 0
            backend_rating = 0
            database_rating = 0
            devops_rating = 0
            cloud_rating = 0
            ai_source = 'Local NLP (No AI)'
            score_breakdown = {}
            merged_matched = list({kw.lower(): kw for kw in local_matched}.values())
            merged_missing = list({kw.lower(): kw for kw in local_missing}.values())
            ai_extracted = []

        candidates.append({
            'filename': filename,
            'score': score,
            'display_score': display_score,
            'brief_description': brief_description,
            'experience_years': experience_years,
            'frontend': frontend,
            'backend': backend,
            'database': database,
            'devops': devops,
            'cloud': cloud,
            'other_skills': other_skills,
            'frontend_rating': frontend_rating,
            'backend_rating': backend_rating,
            'database_rating': database_rating,
            'devops_rating': devops_rating,
            'cloud_rating': cloud_rating,
            'score_breakdown': score_breakdown if ai_result else {},
            'matched_keywords': merged_matched,
            'missing_keywords': merged_missing,
            'extracted_skills': ai_extracted,
            'explanation': explanation,
            'strengths': strengths,
            'gaps': gaps,
            'recommendation': recommendation,
            'ai_source': ai_source
        })

        file.seek(0)

    return sorted(candidates, key=lambda x: x['score'], reverse=True)
