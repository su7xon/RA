import os

nlp = None
try:
    import spacy
    _nlp = None

    def get_nlp():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_lg")
            except OSError:
                try:
                    _nlp = spacy.load("en_core_web_md")
                except OSError:
                    _nlp = spacy.load("en_core_web_sm")
        return _nlp
except ImportError:
    spacy = None

skills_db = None
try:
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("skills_db", os.path.join(os.path.dirname(__file__), "skills_db.py"))
    if spec and spec.loader:
        skills_db = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(skills_db)
except Exception:
    skills_db = None


def extract_skills_robust(text: str):
    manual_skills = {}
    if skills_db is not None:
        manual_skills = skills_db.extract_skills_from_text(text)
        manual_skills = {k: v for k, v in manual_skills.items()}

    if spacy is None:
        return manual_skills

    try:
        nlp_model = get_nlp()
        if nlp_model is None:
            return manual_skills
    except Exception:
        return manual_skills

    doc = nlp_model(text)

    extracted = set()
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART"):
            extracted.add(ent.text.lower().strip())

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            if any(t.lower_ in ("using", "with", "in", "developed", "built", "proficient", "expert", "skilled") for t in token.children):
                extracted.add(token.text.lower().strip())

    for cat, skills in manual_skills.items():
        for skill in skills:
            for ext in extracted:
                if skill in ext or ext in skill:
                    skills.append(ext)
        manual_skills[cat] = list(set(skills))

    return manual_skills


if __name__ == "__main__":
    sample = """
    Experiences:
    - Backend development with Node.js and Express.
    - Database: MongoDB, PostgreSQL.
    - DevOps: Docker, Kubernetes, AWS.
    """
    print(extract_skills_robust(sample))
