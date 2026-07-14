"""
Comprehensive skill database with synonyms and fuzzy matching.
Ensures MongoDB, PostgreSQL, React etc. are always caught regardless
of spacing / casing / abbreviations in the resume.
"""
import re
from difflib import SequenceMatcher

SKILL_CATEGORIES = {
    "frontend": {
        "skills": [
            "html", "css", "javascript", "js", "typescript", "ts",
            "react", "reactjs", "react.js", "vue", "vuejs", "vue.js",
            "angular", "angularjs", "svelte", "jquery", "bootstrap",
            "tailwind", "tailwindcss", "sass", "scss", "less",
            "webpack", "vite", "babel", "redux", "mobx", "zustand",
            "next", "nextjs", "next.js", "nuxt", "nuxtjs", "nuxt.js",
            "gatsby", "flutter", "react native", "electron", "pwa",
            "web components", "dom", "ajax", "json", "xml",
            "semantic ui", "material ui", "mui", "antd", "chakra ui",
            "styled components", "emotion", "three.js", "threejs", "d3.js", "d3",
        ]
    },
    "backend": {
        "skills": [
            "python", "django", "flask", "fastapi", "tornado", "bottle",
            "nodejs", "node.js", "node", "express", "expressjs", "nestjs", "nest.js",
            "java", "spring", "spring boot", "springboot", "kotlin",
            "php", "laravel", "symfony", "codeigniter", "wordpress",
            "ruby", "rails", "ruby on rails",
            "go", "golang", "gin", "echo",
            "c#", "csharp", "cs", "asp.net", "aspnet", ".net", "dotnet", ".net core", "dotnet core",
            "c++", "cpp",
            "rust", "actix", "rocket",
            "elixir", "phoenix",
            "scala", "play framework", "akka",
            "perl", "cgi",
            "graphql", "rest", "restful", "soap", "grpc",
            "webpack", "babel",
        ]
    },
    "database": {
        "skills": [
            "sql", "mysql", "postgresql", "postgres", "psql", "pl/sql",
            "oracle", "oracle db", "sqlite",
            "mongodb", "mongo db", "mongoose",
            "redis", "memcached",
            "cassandra", "couchdb", "couchbase",
            "neo4j", "dynamodb", "dynamo db", "amazon dynamodb",
            "firebase", "firestore", "realm",
            "elasticsearch", "elastic search", "solr",
            "mariadb", "mssql", "sql server", "microsoft sql server",
            "db2", "sybase",
            "nosql", "newsql",
        ]
    },
    "devops": {
        "skills": [
            "docker", "dockerfile", "docker compose",
            "kubernetes", "k8s", "helm", "openshift",
            "jenkins", "gitlab ci", "github actions", "githubactions", "circleci", "travis",
            "terraform", "pulumi", "ansible", "chef", "puppet", "vagrant",
            "aws", "gcp", "google cloud", "azure", "ibm cloud", "oracle cloud",
            "nginx", "apache", "haproxy", "traefik",
            "prometheus", "grafana", "datadog", "splunk", "new relic",
            "linux", "unix", "bash", "shell scripting", "powershell",
            "ci/cd", "cicd", "continuous integration", "continuous deployment",
        ]
    },
    "cloud": {
        "skills": [
            "aws", "amazon web services", "ec2", "rds", "s3", "lambda", "cloudfront",
            "azure", "microsoft azure", "azure devops",
            "gcp", "google cloud platform", "google cloud",
            "aws lambda", "aws ec2", "aws s3", "aws rds",
            "azure functions", "azure blob storage", "azure app service",
            "google cloud functions", "gcp cloud run", "gke", "google kubernetes engine",
            "cloudflare", "vercel", "netlify", "heroku", "digitalocean", "linode",
            "faas", "saas", "paas", "iaas",
        ]
    },
}


def normalize_text(text: str) -> str:
    text = text.lower()
    
    text = re.sub(r'[\-_/]', ' ', text)
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def extract_skills_from_text(text: str):
    """Extract skills from raw resume text using fuzzy matching."""
    normalized = normalize_text(text)
    words = normalized.split()

    
    ngrams = set()
    max_len = min(len(words), 3)
    for i in range(len(words)):
        ngrams.add(words[i])
        if i + 1 < len(words):
            ngrams.add(words[i] + " " + words[i + 1])
        if i + 2 < len(words):
            ngrams.add(words[i] + " " + words[i + 1] + " " + words[i + 2])

    found = {cat: set() for cat in SKILL_CATEGORIES}

    for cat, data in SKILL_CATEGORIES.items():
        for skill in data["skills"]:
            skill_norm = normalize_text(skill)
            
            if skill_norm in ngrams:
                found[cat].add(normalize_text(skill))
                continue
            
            
            if len(skill_norm) > 4:
                best = 0.0
                for token in ngrams:
                    if abs(len(token) - len(skill_norm)) <= 2:
                        sim = SequenceMatcher(None, skill_norm, token).ratio()
                        if sim > best:
                            best = sim
                if best >= 0.90:
                    found[cat].add(normalize_text(skill))

    
    return {cat: sorted(list(skills)) for cat, skills in found.items() if skills}


def calculate_category_rating(found_skills: list, category: str) -> int:
    """Calculate a consistent 1-10 rating based on number of found skills."""
    total_in_category = len(SKILL_CATEGORIES[category]["skills"])
    if not total_in_category or not found_skills:
        return 0
    
    count = len(found_skills)
    if count >= 5:
        return 10
    if count == 4:
        return 8
    if count == 3:
        return 6
    if count == 2:
        return 4
    if count == 1:
        return 2
    return 0
