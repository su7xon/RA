import re
CATEGORY_KEYWORDS = {
    "frontend": [
        "html", "css", "javascript", "js", "typescript", "ts", "react", "vue",
        "angular", "svelte", "jquery", "bootstrap", "tailwind", "sass", "scss",
        "webpack", "vite", "babel", "redux", "next", "nuxt", "gatsby", "flutter",
        "electron", "pwa", "dom", "ajax", "json", "xml", "mui", "antd", "chakra",
        "styled", "emotion", "three", "d3", "web components", "ui", "ux"
    ],
    "backend": [
        "python", "django", "flask", "fastapi", "tornado", "bottle", "nodejs",
        "node.js", "node", "express", "nestjs", "java", "spring", "kotlin", "php",
        "laravel", "symfony", "ruby", "rails", "go", "golang", "gin", "echo",
        "c#", "csharp", "asp.net", ".net", "dotnet", "c++", "cpp", "rust", "elixir",
        "phoenix", "scala", "play", "akka", "perl", "graphql", "rest", "restful",
        "soap", "grpc", "api"
    ],
    "database": [
        "sql", "mysql", "postgresql", "postgres", "oracle", "sqlite", "mongodb",
        "mongo", "mongoose", "redis", "memcached", "cassandra", "couchdb", "neo4j",
        "dynamodb", "dynamo", "firebase", "firestore", "realm", "elasticsearch",
        "solr", "mariadb", "mssql", "sql server", "db2", "nosql", "newsql", "pl/sql"
    ],
    "devops": [
        "docker", "kubernetes", "k8s", "helm", "jenkins", "gitlab", "github actions",
        "circleci", "travis", "terraform", "pulumi", "ansible", "chef", "puppet",
        "vagrant", "nginx", "apache", "haproxy", "traefik", "prometheus", "grafana",
        "datadog", "splunk", "linux", "unix", "bash", "shell", "powershell", "ci/cd",
        "cicd", "continuous integration", "deployment"
    ],
    "cloud": [
        "aws", "amazon", "azure", "gcp", "google cloud", "ec2", "rds", "s3",
        "lambda", "cloudfront", "cloudflare", "vercel", "netlify", "heroku",
        "digitalocean", "linode", "faas", "saas", "paas", "iaas", "serverless"
    ],
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[\-_/]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def auto_categorize_skills(skills: list) -> dict:
    """
    Takes a list of skill strings and returns dict with keys:
    frontend, backend, database, devops, cloud, other
    """
    categorized = {
        "frontend": [],
        "backend": [],
        "database": [],
        "devops": [],
        "cloud": [],
        "other": [],
    }

    for skill in skills:
        skill_norm = _normalize(skill)
        skill_words = skill_norm.split()
        assigned = False

        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in skill_norm or any(kw in word for word in skill_words):
                    categorized[category].append(skill)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            categorized["other"].append(skill)

    return categorized


if __name__ == "__main__":
    test_skills = ["React", "TensorFlow", "PyTorch", "Kubernetes", "MongoDB", "Figma", "Tableau"]
    print(auto_categorize_skills(test_skills))
