from flask import Flask, render_template, request, jsonify
import sys
import os
import html
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import process_resumes  

app = Flask(__name__)

@app.template_filter('e')
def escape_html_filter(val):
    if val is None:
        return ''
    return html.escape(str(val))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    job_description = request.form.get('job_description', '')
    jd_file = request.files.get('jd_file')

    if jd_file and jd_file.filename != '':
        try:
            if jd_file.filename.lower().endswith('.pdf'):
                from engine import extract_text_from_pdf_stream
                import io
                job_description = extract_text_from_pdf_stream(io.BytesIO(jd_file.read()))
            else:
                job_description = jd_file.read().decode('utf-8', errors='ignore')
        except Exception as e:
            return jsonify({'error': f'Failed to read JD file: {str(e)}'}), 400

    if 'resumes' not in request.files:
        return jsonify({'error': 'No resumes uploaded'}), 400

    uploaded_files = request.files.getlist('resumes')

    if len(uploaded_files) == 0 or uploaded_files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    if len(uploaded_files) > 4:
        return jsonify({'error': 'Maximum 4 resumes can be uploaded at once'}), 400

    if not job_description.strip():
        return jsonify({'error': 'Job description is required'}), 400

    try:
        candidates = process_resumes(job_description, uploaded_files)
        return render_template('results.html', candidates=candidates)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions_route():
    try:
        data = request.get_json() or {}
        raw_skills = data.get('skills', [])

        from engine import groq_clients, GROQ_MODEL, google_client
        import random
        categories = {
            'frontend': {'label': 'Frontend', 'skills': []},
            'backend': {'label': 'Backend', 'skills': []},
            'database': {'label': 'Database', 'skills': []},
            'devops': {'label': 'DevOps', 'skills': []},
            'cloud': {'label': 'Cloud', 'skills': []}
        }
        for item in raw_skills:
            cat = item.lower()
            if 'frontend' in cat or 'react' in cat or 'vue' in cat or 'angular' in cat or 'css' in cat or 'html' in cat:
                categories['frontend']['skills'].append(item)
            elif 'backend' in cat or 'python' in cat or 'java' in cat or 'node' in cat or 'django' in cat or 'express' in cat:
                categories['backend']['skills'].append(item)
            elif 'database' in cat or 'sql' in cat or 'mongo' in cat or 'postgres' in cat or 'redis' in cat:
                categories['database']['skills'].append(item)
            elif 'devops' in cat or 'docker' in cat or 'kubernetes' in cat or 'jenkins' in cat or 'ci' in cat or 'cd' in cat:
                categories['devops']['skills'].append(item)
            elif 'cloud' in cat or 'aws' in cat or 'azure' in cat or 'gcp' in cat or 'lambda' in cat or 'ec2' in cat:
                categories['cloud']['skills'].append(item)

        sample_questions = {
            'frontend': {
                'easy': ['What are the main differences between React and Vue?', 'Explain CSS specificity and how it works.'],
                'medium': ['How does Virtual DOM improve performance in React?', 'Describe the lifecycle methods in Vue 3.'],
                'hard': ['Implement a custom hook that handles debouncing in React.', 'Design a real-time collaborative editor using WebSockets and Vue.'
                ]
            },
            'backend': {
                'easy': ['What is REST and how does it work?', 'Explain the difference between SQL and NoSQL.'],
                'medium': ['How does database indexing improve query performance?', 'Describe the CAP theorem and its implications.'],
                'hard': ['Design a distributed rate limiter for a high-traffic API.', 'Implement a custom ORM with lazy loading and caching.']
            },
            'database': {
                'easy': ['What is ACID and why is it important?', 'Explain the difference between JOIN and UNION.'],
                'medium': ['How do you optimize a slow SQL query?', 'Describe database sharding strategies.'],
                'hard': ['Design a database schema for a scalable e-commerce platform.', 'Implement a custom database indexing algorithm.']
            },
            'devops': {
                'easy': ['What is Docker and how does it work?', 'Explain the difference between CI and CD.'],
                'medium': ['How does Kubernetes handle pod scaling?', 'Describe blue-green deployment strategy.'],
                'hard': ['Design a zero-downtime deployment pipeline.', 'Implement a custom Kubernetes operator.']
            },
            'cloud': {
                'easy': ['What is the difference between IaaS, PaaS, and SaaS?', 'Explain how load balancing works in AWS.'],
                'medium': ['How do you design a multi-region architecture?', 'Describe cost optimization strategies in cloud.'],
                'hard': ['Design a serverless architecture for a million-user application.', 'Implement a custom cloud cost optimization engine.']
            }
        }

        if groq_clients:
            import json as _json
            prompt = (
                "Generate 5 easy, 5 medium, 5 hard interview questions for each category based on these skills: "
                + str(raw_skills) +
                "\\nReturn JSON with keys: frontend, backend, database, devops, cloud. Each key has easy, medium, hard arrays of strings."
            )
            for i, client in enumerate(groq_clients):
                try:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=GROQ_MODEL,
                        response_format={"type": "json_object"},
                        temperature=0.7,
                        max_tokens=1024
                    )
                    ai_questions = _json.loads(response.choices[0].message.content)
                    return jsonify({'questions': ai_questions})
                except Exception as e:
                    print(f"AI question generation error with Groq key {i+1}: {e}")

        if google_client:
            try:
                import json as _json
                prompt = (
                    "Generate 5 easy, 5 medium, 5 hard interview questions for each category based on these skills: "
                    + str(raw_skills) +
                    "\\nReturn JSON with keys: frontend, backend, database, devops, cloud. Each key has easy, medium, hard arrays of strings."
                )
                response = google_client.generate_content(prompt)
                text = response.text
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0]
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0]
                ai_questions = _json.loads(text.strip())
                return jsonify({'questions': ai_questions})
            except Exception as e:
                print(f"AI question generation error with Google Gemini: {e}")

        return jsonify({'questions': sample_questions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
