# HR Resource Query Chatbot

## Overview
A full-stack HR assistant that helps you find, recommend, and manage employees based on skills, experience, and project history. The system matches both uploaded resumes and existing employees to job descriptions using semantic search and skill extraction. Built with FastAPI (Python) and a responsive HTML/JS frontend.

## Features
- Natural language HR queries (chatbot)
- AI-powered employee recommendations
- Upload and match resumes to job descriptions (JD)
- Search/filter employees by skills, experience, department, and availability
- Add, edit, and delete employees (with persistent storage)
- View employee analytics (top skills, departments, experience)
- Responsive, user-friendly frontend with modals for management and matching
- Resume info extraction (name, email, contact, skills)
- Matches both uploaded resumes and existing employees

## Architecture
- **Backend:** FastAPI (Python), in-memory and JSON file storage, scikit-learn for TF-IDF/cosine similarity, basic NLP for info extraction.
- **Frontend:** Single-page HTML/JS/CSS, responsive design, modals for management and matching.
- **Matching:** Both uploaded resumes and existing employees are matched to the JD using semantic similarity and skill overlap. Info is extracted from resumes using regex and heuristics.
- **Persistence:** Employees are stored in a JSON file for easy editing and retrieval.
- **Extensibility:** Designed to allow future integration of advanced LLMs or cloud APIs for improved matching and extraction.

## Setup and Installation

1. **Clone the repository**
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Key packages: fastapi, uvicorn, scikit-learn, numpy, python-multipart, PyPDF2, python-docx)
3. Start the backend:
   ```bash
   python main.py
   ```
4. Open `index.html` in your browser for the frontend UI.

## API Documentation

### Endpoints

- `POST /chat`
  - Description: Process natural language HR queries.
  - Request: JSON object with `query` string.
  - Response: JSON with chatbot response, matched employees, and confidence score.

- `POST /recommendations`
  - Description: Get AI-powered top employee recommendations based on query.
  - Request: JSON object with `query` string.
  - Response: JSON with list of recommended employees and count.

- `GET /employees`
  - Description: Retrieve all employees.
  - Response: JSON with list of all employees.

- `GET /employees/search`
  - Description: Search employees with filters.
  - Query Parameters: `skills` (comma-separated), `min_experience` (int), `availability` (string), `department` (string).
  - Response: JSON with filtered employees and count.

- `GET /employees/{employee_id}`
  - Description: Get details of a specific employee by ID.
  - Response: JSON with employee details.

- `POST /employees`
  - Description: Add a new employee.
  - Request: JSON with employee data.
  - Response: Success message and added employee or error.

- `PUT /employees/{employee_id}`
  - Description: Update an existing employee.
  - Request: JSON with updated employee data.
  - Response: Success message and updated employee or error.

- `DELETE /employees/{employee_id}`
  - Description: Delete an employee by ID.
  - Response: Success message or error.

- `GET /stats`
  - Description: Get statistics about employees (availability, departments, skills, experience).
  - Response: JSON with stats data.

- `POST /match-jd-resumes`
  - Description: Match uploaded resumes to a job description.
  - Request: Multipart form with `jd_text` or `jd_file`, and multiple `resumes`.
  - Response: JSON with matched resumes and employees.


## AI Development Process

- **Code Generation:**
  - AI generated most backend endpoints, data models, and matching logic.
  - AI wrote the initial frontend (HTML/CSS/JS) and responsive design.
- **Debugging:**
  - AI helped trace 404 errors, missing dependencies, and improved error logging.
  - AI suggested fixes for file upload, CORS, and form handling issues.
- **Architecture Decisions:**
  - AI recommended FastAPI for rapid prototyping and scikit-learn for simple semantic search.
  - AI suggested a hybrid approach: match both resumes and employees for best results.
- **AI Assistance:**
  - ~60% of code was AI-generated, especially for boilerplate, endpoints, and UI.
  - Manual work: fine-tuning info extraction, frontend polish, and some bug fixes.
- **Interesting AI Solutions:**
  - AI-generated regex for extracting emails/contacts from resumes.
  - Responsive CSS and modal logic were mostly AI-suggested.
- **Manual Challenges:**
  - Handling edge cases in resume parsing (PDF/DOCX errors) required manual debugging.
  - Some UI/UX tweaks and error messages were hand-written for clarity.

## Technical Decisions

- **OpenAI vs Open-Source Models:**
  - Used only open-source (scikit-learn TF-IDF) for semantic search to keep everything local and free.
  - No OpenAI/cloud LLMs used for privacy and cost reasons.
- **Local LLM vs Cloud:**
  - Local approach (no LLM) is fast, private, and has no API cost, but less accurate than modern LLMs.
  - For production, consider hybrid: local for speed, cloud for advanced NLP.
- **Performance/Cost/Privacy:**
  - Chose scikit-learn for zero cost, instant response, and no data leaves your machine.
  - Trade-off: less nuanced understanding than LLMs, but good enough for skill/keyword matching.

## Future Improvements

- Integrate advanced LLMs (OpenAI, Ollama, or HuggingFace) for better semantic matching and info extraction.
- Add authentication and user management.
- Improve resume parsing (handle more formats, extract more fields).
- Add analytics dashboards and export features.
- Deploy as a Dockerized web app for easy installation.
- Add automated tests and CI/CD.

## Demo

Try the live demo of the frontend hosted on GitHub Pages:

[https://ombarde.github.io/HR-Resource-Assistant/](https://ombarde.github.io/HR-Resource-Assistant/)

The backend API is hosted on Render at:

[https://hr-resource-assistant.onrender.com](https://hr-resource-assistant.onrender.com)
