# AI Support Copilot

A local AI application that analyzes fictional support tickets and generates grounded support response plans.

The application uses semantic search to find relevant guidance in a knowledge base, then uses a local LLM to create a structured support plan.

## Features

- Accepts a support ticket title and description
- Searches a local knowledge base with semantic similarity
- Identifies relevant support guidance and similarity scores
- Generates a structured support plan with:
  - Category
  - Priority
  - Recommended next steps
  - Customer response
  - Sources used
- Provides a browser interface built with Streamlit
- Uses a local Ollama model (`gemma3:4b`)
- Uses fictional support documentation only

## Example

```text
Ticket title: API requests are failing

Ticket description:
Customers receive a 500 response from the Orders API.

Category:
API Errors

Priority:
High

Recommended next steps:
1. Collect the endpoint URL, timestamp, request ID, response code, and affected customer details.
2. Check service health and recent deployments.
3. Escalate if the issue continues.

Customer response:
Thank you for reporting this issue. Our support team is investigating the API errors and will keep you updated.

Sources:
api_errors.txt
```

## Project Structure

```text
ai-support-copilot/
├── knowledge_base/
│   ├── api_errors.txt
│   ├── authentication_incidents.txt
│   └── deployment_incidents.txt
├── app.py
├── copilot.py
└── requirements.txt
```

## Run Locally

### 1. Create and install the Python environment

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Start Ollama with Docker

Docker Desktop must be running.

```powershell
docker start ollama
```

If the container does not exist yet:

```powershell
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 3. Download the local model

```powershell
docker exec -it ollama ollama pull gemma3:4b
```

### 4. Run the web application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- Ollama
- Gemma 3 4B
- Docker
- Git and GitHub