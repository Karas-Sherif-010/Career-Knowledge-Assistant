# Career Knowledge Assistant

An AI-powered career assistant that uses **Retrieval-Augmented Generation (RAG)** to help users understand their CV, explore career opportunities, compare their skills with job requirements, and prepare for CV writing and interviews.

The project was developed as a **Midterm Project** during my AI Agents training with **Digital HUB (D-HUB)** and **Orange Digital Center Egypt**.

---

## 1. Project Overview

Finding suitable career opportunities can be challenging, especially for students and early-career professionals who may not know which job roles match their current skills.

The **Career Knowledge Assistant** was developed to provide a conversational interface where users can upload their CV and ask natural-language questions about their career profile.

The assistant combines:

* User CV information
* Career-related PDF documents
* Job information
* Vector-based semantic search
* Large Language Models
* Conversation memory

This allows users to interact with their career information through natural-language conversations instead of manually searching through documents and job descriptions.

---

## 2. Project Objectives

The main objectives of the project are:

* Build a practical RAG-based AI application.
* Allow users to upload and query their CV.
* Retrieve relevant career information from documents.
* Search and retrieve relevant job opportunities.
* Compare user skills with job requirements.
* Provide CV writing guidance.
* Provide interview preparation tips.
* Maintain context across multiple questions.
* Reduce hallucination by restricting responses to retrieved information.
* Deploy the application as a public web application.

---

## 3. Main Features

### 📄 CV Analysis

Users can upload their CV in PDF format.

The assistant can answer questions about information explicitly available in the CV, such as:

* Name
* Contact information
* Education
* Skills
* Experience
* Projects
* Certifications

The system is instructed not to invent information that is not available in the uploaded CV.

---

### 💼 Job Search

The assistant can retrieve available job opportunities based on the user's questions.

Users can ask questions such as:

> What jobs are available?

or:

> What skills are required for the Data Analyst job?

The job information is converted into searchable documents and indexed using FAISS.

---

### 🎯 CV–Job Matching

One of the main features of the project is matching the user's CV with available job opportunities.

The system can identify:

* Matching skills
* Missing skills
* Relevant experience
* Job requirements explicitly available in the dataset

For example:

> Which jobs match my CV?

The assistant retrieves relevant jobs and uses the LLM to explain the relationship between the CV and the job requirements.

---

### 📝 CV Writing Guidance

The system includes a CV writing guide that users can query.

Example:

> What are some important CV writing tips?

The assistant retrieves relevant information from the CV guide before generating the answer.

---

### 🗣️ Interview Preparation

The project also includes an interview preparation guide.

Users can ask questions such as:

> What are some important interview tips?

The assistant retrieves relevant information from the interview guide.

---

### 💬 Conversation Memory

The assistant supports multi-turn conversations.

For example:

**User:**

> Which job matches my CV?

**Assistant:**

> The Machine Learning Intern role has several matching skills...

**User:**

> What about the second job?

**Assistant:**

> The second job is the Machine Learning Intern role...

**User:**

> Does it require SQL?

The assistant uses the previous conversation to understand what **"it"** refers to.

This provides a more natural conversational experience.

---

### 🛡️ Anti-Hallucination

The system prompt contains strict instructions that require the assistant to:

* Use only the provided context.
* Avoid outside knowledge.
* Avoid guessing.
* Avoid inventing information.
* Clearly state when information is unavailable.
* Distinguish between CV information and career-guide information.
* Distinguish matching skills from missing skills.

For example, if a job's salary is not available in the database, the assistant should not invent a salary value.

---

# 4. System Architecture

The overall architecture can be summarized as:

```text
                    ┌──────────────────┐
                    │      User        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Upload CV /     │
                    │  Ask Question    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Query Processing │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       Document Retrieval     │
              │                              │
              │ • CV                         │
              │ • Jobs                       │
              │ • CV Guide                   │
              │ • Interview Guide            │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      FAISS       │
                    │ Semantic Search  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Retrieved Context│
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
          Conversation History       Context
                  │                     │
                  └──────────┬──────────┘
                             ▼
                    ┌──────────────────┐
                    │      Groq LLM    │
                    │ GPT-OSS-120B     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Response     │
                    └──────────────────┘
```

---

# 5. RAG Pipeline

The project follows a **Retrieval-Augmented Generation (RAG)** architecture.

The pipeline consists of several stages.

## Step 1 — Document Loading

The system loads PDF documents using `PyPDFLoader`.

The static knowledge base contains:

* `interview_tips.pdf`
* `cv_writing_tips.pdf`

The user CV is uploaded dynamically as a PDF.

---

## Step 2 — Document Splitting

Large documents are split into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

The project uses:

```text
chunk_size = 500
chunk_overlap = 50
```

Chunking allows the retrieval system to search smaller and more relevant sections of the documents.

---

## Step 3 — Embeddings

The text chunks are converted into vector representations using:

**Hugging Face Sentence Transformers**

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings allow the system to represent text based on semantic meaning rather than simple keyword matching.

---

## Step 4 — Vector Database

The generated embeddings are stored in:

**FAISS**

FAISS provides efficient similarity search over the generated vectors.

Separate vector stores/retrievers are used for different information sources, including:

* CV
* Jobs
* CV writing guide
* Interview guide

This allows the application to route questions to the most relevant source.

---

## Step 5 — Retrieval

When the user asks a question, the system determines which information source is most relevant.

Examples:

```text
"My CV"
      ↓
CV Retriever

"Jobs"
      ↓
Jobs Retriever

"CV writing"
      ↓
CV Guide Retriever

"Interview"
      ↓
Interview Retriever
```

For general questions, the general retriever can be used.

---

## Step 6 — Context Construction

The retrieved documents are converted into a context string.

The context is then provided to the LLM together with the user's question and recent conversation history.

---

## Step 7 — Generation

The retrieved context is passed to the Groq-hosted LLM:

```text
openai/gpt-oss-120b
```

The model generates the final response while following the system instructions.

---

# 6. Conversation Memory

The application maintains conversation history so that follow-up questions can be understood.

The history is stored as user/assistant message pairs.

Example:

```text
User:
Which job matches my CV?

Assistant:
The Data Analyst role matches several skills...

User:
What about the second job?

Assistant:
The second job is the Machine Learning Intern role...

User:
Does it require SQL?
```

The history allows the assistant to resolve references such as:

* it
* this job
* that job
* the second job
* the previous role

The system uses conversation history to understand references but relies on the current retrieved context for factual answers.

---

# 7. Job Representation

Job information is converted into searchable documents.

Each job contains information such as:

```text
Job Title
Required Skills
Experience
Location
```

For example:

```text
Job Title: Machine Learning Intern

Required Skills:
Python, Scikit-learn, Pandas, Machine Learning

Experience:
0-1 years

Location:
Cairo
```

These job documents are embedded and indexed in FAISS.

---

# 8. Intelligent Query Routing

The application uses simple query routing to determine which retriever should be used.

Examples:

| User Question                       | Retriever            |
| ----------------------------------- | -------------------- |
| What skills are in my CV?           | CV Retriever         |
| What jobs are available?            | Jobs Retriever       |
| What are important CV tips?         | CV Guide Retriever   |
| How can I prepare for an interview? | Interview Retriever  |
| Which job matches my CV?            | CV + Jobs            |
| Does the second job require SQL?    | Job context + Memory |

This helps prevent irrelevant information from being retrieved.

---

# 9. Prompt Design

The system prompt contains several grounding rules.

Important rules include:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent or guess information.
4. Clearly state when information is unavailable.
5. Use only the CV when answering CV-specific questions.
6. Do not confuse example CVs with the user's CV.
7. Use numbered job information when the user refers to a specific job.
8. Use conversation history to resolve references.
9. Prefer current retrieved context over old conversation context.
10. Clearly distinguish matching skills, missing skills, and unavailable information.

These rules help improve the reliability of the assistant.

---

# 10. Technologies Used

### Programming

**Python**

The main programming language used to build the application.

### LLM Framework

**LangChain**

Used for:

* Prompt management
* Document processing
* Retrieval pipeline
* LLM integration

### Embeddings

**Hugging Face Sentence Transformers**

Model:

```text
all-MiniLM-L6-v2
```

Used to create vector representations of documents.

### Vector Search

**FAISS**

Used for semantic similarity search.

### LLM

**Groq**

Model:

```text
openai/gpt-oss-120b
```

Used to generate responses based on retrieved context.

### PDF Processing

**PyPDF / PyPDFLoader**

Used to extract text from PDF documents.

### Data Processing

**Pandas**

Used to load and process structured job information.

### Interface

**Gradio**

Used to build the interactive web interface.

### Deployment

**Hugging Face Spaces**

Used to deploy the application as a public web application.

---

# 11. Testing

The application was tested using different scenarios to verify its functionality.

### Test 1 — CV Retrieval

Question:

> What skills are mentioned in my CV?

Expected behavior:

The assistant retrieves information from the uploaded CV.

---

### Test 2 — Job Matching

Question:

> Which job from the list fits me based on my CV?

Expected behavior:

The assistant retrieves CV and job information and identifies matching and missing skills.

---

### Test 3 — Conversation Memory

Question:

> What about the second job on the list?

Expected behavior:

The assistant identifies the second job using the conversation context and job numbering.

---

### Test 4 — Follow-up Question

Question:

> Does it require SQL?

Expected behavior:

The assistant understands which job "it" refers to and answers using the relevant job information.

---

### Test 5 — CV Writing

Question:

> What are some important CV writing tips?

Expected behavior:

The assistant retrieves information from the CV writing guide.

---

### Test 6 — Interview Preparation

Question:

> What are some important interview tips?

Expected behavior:

The assistant retrieves information from the interview guide.

---

### Test 7 — Missing Information

Question:

> What is the salary of the AI Engineer job?

Expected behavior:

The assistant should state that salary information is not available instead of inventing a value.

---

# 12. Deployment

The application was deployed using:

**Hugging Face Spaces + Gradio**

The application provides a public interface where users can:

1. Upload their CV.
2. Ask career-related questions.
3. Explore available jobs.
4. Compare their skills with job requirements.
5. Ask follow-up questions.

---

# 13. Running the Project Locally

## Clone the Repository

```bash
git clone https://github.com/Karas-Sherif-010/Career-Knowledge-Assistant.git

cd Career-Knowledge-Assistant
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure API Key

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit the `.env` file to GitHub.

---

## Run the Application

```bash
python app.py
```

The Gradio interface will then be available locally.

---

# 14. Project Structure

```text
Career-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── Career_Assistant.ipynb
├── .gitignore
│
└── data/
    ├── interview_tips.pdf
    ├── cv_writing_tips.pdf
    └── jobs.csv
```

---

# 15. Future Improvements

Several improvements can be added in future versions.

### 🔹 Larger Job Database

Replace the initial job dataset with thousands of real job postings.

This would allow the assistant to search a much larger career database.

---

### 🔹 Advanced CV–Job Matching

Develop a more advanced matching system that considers:

* Skills
* Experience
* Education
* Job title
* Location
* Work mode
* Seniority

---

### 🔹 Job Ranking

Return the most relevant jobs based on semantic similarity and explain why each job was retrieved.

---

### 🔹 OCR Support

Add OCR for CVs where important information is stored inside scanned images rather than selectable text.

---

### 🔹 Improved Memory

Store structured information about the last retrieved jobs so that references such as:

> the second job

can reliably refer to the exact jobs previously displayed.

---

### 🔹 Live Job Data

Connect the system to a job API or continuously updated job database so that users can search more recent opportunities.

---

# 16. Project Links

### GitHub Repository

https://github.com/Karas-Sherif-010/Career-Knowledge-Assistant

### Live Application

https://huggingface.co/spaces/karas-sherif/Career-Knowledge-Assistant

---

# 17. Conclusion

The **Career Knowledge Assistant** demonstrates how multiple AI concepts can be combined into one practical end-to-end application.

Instead of using an LLM alone, the project combines:

```text
Documents
    +
Embeddings
    +
FAISS
    +
RAG
    +
LLM
    +
Conversation Memory
    =
Career Knowledge Assistant
```

The project provides a foundation for a more advanced career platform that can help students, graduates, and professionals understand their skills, explore relevant opportunities, identify skill gaps, and prepare for their next career step.

---

## Author

**Karas Sherif**

Faculty of Science — Helwan University
Statistics & Computer Science

Interested in:

**Data Analysis • Machine Learning • Artificial Intelligence • LLMs • AI Agents**
