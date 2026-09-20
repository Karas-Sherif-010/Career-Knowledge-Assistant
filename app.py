```python
import os
import pandas as pd
import gradio as gr

from dotenv import load_dotenv

# ============================================================
# ZeroGPU support for Hugging Face Spaces
# ============================================================

try:
    import spaces

    @spaces.GPU(duration=1)
    def zerogpu_startup_probe():
        return None

except ImportError:
    spaces = None


from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please add it to Hugging Face Space Secrets."
    )


# ============================================================
# 2. INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    temperature=0
)


# ============================================================
# 3. INITIALIZE EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. LOAD STATIC PDF FILES
# ============================================================

static_pdf_paths = [
    "data/interview_tips.pdf",
    "data/cv_writing_tips.pdf"
]

pdf_docs = []

print("Loading static PDFs...")

for path in static_pdf_paths:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    loader = PyPDFLoader(path)

    docs = loader.load()

    pdf_docs.extend(docs)


print(f"Loaded {len(pdf_docs)} PDF pages.")


# ============================================================
# 5. SPLIT STATIC PDF DOCUMENTS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

static_chunks = splitter.split_documents(pdf_docs)


# ============================================================
# 6. IDENTIFY STATIC DOCUMENT TYPES
# ============================================================

for doc in static_chunks:

    source = doc.metadata.get(
        "source",
        ""
    ).lower()

    if "cv_writing_tips" in source:

        doc.metadata["source_type"] = "cv_guide"

    elif "interview_tips" in source:

        doc.metadata["source_type"] = "interview_guide"


print(
    f"Created {len(static_chunks)} static PDF chunks."
)


# ============================================================
# 7. LOAD JOB DATA
# ============================================================

jobs_path = "data/jobs.csv"

if not os.path.exists(jobs_path):

    raise FileNotFoundError(
        f"File not found: {jobs_path}"
    )

df = pd.read_csv(jobs_path)

print(
    f"Loaded {len(df)} jobs."
)


# ============================================================
# 8. CONVERT JOBS INTO DOCUMENTS
# ============================================================

job_chunks = []

for i, row in df.iterrows():

    text = (
        f"Job {i + 1}: "
        f"Job Title: {row['Title']}. "
        f"Required Skills: {row['Skills']}. "
        f"Experience: {row['Experience']}. "
        f"Location: {row['Location']}."
    )

    job_chunks.append(

        Document(
            page_content=text,
            metadata={
                "source_type": "jobs",
                "job_number": i + 1
            }
        )
    )


print(
    f"Created {len(job_chunks)} job documents."
)


# ============================================================
# 9. CREATE GENERAL VECTOR STORE
# ============================================================

all_static_chunks = (
    static_chunks + job_chunks
)

general_vectorstore = FAISS.from_documents(
    all_static_chunks,
    embeddings
)

retriever = general_vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 10. CREATE SEPARATE VECTOR STORES
# ============================================================

# CV Guide

cv_guide_docs = [
    doc
    for doc in static_chunks
    if doc.metadata.get("source_type") == "cv_guide"
]

cv_guide_vectorstore = FAISS.from_documents(
    cv_guide_docs,
    embeddings
)


# Interview Guide

interview_docs = [
    doc
    for doc in static_chunks
    if doc.metadata.get("source_type") == "interview_guide"
]

interview_vectorstore = FAISS.from_documents(
    interview_docs,
    embeddings
)


# Jobs

jobs_vectorstore = FAISS.from_documents(
    job_chunks,
    embeddings
)


# ============================================================
# 11. CREATE RETRIEVERS
# ============================================================

cv_guide_retriever = (
    cv_guide_vectorstore
    .as_retriever(
        search_kwargs={"k": 3}
    )
)

interview_retriever = (
    interview_vectorstore
    .as_retriever(
        search_kwargs={"k": 3}
    )
)

jobs_retriever = (
    jobs_vectorstore
    .as_retriever(
        search_kwargs={"k": 5}
    )
)


# ============================================================
# 12. PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",

            """
You are a helpful Career Assistant.

Use ONLY the provided Current Context and Conversation History.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not invent, guess, or assume information.

3. Every factual statement must be supported by
   the Current Context.

4. If information is not available in the Current Context,
   clearly say that it is not available.

5. Never use information from one source as if it came
   from another source.

6. When the question is about the user's CV,
   use only information from the user's CV.

7. Do not use example resumes from the CV writing guide
   as the user's CV.

8. When the user refers to first job, second job,
   third job, etc., use the numbered job information
   in the Current Context.

9. When the user says "it", "that job", or "this job",
   use Conversation History only to understand the reference.

10. If Current Context conflicts with Conversation History,
    prefer Current Context.

11. When comparing the CV with jobs, clearly distinguish:

    - Matching skills
    - Missing skills
    - Information that is not provided

12. Do not provide salary, experience, skills,
    qualifications, or other details unless they are
    explicitly present in the Current Context.

Conversation History:
{chat_history}

Current Context:
{context}
"""
        ),

        (
            "human",
            "{input}"
        )
    ]
)


# ============================================================
# 13. LOAD USER CV
# ============================================================

def load_cv(cv_file):

    if cv_file is None:
        return []

    print("Processing uploaded CV...")

    try:

        cv_loader = PyPDFLoader(cv_file)

        cv_docs = cv_loader.load()

        for doc in cv_docs:

            doc.metadata["source_type"] = "cv"

        cv_chunks = splitter.split_documents(
            cv_docs
        )

        for doc in cv_chunks:

            doc.metadata["source_type"] = "cv"

        print(
            f"Created {len(cv_chunks)} CV chunks."
        )

        return cv_chunks

    except Exception as e:

        print(
            f"Error while processing CV: {e}"
        )

        return []


# ============================================================
# 14. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    cv_file
):

    question_lower = question.lower()

    # --------------------------------------------------------
    # Load CV if uploaded
    # --------------------------------------------------------

    cv_chunks = load_cv(cv_file)

    cv_retriever = None

    if cv_chunks:

        cv_vectorstore = FAISS.from_documents(
            cv_chunks,
            embeddings
        )

        cv_retriever = (
            cv_vectorstore
            .as_retriever(
                search_kwargs={"k": 6}
            )
        )


    # --------------------------------------------------------
    # FIRST / SECOND / THIRD / FOURTH / FIFTH JOB
    # --------------------------------------------------------

    job_numbers = {

        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4

    }

    for word, index in job_numbers.items():

        if f"{word} job" in question_lower:

            job_docs = []

            for i, (_, row) in enumerate(
                df.iterrows()
            ):

                job_docs.append(

                    Document(

                        page_content=(
                            f"Job {i + 1}: "
                            f"Job Title: {row['Title']}. "
                            f"Required Skills: {row['Skills']}. "
                            f"Experience: {row['Experience']}. "
                            f"Location: {row['Location']}."
                        ),

                        metadata={
                            "source_type": "jobs",
                            "job_number": i + 1
                        }
                    )
                )

            return job_docs


    # --------------------------------------------------------
    # JOB + MY CV
    # --------------------------------------------------------

    if (
        ("job" in question_lower or "jobs" in question_lower)
        and
        (
            "my cv" in question_lower
            or
            "my resume" in question_lower
        )
    ):

        docs = []

        if cv_retriever:

            cv_docs = cv_retriever.invoke(
                question
            )

            docs.extend(cv_docs)

        job_docs = jobs_retriever.invoke(
            question
        )

        docs.extend(job_docs)

        return docs


    # --------------------------------------------------------
    # CV QUESTIONS
    # --------------------------------------------------------

    cv_keywords = [

        "my cv",
        "my resume",
        "cv owner",
        "resume owner",
        "name of cv owner",
        "name of resume owner",
        "my name",
        "my email",
        "my phone",
        "phone number",
        "my education",
        "my skills",
        "my experience"

    ]

    if any(
        keyword in question_lower
        for keyword in cv_keywords
    ):

        if cv_retriever:

            return cv_retriever.invoke(
                question
            )

        else:

            return []


    # --------------------------------------------------------
    # JOB QUESTIONS
    # --------------------------------------------------------

    elif (
        "job" in question_lower
        or
        "jobs" in question_lower
    ):

        return jobs_retriever.invoke(
            question
        )


    # --------------------------------------------------------
    # INTERVIEW QUESTIONS
    # --------------------------------------------------------

    elif "interview" in question_lower:

        return interview_retriever.invoke(
            question
        )


    # --------------------------------------------------------
    # CV WRITING QUESTIONS
    # --------------------------------------------------------

    elif (
        "cv tips" in question_lower
        or
        "resume tips" in question_lower
        or
        "cv writing" in question_lower
        or
        "resume writing" in question_lower
    ):

        return cv_guide_retriever.invoke(
            question
        )


    # --------------------------------------------------------
    # GENERAL QUESTIONS
    # --------------------------------------------------------

    else:

        return retriever.invoke(
            question
        )


# ============================================================
# 15. FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    if not docs:

        return "No relevant information was found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 16. FORMAT CHAT HISTORY
# ============================================================

def format_history(
    history,
    max_turns=5
):

    if not history:

        return "No previous conversation."


    recent = history[
        -(max_turns * 2):
    ]

    formatted = []

    for message in recent:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        if role == "user":

            formatted.append(
                f"User: {content}"
            )

        elif role == "assistant":

            formatted.append(
                f"Assistant: {content}"
            )

    if not formatted:

        return "No previous conversation."

    return "\n".join(
        formatted
    )


# ============================================================
# 17. CHAT FUNCTION
# ============================================================

def chat(
    question,
    cv_file,
    history
):

    if history is None:

        history = []


    if not question or not question.strip():

        return history, ""


    print(
        f"\nUser Question: {question}"
    )


    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    docs = retrieve_documents(
        question,
        cv_file
    )


    print(
        f"Retrieved {len(docs)} documents."
    )


    # --------------------------------------------------------
    # Format context
    # --------------------------------------------------------

    context = format_docs(
        docs
    )


    # --------------------------------------------------------
    # Format conversation history
    # --------------------------------------------------------

    formatted_history = format_history(
        history
    )


    # --------------------------------------------------------
    # Run LLM
    # --------------------------------------------------------

    response = llm.invoke(

        prompt.format_messages(

            chat_history=formatted_history,

            context=context,

            input=question

        )
    )


    answer = response.content


    # --------------------------------------------------------
    # Add messages to Gradio history
    # --------------------------------------------------------

    history.append(

        {
            "role": "user",
            "content": question
        }

    )

    history.append(

        {
            "role": "assistant",
            "content": answer
        }

    )


    print(
        "Response generated successfully."
    )


    return history, ""


# ============================================================
# 18. CLEAR CHAT
# ============================================================

def clear_chat():

    return []


# ============================================================
# 19. GRADIO UI
# ============================================================

with gr.Blocks(
    title="Career Knowledge Assistant"
) as demo:

    gr.Markdown(
        """
# 🎓 Career Knowledge Assistant

Upload your CV and ask questions about:

- 📄 Your CV
- 💼 Available jobs
- 🎯 CV writing
- 🗣️ Interview preparation
- 🔎 Job-CV matching

The assistant uses **RAG + FAISS + HuggingFace Embeddings + Groq LLM**
with conversation memory.
"""
    )


    # --------------------------------------------------------
    # CV Upload
    # --------------------------------------------------------

    cv_file = gr.File(
        label="Upload Your CV (PDF)",
        file_types=[".pdf"],
        type="filepath"
    )


    # --------------------------------------------------------
    # Chatbot
    # --------------------------------------------------------

    chatbot = gr.Chatbot(
        label="Career Assistant",
        height=500
    )


    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = gr.Textbox(
        label="Ask a question",
        placeholder=(
            "Example: What skills are mentioned in my CV?"
        ),
        lines=2
    )


    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    with gr.Row():

        send_button = gr.Button(
            "Send",
            variant="primary"
        )

        clear_button = gr.Button(
            "Clear Chat"
        )


    # --------------------------------------------------------
    # Send button
    # --------------------------------------------------------

    send_button.click(

        fn=chat,

        inputs=[
            question,
            cv_file,
            chatbot
        ],

        outputs=[
            chatbot,
            question
        ]

    )


    # --------------------------------------------------------
    # Press Enter
    # --------------------------------------------------------

    question.submit(

        fn=chat,

        inputs=[
            question,
            cv_file,
            chatbot
        ],

        outputs=[
            chatbot,
            question
        ]

    )


    # --------------------------------------------------------
    # Clear button
    # --------------------------------------------------------

    clear_button.click(

        fn=clear_chat,

        inputs=[],

        outputs=[
            chatbot
        ]

    )


# ============================================================
# 20. LAUNCH APP
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Career Knowledge Assistant"
    )

    print(
        "Starting Gradio..."
    )

    print(
        "==========================================\n"
    )

    demo.launch()
```
