import os
import openai
from dotenv import load_dotenv
from embedder import get_embedding
from faiss_index import query_index

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def explain_anomaly(anomaly_description):
    """Retrieve context from FAISS and generate anomaly explanation."""
    embedding = get_embedding(anomaly_description)

    # Get similar guidelines/docs from FAISS
    context_snippets = query_index(embedding, top_k=3)

    # Construct prompt for OpenAI LLM
    prompt = f"""
    You're an anomaly detection assistant. Explain the following anomaly:

    Anomaly Details: {anomaly_description}

    Relevant Guidelines:
    {context_snippets}

    Provide a concise explanation:
    """

    # Call OpenAI GPT
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
