import json
import requests
import numpy as np
import yfinance as yf
import streamlit as st

from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from pinecone import Pinecone

# Use Streamlit secrets for sensitive information
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]
yahoo_access_token = st.secrets.get("YAHOO_ACCESS_TOKEN", "")

index_name = "stocks"
namespace = "stock-descriptions"

# Initialize embeddings and vector store
hf_embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index(index_name)

# Initialize LLM client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

def get_huggingface_embeddings(text: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> np.ndarray:
    """Compute HuggingFace embeddings for a given text input."""
    model = SentenceTransformer(model_name)
    return model.encode(text)

def get_stock_info_all(symbol: str) -> dict:
    """
    Retrieve stock information for a given symbol using Yahoo Finance.
    Requires a Yahoo access token if protected requests are needed.
    """
    headers = {}
    if yahoo_access_token:
        headers['Authorization'] = f'Bearer {yahoo_access_token}'

    session = requests.Session()
    session.headers.update(headers)
    try:
        data = yf.Ticker(symbol, session=session)
        return data.info or {}
    except Exception as e:
        print(f"Error fetching stock info for {symbol}: {e}")
        return {}

def format_filter_conditions(filter_conditions: dict) -> str:
    """Format filter conditions into a human-readable string."""
    if not filter_conditions:
        return ""

    formatted_filters = []
    for key, value in filter_conditions.items():
        if isinstance(value, dict):
            for op, val in value.items():
                operator_map = {
                    "$gte": "greater than or equal to",
                    "$lte": "less than or equal to",
                    "$gt": "greater than",
                    "$lt": "less than",
                    "$eq": "equals",
                    "$in": "in",
                }
                op_text = operator_map.get(op, op)
                formatted_filters.append(f"{key} is {op_text} {val}")
        else:
            if value:
                formatted_filters.append(f"{key}: {value}")

    return ", ".join(formatted_filters)

def handle_query(query: str, filter_conditions: dict) -> str:
    """Process the user's query and filters, retrieve relevant info, and get an LLM response."""
    filter_str = format_filter_conditions(filter_conditions)
    query_embedding = get_huggingface_embeddings(query)
    
    top_matches = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace=namespace,
        filter=filter_conditions if filter_conditions else None
    )

    contexts = [item['metadata']['text'] for item in top_matches.get('matches', [])]
    context_str = "\n\n-------\n\n".join(contexts[:10])

    augmented_query = (
        f"<CONTEXT>\n{context_str}\n-------\n</CONTEXT>\n\n"
        f"MY QUESTION:\n{query}\n{filter_str}"
    )

    system_prompt = (
        "You are an expert at providing answers about stocks. Please answer my question.\n\n"
        "When giving your response, do not mention the context or the query directly.\n"
        "Provide a detailed answer in markdown format, listing all relevant stocks and information.\n"
        "Order the answers from most relevant to least relevant.\n"
        "If no direct question is asked, list all the stocks that match the filters and their details.\n"
    )

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content.strip()
    return response

# Streamlit UI Configuration
st.set_page_config(page_title="AI-Powered Stock Analysis", layout="wide")

# Sidebar for Filters and Instructions
st.sidebar.title("Stock Analysis Filters")
st.sidebar.write("**Use the filters below to narrow down results.**")

industry = st.sidebar.text_input('Industry:', help="e.g., Technology, Healthcare, Finance")
sector = st.sidebar.text_input('Sector:', help="e.g., Consumer Defensive, Energy")
market_cap = st.sidebar.number_input('Minimum Market Cap:', min_value=0, max_value=10**9, step=1000000, help="Enter a number. Stocks with market cap greater than or equal to this value will be shown.")
volume = st.sidebar.number_input('Minimum Volume:', min_value=0, max_value=10**9, step=100000, help="Enter a number. Stocks with volume greater than or equal to this value will be shown.")

st.sidebar.write("---")
st.sidebar.write("**How to Use:**")
st.sidebar.markdown("- Enter a natural language query about stocks in the main page.\n- Apply filters in the sidebar if desired.\n- Click **Get Stock Info** to view results.")

# Main Page
st.title("AI-Powered Stock Analysis")
st.write("Use natural language queries to find stocks that match certain criteria and get detailed AI-generated insights. For best results, be specific in your query and utilize the filters.")

query = st.text_input('Ask about stocks:', placeholder="e.g., 'Show me top performing tech companies.'")

filters = {
    "industry": industry if industry else None,
    "sector": sector if sector else None,
    "marketCap": {"$gte": market_cap} if market_cap > 0 else None,
    "volume": {"$gte": volume} if volume > 0 else None
}
filters = {k: v for k, v in filters.items() if v is not None}

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Get Stock Info"):
        if not query and not filters:
            st.error("Please provide a query or at least one filter.")
        else:
            with st.spinner("Analyzing stocks..."):
                response = handle_query(query, filters)

            st.write("### Response:")
            if response:
                st.markdown(response)
            else:
                st.error("No information found for this query.")

with col2:
    st.info("**Tips:**\n- Try asking about companies in a specific industry or sector.\n- Combine filters and queries for more targeted results.\n- Example query: 'Which companies have a market cap over 1B in the healthcare sector?'")
