from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
import yfinance as yf
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 1rem;
        }
        .subheader {
            color: #64748B;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 0.75rem;
            background-color: #FFFFFF;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            margin-bottom: 1rem;
        }
        .metric-card {
            text-align: center;
            padding: 1rem;
            background-color: #F8FAFC;
            border-radius: 0.5rem;
            border: 1px solid #E2E8F0;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #0EA5E9;
        }
        .metric-label {
            font-size: 0.875rem;
            color: #64748B;
        }
        .stButton > button {
            width: 100%;
            background-color: #0EA5E9;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: #0284C7;
        }
        .stTextInput > div > div > input {
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize clients
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "stocks"
namespace = "stock-descriptions"
pinecone_index = pc.Index(index_name)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Initialize embeddings
hf_embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# [Keep your existing helper functions here]

# Main UI
st.markdown("<h1 class='main-header'>📈 Stock Market Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Get insights about stocks using advanced AI analysis</p>", unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Stock Analysis", "Market Overview"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container():
            st.markdown("### 🔍 Query Parameters")
            query = st.text_area(
                "Ask about stocks:",
                placeholder="E.g., 'What are the top performing tech stocks?' or 'Show me stocks with high dividend yields'",
                height=100
            )
    
    with col2:
        with st.container():
            st.markdown("### 🎯 Filters")
            industry = st.selectbox(
                'Industry',
                options=[''] + ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods', 'Other']
            )
            sector = st.selectbox(
                'Sector',
                options=[''] + ['Information Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer Discretionary', 'Other']
            )
            
            col3, col4 = st.columns(2)
            with col3:
                market_cap = st.number_input(
                    'Min Market Cap (M)',
                    min_value=0,
                    max_value=1000000,
                    value=0,
                    step=1000
                )
            with col4:
                volume = st.number_input(
                    'Min Volume',
                    min_value=0,
                    max_value=1000000,
                    value=0,
                    step=10000
                )

    # Create filter dictionary
    filter = {
        "industry": industry if industry else None,
        "sector": sector if sector else None,
        "marketCap": {"$gte": market_cap} if market_cap > 0 else None,
        "volume": {"$gte": volume} if volume > 0 else None
    }
    
    # Remove None values
    filter = {k: v for k, v in filter.items() if v is not None}

    if st.button('Analyze Stocks', key='analyze'):
        with st.spinner('Analyzing stocks...'):
            response = HandleQuery(query, filter)
            
            st.markdown("### 📊 Analysis Results")
            st.markdown(response)

with tab2:
    st.markdown("### 📈 Market Overview")
    
    # Add market overview metrics
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">$34.5T</div>
                <div class="metric-label">Total Market Cap</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">3.2%</div>
                <div class="metric-label">Average Dividend Yield</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">24.5</div>
                <div class="metric-label">Average P/E Ratio</div>
            </div>
        """, unsafe_allow_html=True)

    # Add a sample market trend chart
    chart_data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=30),
        'Value': [100 + i + np.random.randn() * 5 for i in range(30)]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['Date'],
        y=chart_data['Value'],
        mode='lines',
        fill='tonexty',
        line=dict(color='#0EA5E9')
    ))
    
    fig.update_layout(
        title="Market Trend (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Market Value",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.875rem;'>
        Data provided by various financial sources. Use at your own discretion.
    </div>
""", unsafe_allow_html=True)
