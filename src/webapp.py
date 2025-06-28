#!/usr/bin/env python3
"""
Streamlit web application for the F1 expert model.

This creates an interactive chat interface for asking F1 questions,
with additional features like live standings and race information.

Usage:
    streamlit run webapp.py

Features:
    - Chat interface for F1 questions
    - Live F1 standings from Jolpica API
    - Question examples and suggestions
    - Model information and credits
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# Page configuration
st.set_page_config(
    page_title="F1 Expert Assistant",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_f1_model(model_path: str = "models/gemma-f1-lora", base_model: str = "google/gemma-3n"):
    """Load the F1 expert model (cached for performance)."""
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        st.info("Please train the model first: `python src/train_lora.py`")
        return None, None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_4bit=True if torch.cuda.is_available() else False
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model_obj, model_path)
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_standings():
    """Fetch current F1 standings from Jolpica API."""
    try:
        # Current season driver standings
        url = "https://api.jolpi.ca/ergast/f1/current/driverStandings.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        standings_list = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        
        if standings_list:
            return standings_list[0].get("DriverStandings", [])
        return []
        
    except Exception as e:
        st.error(f"Error fetching standings: {e}")
        return []


def generate_f1_answer(model, tokenizer, question: str) -> str:
    """Generate answer using the F1 expert model."""
    if model is None or tokenizer is None:
        return "Model not available. Please check the model path."
    
    # Create prompt
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Formula 1 assistant. Answer the following question accurately and concisely.

{question}

### Response:
"""
    
    try:
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        response_marker = "### Response:"
        if response_marker in response:
            answer = response.split(response_marker, 1)[1].strip()
        else:
            answer = response.strip()
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {e}"


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🏎️ F1 Expert Assistant")
    st.markdown("*Ask me anything about Formula 1!*")
    
    # Sidebar
    with st.sidebar:
        st.header("🏁 Navigation")
        
        # Model status
        st.subheader("🤖 Model Status")
        model, tokenizer = load_f1_model()
        
        if model is not None:
            st.success("✅ Model loaded")
            if torch.cuda.is_available():
                st.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("💻 Running on CPU")
        else:
            st.error("❌ Model not loaded")
        
        st.divider()
        
        # Current standings
        st.subheader("🏆 Current Standings")
        if st.button("🔄 Refresh Standings", type="secondary"):
            st.cache_data.clear()
        
        standings = get_current_standings()
        if standings:
            st.markdown("**Top 5 Drivers:**")
            for i, driver in enumerate(standings[:5]):
                driver_name = f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}"
                points = driver['points']
                constructor = driver['Constructors'][0]['name']
                st.markdown(f"{i+1}. **{driver_name}** - {points} pts ({constructor})")
        else:
            st.info("Standings unavailable")
        
        st.divider()
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "Who won the 2023 Monaco Grand Prix?",
            "How does DRS work in Formula 1?",
            "What is the current F1 points system?",
            "Explain F1 qualifying format",
            "Who holds the most F1 championships?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                st.session_state.example_question = question
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Handle example question from sidebar
        default_question = ""
        if "example_question" in st.session_state:
            default_question = st.session_state.example_question
            del st.session_state.example_question
        
        # Question input
        question = st.text_area(
            "Your F1 question:",
            value=default_question,
            height=100,
            placeholder="e.g., Who won the last Monaco Grand Prix?"
        )
        
        # Generate button
        col_btn1, col_btn2 = st.columns([1, 3])
        
        with col_btn1:
            generate_clicked = st.button("🚀 Ask Expert", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
        
        # Generate answer
        if generate_clicked and question.strip():
            if model is not None:
                with st.spinner("🤔 Thinking..."):
                    start_time = time.time()
                    answer = generate_f1_answer(model, tokenizer, question)
                    response_time = time.time() - start_time
                
                st.subheader("🤖 F1 Expert Response:")
                st.markdown(answer)
                st.caption(f"*Response generated in {response_time:.2f} seconds*")
                
                # Save to chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now()
                })
                
            else:
                st.error("Model not available. Please check the setup.")
        
        elif generate_clicked:
            st.warning("Please enter a question first.")
    
    with col2:
        st.subheader("📋 Information")
        
        # Quick stats
        st.markdown("**Current Season:** 2024")
        st.markdown("**Next Race:** *Check F1 calendar*")
        st.markdown("**Model:** Gemma-3 + LoRA")
        
        # Model info
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            This F1 expert assistant is powered by:
            - **Base Model:** Google Gemma-3
            - **Fine-tuning:** LoRA (Low-Rank Adaptation)
            - **Training Data:** ~3,000 F1 Q-A pairs
            - **Data Sources:** Jolpica-F1 API, official press releases
            
            The model specializes in:
            - Race results and statistics
            - Technical explanations (DRS, tyres, etc.)
            - Rules and regulations
            - Historical F1 data
            """)
        
        # Performance tips
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - Be specific in your questions
            - Ask about races from 2000 onwards
            - Use proper F1 terminology
            - Questions about current season may need recent data
            """)
    
    # Chat history
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.divider()
        st.subheader("💭 Recent Questions")
        
        # Show last 3 conversations
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
            with st.expander(f"Q: {chat['question'][:50]}..." if len(chat['question']) > 50 else f"Q: {chat['question']}"):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"Asked at {chat['timestamp'].strftime('%H:%M:%S')}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🏎️ F1 Expert Assistant | Powered by Gemma-3 + LoRA | 
    <a href='https://jolpi.ca/' target='_blank'>Data from Jolpica-F1</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
