import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="AI Customer Assistant", page_icon="🤖")
st.title("🤖 AI Support Assistant")
st.write("Welcome! Ask me anything about your queries.")

# Gemini API Integration
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-pro')

user_input = st.text_input("Type your query here:")

if user_input:
    with st.spinner("AI is thinking..."):
        try:
            response = model.generate_content(user_input)
            st.success("Response:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")