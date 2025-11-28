import streamlit as st
import os

# Debug info
st.write(f"PORT: {os.environ.get('PORT', 'Not Set')}")

st.title("Minimal Streamlit App")
st.write("This is a very simple Streamlit app to verify deployment.")

name = st.text_input("Enter your name:")

if st.button("Submit"):
    st.write(f"Hello, {name} 👋")
else:
    st.write("Click the button after typing your name.")
