import streamlit as st

# App title
st.title("Minimal Streamlit App")

# Description
st.write("This is a very simple Streamlit app to verify deployment.")

# Text input
name = st.text_input("Enter your name:")

# Button
if st.button("Submit"):
    st.write(f"Hello, {name} 👋")
else:
    st.write("Click the button after typing your name.")