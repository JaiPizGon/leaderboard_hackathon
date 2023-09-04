import streamlit as st

def main():
    st.title("Machine Learning Competition Leaderboard App")

    st.write("Welcome to the competition! Please log in to access your dashboard.")

    if st.button("Click me!"):
        st.write("Button clicked! You're ready to participate.")

if __name__ == "__main__":
    main()
