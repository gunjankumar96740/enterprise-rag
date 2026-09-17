import streamlit as st
import requests

st.title("📚 Conversational RAG")

# -------------------------
# CHAT HISTORY
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# PDF UPLOAD
# -------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Process PDF"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/upload",
                files=files,
                timeout=120
            )

            if response.status_code == 200:
                st.success("PDF processed successfully! ✅")

            else:
                st.error(
                    f"Error {response.status_code}: "
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. "
                "Make sure the FastAPI server is running."
            )


# -------------------------
# CLEAR CHAT
# -------------------------

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -------------------------
# CHAT INPUT
# -------------------------

question = st.chat_input(
    "Ask a question about the document..."
)


if question:

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    try:

        response = requests.post(
            "http://127.0.0.1:8000/chat",

            json={
                "question": question,
                "history": st.session_state.messages[:-1]
            },

            timeout=120
        )

        if response.status_code == 200:

            data = response.json()

            answer = data["answer"]
            sources = data["sources"]

            # Display answer
            with st.chat_message("assistant"):
                st.write(answer)

                if sources:
                    st.caption(
                        f"Sources: {sources}"
                    )

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:

            st.error(
                f"Error {response.status_code}: "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI."
        )

    except requests.exceptions.Timeout:

        st.error(
            "Request timed out."
        )

    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )