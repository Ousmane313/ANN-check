



    
with SuppressStdout():

# Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role":"assistant",
                "content":"Posez vos questions sur l'actualité"
            }
        ]

# Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Process and store Query and Response
    def llm_function(query):
        #retriever = vectorstore.similarity_search(query)
        response = query_engine.query(query)
    
        # Displaying the Assistant Message
        with st.chat_message("assistant"):
            st.markdown(response)
# Storing the User Message
        st.session_state.messages.append(
            {
                "role":"user",
                "content": query
            }
        )

    # Storing the User Message
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content": response
            }
        )

# Accept user input
    query = st.chat_input("Bonjour!")

# Calling the Function when Input is Provided
    if query:
    # Displaying the User Message
        with st.chat_message("user"):
            st.markdown(query)

        llm_function(query)
