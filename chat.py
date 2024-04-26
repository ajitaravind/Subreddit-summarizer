import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
)

from llama_index.readers.reddit import RedditReader

from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

def start_research(topic_string,post_limit):

    subreddits = ["electricvehicles"]
    if "," in topic_string:  # Check if there's a comma
        search_keys = topic_string.split(",") 
    else:
        search_keys = [topic_string]  # Treat as a single-item list
  
    topic = ", ".join(search_keys)
    print(f"I am topic: {topic}")
    
    post_limit = post_limit
    print(f"I am post_limit: {post_limit}")

    general_system_template = """ 

    **Task:** 
    
    Task 1: Analyze the following context and provide insights based on the main theme ```{topic}``` of discussion.

    **Focus on:**
    * Identifying the main threads of the discussion for the topic.
    * Highlighting contrasting or conflicting opinions on the topic.
    * Identifying the key arguments used to support different sides of the discussion.  For example, do people argue based on financial savings, environmental concerns, or other factors?
    * Analyzing the sentiment of comments. Are they generally positive, negative, or neutral about the topic wrt ```{topic}``? 
    * Answering these questions: 
        * What are the most common arguments and counterarguments raised against the main theme of the discussion?
        * Does the discussion reveal any misconceptions about main ```{topic}``` of the discussion?
        
    **Important:**
    * Do stick to the topic ```{topic}``` of the discussion. Do not include information not directly related to the topic.

    **Context:** {context}
    
    Task 2: Finally give a detailed summary of the overall discussion.
    ----
    """
    qa_prompt = ChatPromptTemplate.from_template(general_system_template)


    loader = RedditReader()
    documents = loader.load_data(
        subreddits=subreddits, search_keys=search_keys, post_limit=post_limit
    )

    new_documents = []
    for doc in documents:
        new_doc = Document(page_content=doc.text, metadata=doc.metadata)
        new_documents.append(new_doc)

    embeddings = OpenAIEmbeddings()

    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    # llm = ChatOpenAI(temperature=0,model = "gpt-3.5-turbo-0613")


    docsearch = Chroma.from_documents(new_documents, embeddings)
    retriever = docsearch.as_retriever(search_kwargs={'k': 30})

    chain = (
        {"context": retriever, "topic": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return chain,topic

st.set_page_config(
page_title='EV Wiz',
page_icon='🤖'
)

st.markdown("<h1 style='text-align: center; color: white;'>Welcome to EV Wiz </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>User research using Reddit</h4>", unsafe_allow_html=True)

st.sidebar.header("Enter required details")  # Sidebar header
topic_string = st.sidebar.text_area("Enter the topic to research",placeholder = "Enter multiple topics in the format: topic1,topic2,topic3 etc.")
post_limit = st.sidebar.number_input("Enter the number of posts to research", 1, 10)

with st.sidebar:
    st.markdown("")


if st.sidebar.button("Start the research"):
    if topic_string and post_limit:
        with st.spinner('Getting the information for you!!!!'):

            chain,topic = start_research(topic_string, post_limit)
            if "chain" not in st.session_state:
                st.session_state['chain'] = chain  
            result = st.session_state.chain.invoke(topic)
            st.markdown(result)    
  