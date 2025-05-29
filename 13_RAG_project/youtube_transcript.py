from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from pytube import YouTube

load_dotenv()

st.title("YouTube Transcript Summarizer")

video_id = st.text_input("Enter the Youtube video Id")

transcript = ""



def get_transcript(video_id):
    try:
        # If you don’t care which language, this returns the “best” one
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video.")


if video_id:
    transcript = get_transcript(video_id)
    # url = f"https://www.youtube.com/watch?v={video_id}"

    # yt = YouTube(url)
    # print("Title:", yt.title)
    # st.write("### Title of the video")
    # st.write(yt.title)

def text_Splitter(transcript):
    # Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.create_documents([transcript])
    return chunks

def create_vector_store(chunks):
    # Create a vector store from the chunks
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_docs(retriever):
    return "\n\n".join(doc.page_content for doc in retriever)



if transcript:
    vector_store = create_vector_store(text_Splitter(transcript))

    retriever = vector_store.as_retriever(search_type = 'mmr', search_kwargs={"k": 3, 'lambda': 0.5})

    llm = ChatOpenAI(model = "gpt-4")

    def get_docs(retriever):
        return "\n\n".join(doc.page_content for doc in retriever)


    parallel_chain = RunnableParallel(
        { "document": retriever | RunnableLambda(get_docs),
        "query": RunnablePassthrough()
        }
    )

    prompt = PromptTemplate(
        input_variables=["document", "query"],
        template="You are a virtual assistant, Answer the queries of the user based on the document available, if query is not relevant to document then just say I don't know. \n\n Document: {document} \n\n Query: {query} \n\n Answer:"
    )

    final_chain = parallel_chain | prompt | llm | StrOutputParser()

    query = st.text_input("Enter your query", key="query")

    if st.button("Process Query"):
        results = final_chain.invoke(query)
        st.write("### Summary")
        st.write(results)









