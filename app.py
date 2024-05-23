import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_doc_text(doc):
    text = ""
    doc = Document(doc)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are given the following context from a document. Generate a unique question based on this context. Then, provide four answer options, one of which is correct based on the context. The other three should be plausible but incorrect (distractors). Format the output as JSON with the correct answer marked as true and distractors as false.
    
    Context: {context}
    
    Your JSON output should follow this structure:
    {{
        "question": "Your generated question?",
        "options": [
            {{"option": "Correct answer", "is_correct": true}},
            {{"option": "Distractor 1", "is_correct": false}},
            {{"option": "Distractor 2", "is_correct": false}},
            {{"option": "Distractor 3", "is_correct": false}}
        ]
    }}
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

async def generate_questions_and_answers(num_questions):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    chain = get_conversational_chain()
    questions_and_answers = []
    
    for _ in range(num_questions):
        docs = new_db.similarity_search("Generate a question based on the provided context.")
        context = docs[0].page_content if docs else ""
        response = await chain.acall({"input_documents": docs, "context": context}, return_only_outputs=True)
        try:
            questions_and_answers.append(json.loads(response["output_text"]))
        except json.JSONDecodeError:
            st.error("Error in generating question. Please try again.")
    
    return questions_and_answers

def main():
    st.set_page_config(page_title="Generate Questions from Notes")
    st.header("Generate Questions and Answers from Notes")
    
    num_questions = st.number_input("Enter the number of questions to generate:", min_value=1, step=1)
    
    if num_questions:
        if st.button("Generate"):
            with st.spinner("Generating questions and answers..."):
                questions_and_answers = asyncio.run(generate_questions_and_answers(num_questions))
                st.json(questions_and_answers)
    
    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Upload your PDF or DOC File and Click on the Submit & Process Button", type=["pdf", "docx"], accept_multiple_files=False)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if uploaded_file:
                    if uploaded_file.type == "application/pdf":
                        raw_text = get_pdf_text(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text = get_doc_text(uploaded_file)
                    else:
                        st.error("Unsupported file type.")
                        return
                    
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("Please upload a PDF or DOC file.")

if __name__ == "__main__":
    main()
