import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
import os

st.set_page_config(page_title="smart study buddy", layout="wide")
st.title("smart study buddy:\nfor makeing your study more feasible")

@st.cache_resource
def embedings_model():
    st.info("initializing Huggingface embeddings")
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs={'device':'cpu'}
    return HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs)

@st.cache_resource
def llm():
    st.info("initializing Qwen/Qwen2-1.5B-instruct via hggingfacepipeline")
    model_id = "Qwen/Qwen2-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=1024, 
        temperature=0.1
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def pdf_prossecing(uploaded_file, text_splitter):
    st.info("started pdf scaning")
    #pdf_bytes = uploaded_file.read()
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_docs = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text = page.get_text()
        full_docs.append(Document(
            page_content=text, 
            metadata={"source": uploaded_file.name, "page": page_num + 1}
        ))
    final_chunks = text_splitter.split_documents(full_docs)
    
    st.success(f"PDF processed into {len(final_chunks)} searchable chunks.")
    return final_chunks
try:
    embeddings = embedings_model()
    llm_chain = llm()
except Exception as e:
    st.error(f"Error initializing models. Check your dependencies (torch, transformers) and Hugging Face login. Error: {e}")
    st.stop()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

with st.sidebar:
    st.header("Upload Document")
    uploaded_file=st.file_uploader("choose a pdf file ",type = 'pdf')

if uploaded_file and 'vectorstore' not in st.session_state:
    with st.spinner("Processing document and creating vector index..."):
        chunks = pdf_prossecing(uploaded_file, text_splitter)
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        st.sidebar.success("Index Ready! Ask a question.")
        uploaded_file = None

if 'vectorstore' in st.session_state and st.session_state.vectorstore:
    
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    raw_prompt_template = """
        you are an expert study buddy. Use ONLY the following context to answer the question. 
        for every fact in your answer, mention the corresponding page number(s) from the context
        in brackets, e.g., [Page 5]. if the context does not contain the answer, state that you
        do not know.

        Context: {context}

        Question: {input}

        Answer:
        """
    qa_prompt = PromptTemplate.from_template(raw_prompt_template)
    
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm_chain,
        prompt=qa_prompt
        )
    
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    user_question = st.chat_input("ask a question about your notes...")
    if user_question:
        
        st.chat_message("user").write(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("searching and generating answer..."):

                result = qa_chain.invoke({"input": user_question})
                st.write(result['answer'])
                st.markdown("---")
                st.subheader("cited Sources:")

                sources = set()
                for doc in result['context']:
                    page = doc.metadata.get('page', 'N/A')
                    sources.add(f"Page {page}")
                
                if sources:
                    st.markdown(f"**referenced Pages:** {', '.join(sorted(list(sources)))}")
                else:
                    st.markdown("no sources cited.")

"""qa_prompt = PromptTemplate.from_template(raw_prompt_template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_chain,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )""" #ignore this 