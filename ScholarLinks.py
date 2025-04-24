import nest_asyncio
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
nest_asyncio.apply()
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


attachment=""
# Articles to index
articles = [
            " https://scholar.google.co.in/citations?view_op=list_works&hl=en&hl=en&user=jpEQplwAAAAJ",
            "https://scholar.google.co.in/citations?user=Od8qqtMAAAAJ&hl=en",
            "https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=GF0YZuUAAAAJ",
            "https://scholar.google.com/citations?user=p3IW9AkAAAAJ&hl=en","https://scholar.google.com/citations?user=5YK8wMoAAAAJ&hl=en", "https://scholar.google.co.in/citations?user=KbiV7csAAAAJ&hl=en","https://scholar.google.com/citations?user=t8NaoXkAAAAJ&hl=en&oi=ao","https://scholar.google.com/citations?user=Brfm5C4AAAAJ&hl=en","https://scholar.google.com/citations?view_op=list_works&hl=en&user=mLJdC2wAAAAJ","https://scholar.google.com/citations?hl=en&user=whKHzp0AAAAJ","https://scholar.google.com/citations?hl=en&user=I18gfJQAAAAJ"
    
]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()
print(docs)

# Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
print(docs_transformed)

 # Chunk text

text_splitter = CharacterTextSplitter(chunk_size=100,
                                        chunk_overlap=0)  #chunk overlap=200

chunked_documents=text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                            FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

retriever = db.as_retriever()


#PROMPT TEMPLATE

prompt_template = """
### [INST] Instruction: Answer any questions regarding the information present in Google Scholar for each faculty member with utmost accuracy and do not answer questions if you do not know the answer. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain

######### Groq API ###########

import os
# GROQ_API_KEY= "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"
# groq_api_key = os.getenv(GROQ_API_KEY)

llm = ChatGroq(model="mixtral-8x7b-32768", api_key="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")
llm_chain = LLMChain(llm=llm, prompt=prompt)


#RAG CHAIN 

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain


)

# result = rag_chain.invoke("List all the publications under Charanraj B R. ")

# print(result['context'])
# print(result['text'])

# Initialize message history for conversation
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# message_history = ChatMessageHistory()
    
#     # Memory for conversational context
# memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         output_key="answer",
#         chat_memory=message_history,
#         return_messages=True,
#     )

#     # Create a chain that uses the vector store
# chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(),
#         memory=memory,
#         return_source_documents=True,
#     )

# def answer_question(query):
#     response = chain(query)
#     return response


# second function
def chat(chat_history, user_input):

  result = rag_chain.invoke(user_input)

  response = ""
  for letter in ''.join(result['text']): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
      response += letter + ""
      yield chat_history + [(user_input, response)]

# import gradio as gr
#  # Gradio interface
# iface = gr.Interface(
#     # fn=rag_chain,
#     fn=answer_question,
#     inputs=["text"],
#     outputs="text",
#     title="RAG Chain Question Answering",
#     description="Google Scholar - PESU ISFCR"
# )

# # Launch the app
# iface.launch(share=True)


import gradio as gr
with gr.Blocks() as demo:
    with gr.Tab("Knowledge Bot"):
#          inputbox = gr.Textbox("Google Scholar PESU ISFCR Bot....")
          chatbot = gr.Chatbot()
          message = gr.Textbox ("What is this document about?")
          message.submit(chat, [chatbot, message], chatbot)

demo.queue().launch(debug = True)