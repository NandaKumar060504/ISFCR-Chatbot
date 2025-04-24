import gradio as gr
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import ollama
import imaplib
import email
import PyPDF2
import os 
import langchain_core
from langchain.vectorstores import FAISS
# #DataPreprocessing And Loading 

def load_retriever(mails):

    document=[] #to Store mail

    if len(mails):
        for i in mails:
            document.append(langchain_core.documents.base.Document(page_content=i))


    loader = WebBaseLoader(
        web_paths=("https://www.isfcr.pes.edu/",
              "https://www.isfcr.pes.edu/about",
              "https://www.isfcr.pes.edu/team",
              "https://www.isfcr.pes.edu/research",
              "https://www.isfcr.pes.edu/courses",
              "https://www.isfcr.pes.edu/team/preet-kanwal",'https://www.isfcr.pes.edu/about',
              'https://www.isfcr.pes.edu/testimonials',
            'https://www.isfcr.pes.edu/events',
            'https://www.isfcr.pes.edu/bug-bounty',
            'https://www.isfcr.pes.edu/resources','https://www.isfcr.pes.edu/team/vanitha-g', 'https://www.isfcr.pes.edu/team/dr.-ashok-kumar-patil', 'https://www.isfcr.pes.edu/team/dr.-radhika-m.-hirannaiah', 'https://www.isfcr.pes.edu/team/dr.-shruti-jadon', 'https://www.isfcr.pes.edu/team/pavan-a-c', 'https://www.isfcr.pes.edu/team/vadiraja-acharya', 'https://www.isfcr.pes.edu/team/charanraj-b-r', 'https://www.isfcr.pes.edu/team/dr.-adithya-b', 'https://www.isfcr.pes.edu/team/dr.-s.s.-iyengar', 'https://www.isfcr.pes.edu/team/dr-nagasundari-s', 'https://www.isfcr.pes.edu/team/dr.-roopa-ravish', 'https://www.isfcr.pes.edu/team/revathi', 'https://www.isfcr.pes.edu/team/preet-kanwal', 'https://www.isfcr.pes.edu/team/indu-radhakrishnan', 'https://www.isfcr.pes.edu/team/prasad-b-honnavalli', 'https://www.isfcr.pes.edu/team/sapna-v-m', 'https://www.linkedin.com/company/c-isfcr/', 'https://www.isfcr.pes.edu/team/sushma-e')
    )

    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    

    print("in here")

    if len(document):
        splits+=document
    
    print(splits)

    # Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # # Create the retriever
    # retriever = vectorstore.as_retriever()

    db = FAISS.from_documents(splits,embeddings
                            )

    retriever = db.as_retriever()
    return retriever


def get_attachments(msg):
    for part in msg.walk():
        if part.get_content_maintype()=='multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        fileName = part.get_filename()

        if bool(fileName):
            filePath = os.path.join(fileName)
            with open(filePath,'wb') as f:
                f.write(part.get_payload(decode=True))
        from PyPDF2 import PdfReader

        # creating a pdf reader object
        reader = PdfReader(fileName)

        # printing number of pages in pdf file
        print(len(reader.pages))

        # getting a specific page from the pdf file
        page = reader.pages[0]

        # extracting text from page
        text = page.extract_text()
        return (text)
    

def getmail():
    #Load the user name and passwd from yaml file
    user, password = "vivekras698@gmail.com","hpmvorxgtjuwtfpb"

    #URL for IMAP connection
    imap_url = 'imap.gmail.com'

    # Connection with GMAIL using SSL
    my_mail = imaplib.IMAP4_SSL(imap_url)

    # Log in using your credentials
    my_mail.login(user, password)

    # Select the Inbox to fetch messages
    my_mail.select('Inbox')

    key = 'FROM'
    value = 'saddy2410@gmail.com'
    _, data = my_mail.search(None, key, value)

    mail_id_list = data[0].split()

    msgs = []
    for num in mail_id_list:
        typ, data = my_mail.fetch(num, '(RFC822)')
        msgs.append(data)

    gmail_text = []
    for msg in msgs[::-1]:
        for response_part in msg:
            if type(response_part) is tuple:
                my_msg=email.message_from_bytes((response_part[1]))
                print("_________________________________________")
                print ("subj:", my_msg['subject'])
                print ("from:", my_msg['from'])
                print ("body:")
                for part in my_msg.walk():
                    #print(part.get_content_type())
                    if part.get_content_type() == 'text/plain':
                        print (part.get_payload())
                        gmail_text.append("mail from : "+my_msg['from'] +"Body :"+part.get_payload())
                    attachment=get_attachments(my_msg)
                    if len(attachment)!=0:

                        gmail_text.append("The Pdf sent by " +my_msg['from']  +"Content of the pdf :"+attachment)

        break

    return load_retriever(gmail_text)



    

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
retriever=getmail()

def rag_chain(question):

    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    print(formatted_context)

    # Create prompt from prompt template
  
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    
    response = ollama.chat(model='mistral:instruct', messages=[{'role': 'user', 'content': formatted_prompt}])

    return response['message']['content']

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs=["text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a URL and a query to get answers from the RAG chain."
)

# Launch the app
iface.launch(share=True)