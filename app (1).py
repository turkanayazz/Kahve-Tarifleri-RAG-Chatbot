# Hugging Face Spaces tarafından otomatik olarak çalıştırılacak ana dosya (app.py)

import gradio as gr
from gradio.themes.utils import colors
import os 
from operator import itemgetter # Zincirleri birleştirmek için gerekli

# ------------------------------------------------------------------------------------------------
# KARARLI VE GÜNCEL IMPORT'LAR (Core/Community tabanlı)
# ------------------------------------------------------------------------------------------------
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain'in kararlı yapısı: Core Runnables
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------------------------
# YAPILANDIRMA VE RAG BAŞLATMA
# ------------------------------------------------------------------------------------------------

# Model ve Dosya Yolları
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-2.5-flash"
FILE_PATH = "kahve_tarifleri.txt" 

# GEMINI_API_KEY, Hugging Face Secrets'tan otomatik alınır.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

# 1. Veri Yükleme
loader = TextLoader(FILE_PATH)
documents = loader.load()

# 2. Veri Parçalama (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
)
texts = text_splitter.split_documents(documents)

# 3. Embedding Modelinin Ayarlanması
model_kwargs = {'device': 'cpu'} 
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 4. Vektör Veritabanı ve Retriever Oluşturma
db = Chroma.from_documents(texts, embeddings) 
retriever = db.as_retriever(search_kwargs={"k": 3})

# 5. RAG Zincirinin YENİDEN OLUŞTURULMASI (Runnable Mantığı)
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, api_key=GEMINI_API_KEY) 
prompt_template = """
Sen bir Kahve Tarifleri Uzmanı Chatbot'sun. Yalnızca sağlanan 'Bağlam'daki kahve tariflerini kullanarak kullanıcı sorularını yanıtla.
Eğer tarif bağlamda yoksa, kibarca 'Bu tarifi maalesef bilgi kaynaklarımda bulamadım.' de.

Bağlam: {context}

Soru: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(prompt_template)

# RAG Zinciri (Runnable)
rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | rag_prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------------------------------------------------------------------
# GRADIO ARAYÜZÜ İŞLEVLERİ
# ------------------------------------------------------------------------------------------------

# Gradio Chatbot Fonksiyonu
def chatbot_response(message, history):
    try:
        if not GEMINI_API_KEY:
             return "Hata: Gemini API Anahtarı yüklenemedi. Lütfen 'Settings -> Repository Secrets' kontrol edin."

        # rag_chain.invoke yerine .invoke({"question": message}) kullanılır
        answer = rag_chain.invoke({"question": message})
        
        return answer + "\n\n*(Bilgiler, kahve tarifleri bilgi kaynağımızdan derlenmiştir.)*"
            
    except Exception as e:
        return f"Üzgünüm, bir hata oluştu: {e}"

# Gradio Arayüzünün Oluşturulması
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="☕ Kahve Tarifleri Uzmanı Chatbot ☕",
    textbox=gr.Textbox(placeholder="Örn: Frappe nasıl yapılır?"),
    chatbot=gr.Chatbot(height=400),
    examples=["Latte nasıl yapılır?", "Affogato için malzemeler nelerdir?", "Dibek kahvesi ile Türk kahvesi arasındaki fark nedir?"],
    theme=gr.themes.Soft(
        primary_hue=colors.orange, 
        secondary_hue=colors.blue, 
        neutral_hue=colors.gray
    ),
    css="h1 {color: #6F4E37; font-family: 'Georgia', serif;}"
)

if __name__ == "__main__":
    demo.launch()