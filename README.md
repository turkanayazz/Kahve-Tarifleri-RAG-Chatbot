# ☕ AKBANK GENAI BOOTCAMP PROJESİ: KAHVE TARİFLERİ RAG CHATBOT'U

## PROJENİN AMACI
Bu proje, RAG (Retrieval Augmented Generation) mimarisini kullanarak kahve tarifleri ve demleme yöntemleri hakkında doğru ve bağlama dayalı bilgi sağlayan bir chatbot geliştirmeyi ve bu çözümü Gradio web arayüzü üzerinden sunmayı amaçlamaktadır.

## VERİ SETİ HAKKINDA BİLGİ
Projede, **kahve_tarifleri.txt** adında kendi hazırladığımız bir metin dosyası kullanılmıştır. Bu veri seti, 24 farklı popüler kahve tarifini (Latte, Türk Kahvesi, Cold Brew, vb.), malzemelerini ve adım adım yapılışlarını içermektedir. Bu dosya, RAG sisteminin bilgi tabanını (Knowledge Base) oluşturur.

## KULLANILAN YÖNTEMLER (ÇÖZÜM MİMARİSİ)
Projenin RAG (Retrieval Augmented Generation) mimarisi aşağıdaki kararlı teknolojiler üzerine kurulmuştur:

1.  **RAG Çerçevesi:** LangChain (Runnable yapısı kullanılmıştır).
2.  **Büyük Dil Modeli (Generation):** Gemini 2.5 Flash API.
3.  **Gömme Modeli (Embedding):** HuggingFace BGE Embeddings (Kararlılık için Google Embeddings yerine tercih edilmiştir).
4.  **Vektör Veritabanı:** ChromaDB.
5.  **Web Arayüzü:** Gradio (Hugging Face Spaces'e deploy edilmiştir).

**RAG Süreci:** Metinler $400$ karakterlik parçalara ayrılır, BGE ile vektörleştirilir, ChromaDB'ye kaydedilir. Kullanıcı sorgusu geldiğinde, en alakalı $3$ parça geri getirilir ve Gemini modeline gönderilerek nihai yanıt üretilir.

## ELDE EDİLEN SONUÇLAR ÖZET
Geliştirilen RAG zinciri, kullanıcı sorgularını alıp bilgi kaynağımızdan en alakalı tarifi geri getirmekte ve Gemini 2.5 Flash modeli bu bağlamı kullanarak adım adım, okunaklı bir yanıt üretmektedir. Bu sayede chatbot, yalnızca kendisine sağlanan bilgilere dayanarak tutarlı ve güvenilir cevaplar vermektedir.

## 🔗 CANLI WEB ARAYÜZÜ (KALICI DEPLOY LİNKİ)
Projenin çalışan demosu, kalıcı barındırma için Hugging Face Spaces'e deploy edilmiştir.

**KALICI CHATBOT LİNKİ:** **https://huggingface.co/spaces/turkanayaz/kahvecim**
