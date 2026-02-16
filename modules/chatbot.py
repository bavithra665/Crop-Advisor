import os
import requests
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
try:
    from groq import Groq
except ImportError:
    Groq = None

load_dotenv()

class AgriBot:
    def __init__(self):
        # Configuration (Lightweight)
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.pc_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "agri-knowledge"
        
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2"
        
        # Heavy models (Lazy loaded)
        self.embed_model = None
        self.pc = None
        self.index = None
        self.gemini_model = None
        self.models_loaded = False

    def _load_models(self):
        if self.models_loaded: return
        
        print("⏳ Lazy loading AI models...")
        
        # 1. Load Embedding Model
        if not self.embed_model:
            try:
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Embedding model loaded")
            except Exception as e:
                print(f"⚠️ Embedding model failed: {e}")

        # 2. Load Pinecone
        if self.pc_api_key and self.embed_model and not self.index:
            try:
                self.pc = Pinecone(api_key=self.pc_api_key)
                existing_indexes = [idx.name for idx in self.pc.list_indexes().indexes]
                if self.index_name not in existing_indexes:
                    try:
                        self.pc.create_index(
                            name=self.index_name,
                            dimension=384,
                            metric='cosine',
                            spec=ServerlessSpec(cloud='aws', region='us-east-1')
                        )
                    except: pass # Ignore if creation fails/exists
                self.index = self.pc.Index(self.index_name)
                print("✅ Pinecone initialized")
            except Exception as e:
                print(f"⚠️ Pinecone init error: {e}")

        # 3. Load Gemini
        if self.gemini_key and not self.gemini_model:
            try:
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except: pass
            
        self.models_loaded = True

    def search_context(self, query):
        """Search Pinecone for relevant context (optional enhancement)"""
        if not self.index or not self.embed_model: 
            return ""
        try:
            query_em = self.embed_model.encode(query).tolist()
            results = self.index.query(vector=query_em, top_k=3, include_metadata=True)
            return "\n".join([res['metadata']['text'] for res in results['matches']])
        except: 
            return ""

    def get_answer(self, query, history=[]):
        # Try lightweight API-based answers first (no model loading needed)
        context = ""
        
        # Only load models if we have the API keys (optional enhancement)
        if self.pc_api_key:
            try:
                self._load_models()
                context = self.search_context(query)
            except:
                pass  # Continue without RAG context
        
        # Requesting a more detailed, multi-paragraph explanation
        prompt = f"""
        You are a highly detailed and friendly agricultural expert. 
        Your goal is to provide a comprehensive, multi-paragraph explanation to help the farmer. 
        Don't just give a short answer; explain the 'why' and give specific actionable tips.
        
        Background Information: {context}
        
        User's Question: {query}
        
        Detailed, casual, and encouraging answer:
        """
        
        # Priority 1: Groq (if key exists) - FASTEST, NO MODEL NEEDED
        if self.groq_key and Groq:
            try:
                client = Groq(api_key=self.groq_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Groq failed: {e}")

        # Priority 2: Gemini (if key exists)
        if self.gemini_key:
            try:
                if not self.gemini_model:
                    self._load_models()
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"⚠️ Gemini failed: {e}")

        # Priority 3: Local Ollama (Local Dev Only)
        try:
            payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
            print(f"DEBUG: Trying Local AI ({self.ollama_model})...")
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200: 
                return response.json().get('response')
        except: 
            pass

        # Ultimate Fallback: Detailed Manual Expert Mode
        q_lower = query.lower()
        if "rice" in q_lower:
            return """Rice is a fascinating crop, but it’s quite demanding! It thrives in clayey loam soil which holds water well, as it typically requires over 1000mm of rainfall or consistent irrigation.

To get the best harvest, I recommend an NPK ratio of 80:40:40. You should apply half of the Nitrogen and all of the Phosphorus and Potassium during the transplanting stage. The remaining Nitrogen should be applied in two splits: once at the tillering stage and again at the panicle initiation stage. This ensures the plant has energy exactly when it needs to grow those grains!"""
        
        if "wheat" in q_lower:
            return """Wheat is a 'Rabi' crop, meaning it loves the cooler months of October and November. It needs well-drained, loamy soil to prevent root rot.

For a strong yield, a balanced NPK ratio of 120:60:40 is standard. Actionable tip: ensure the first 'Crown Root Initiation' irrigation happens exactly 21 days after sowing—this is the most critical time for the plant's development. Also, keep an eye out for 'Yellow Rust' disease if the weather gets too humid!"""
        
        if "cotton" in q_lower:
            return """Cotton is often called 'White Gold,' but it needs a lot of sun and careful water management. It grows best in deep black soils (Regur soil) which have high water-retaining capacity.

I highly recommend using drip irrigation; it can save up to 40% more water compared to traditional methods. For nutrients, aim for an NPK of 100:50:50. Keep a close watch for the Pink Bollworm pest during the flowering stage—using pheromone traps early on can save your entire crop without needing heavy chemicals!"""
        
        if "pest" in q_lower:
            return """Managing pests is all about being proactive rather than reactive! Instead of jumping straight to harsh chemicals, try Integrated Pest Management (IPM).

First, use Neem-based sprays or light traps to monitor what bugs are in your field. Biological controls like Trichogramma wasps can naturally hunt down harmful pests. If you must use chemicals, only spray the affected patches to keep your soil health and helpful insects (like bees) safe!"""
        
        if "npk" in q_lower:
            return """NPK is the foundation of plant health, and understanding it is key!
- **Nitrogen (N)**: This is for the 'Green.' It helps with leaf growth and lush foliage.
- **Phosphorus (P)**: This is for the 'Roots' and 'Fruits.' It helps the plant establish a strong base and produce healthy seeds or grains.
- **Potassium (K)**: This is for 'Overall Health.' It helps the plant fight off diseases and survive through tough weather like droughts.

Always test your soil before adding more—sometimes your soil already has plenty of one element, and adding more is just a waste of money!"""
        
        return "I'd love to give you a detailed breakdown! Could you tell me a bit more? For example, what is your soil type, and which specific crop or pest are you most concerned about right now?"

# Example knowledge seed
SEED_DATA = [
    "Rice requires high humidity and heavy rainfall, typically above 1000mm. It grows best in clayey loam soil.",
    "Millets are highly drought-resistant and can grow in regions with less than 500mm of annual rainfall.",
    "NPK (Nitrogen, Phosphorus, Potassium) ratio of 20:20:20 is generally recommended for balanced soil health.",
    "Integrated Pest Management (IPM) for Maize involves using biological controls and monitoring pheromone traps.",
    "Drip irrigation saves up to 40% more water compared to furrow irrigation for cotton crops.",
    "Wheat is a Rabi crop usually planted in late October to November in South Asia."
]
