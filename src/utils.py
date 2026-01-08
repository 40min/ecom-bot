import hashlib
import re
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

def apply_func_to_all_docs(func):
    """Helper to apply a function to all documents in a list"""
    def process_docs(docs):
        for doc in docs:
            doc.page_content = func(doc.page_content)
        return docs
    return process_docs

def filter_and_dedup(docs: list[Document], min_length: int = 10) -> list[Document]:
    unique_hashes = set()
    filtered = []
    stats = {"duplicates": 0, "too_short": 0, "empty": 0}
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            stats["empty"] += 1
            continue
        if len(text) < min_length:
            stats["too_short"] += 1
            continue
        
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in unique_hashes:
            print(f"Пропускаем дубликат: '{text[:50]}...'")
            stats["duplicates"] += 1
            continue
        unique_hashes.add(h)
        filtered.append(doc)
    print(f"\n[filter_and_dedup] Первоначально: {len(docs)} чанков")
    print(f"[filter_and_dedup] Удалено дубликатов: {stats["duplicates"]}, слишком коротких: {stats["too_short"]}, пустых: {stats["empty"]}")
    print(f"[filter_and_dedup] Осталось: {len(filtered)} чанков")
    return filtered

def clean_text(text: str) -> str:
    # Clean whitespace characters
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    
    # Remove file paths and URLs
    text = re.sub(r"file:///[^ \n]*", "", text)
    text = re.sub(r"https?://[^\s\n]*", "", text)
    text = re.sub(r"\[\s*https?[^\]]*\]", "", text)
    
    # Remove date/time patterns like "1/7/26, 3:11 PM" or "2024-01-15, 14:30"
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d{4}-\d{2}-\d{2},\s*\d{1,2}:\d{2}(?::\d{2})?", "", text)
    
    # Remove standalone numbers or numeric patterns (like "1/1", "1.2.3", etc)
    text = re.sub(r"\b\d+(?:/\d+)+\b", "", text)
    text = re.sub(r"\b\d+\.\d+(?:\.\d+)*\b", "", text)
    
    # Remove page numbers and isolated digits
    text = re.sub(r"\b\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    # Remove patterns like "[1]", "[2]" (citation brackets)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    
    # Normalize newlines: max 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Clean up lines but preserve paragraph structure
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)
    
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    
    # Clean up any remaining orphaned punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    
    return text.strip()

clean_docs = RunnableLambda(apply_func_to_all_docs(clean_text))

class LoaderRunnable(RunnableLambda):
    """Wrapper to make loaders compatible with RunnableParallel"""
    def __init__(self, loader, topic: str, source_type: str = "pdf"):
        def load_and_tag(_):
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_type'] = source_type
                doc.metadata['topic'] = topic
            print(f"Загружено {len(docs)} документов ({topic})")
            print(f"Пример метаданных: {docs[0].metadata}")
            return docs
        super().__init__(load_and_tag)
