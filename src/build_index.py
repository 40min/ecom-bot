from functools import partial

from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import (
    LoaderRunnable, 
    clean_docs, 
    filter_and_dedup,
)


DEFAULT_EMBED_MODEL_NAME = "cointegrated/rubert-tiny2"
DELIMETER = "\n" + "=" * 50 + "\n"


class KnowledgeDB:
    def __init__(
        self,         
        source_docs: list[Path], 
        indices_path: Path,
        embed_model: str = DEFAULT_EMBED_MODEL_NAME, 
    ):
        self.docs = source_docs        
        self.embed_model = embed_model
        self.indices_path = indices_path    


    def get_vector_store(self) -> FAISS:
        """Load FAISS index"""
        print("Загрузка индекса...")

        if not self.indices_path.exists():
            print("Индекс не найден")
            return self.build_index()
    
        embed_model = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.load_local(
            str(self.indices_path),
            embed_model,
            allow_dangerous_deserialization=True
        )
    
        print("✅ Индекс загружен")
        return vector_store

    def build_index(self) -> FAISS:
        print("Создание индекса...")

        # Create loaders dynamically
        pdf_loaders_with_cleanup = {
            doc.name: LoaderRunnable(
                PyMuPDFLoader(file_path=doc, extract_images=False, mode="page"),
                topic=doc.name
            ) | clean_docs
            for doc in self.docs
        }        
        
        # Create the chain
        chain = (
            RunnableParallel(**pdf_loaders_with_cleanup)  # type: ignore[operator]
            | RunnableLambda(lambda x: sum(x.values(), []))  # type: ignore[arg-type]
            | RunnableLambda(partial(filter_and_dedup, min_length=10))  # type: ignore[arg-type]
        )

        # Execute the chain
        all_docs_filtered = chain.invoke(None)

        print(DELIMETER)
        print(f"\nВсего документов: {len(all_docs_filtered)}")

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
    
        splitted_docs = splitter.split_documents(all_docs_filtered)
        print(f"Было документов: {len(all_docs_filtered)}, стало фрагментов: {len(splitted_docs)}")
        
            
        if splitted_docs:
            print(DELIMETER)
            print(f"\nПример чанка:\n\n{splitted_docs[0].page_content}")
            print(DELIMETER)
            print(f"Метаданные чанка: \n{splitted_docs[0].metadata}")        
            
        embed_model = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
        vector_store = FAISS.from_documents(splitted_docs, embed_model)
        vector_store.save_local(str(self.indices_path))
    
        print("FAISS индекс сохранен")

        return vector_store    

if __name__ == "__main__":
    
    load_dotenv()

    knwodledge_docs = [
        Path("./data/policy_returns.pdf"),
        Path("./data/shipping_terms.pdf"),
    ]
    db_apth = Path("./vectordb")

    knowledge_db = KnowledgeDB(knwodledge_docs, db_apth)
    knowledge_db.build_index()

    print("✅ Индекс создан")
    
