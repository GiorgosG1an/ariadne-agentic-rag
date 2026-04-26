import json
from time import time
from typing import List
import frontmatter
from pathlib import Path

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from google.genai import types

from ariadne.infrastructure.qdrant import get_qdrant_clients
from ariadne.core.config import settings
from ariadne.core.dependencies import gemini_tokenizer

Settings.tokenizer = gemini_tokenizer

print(f"Setting up embedding model: {settings.embed_model}")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name=settings.embed_model,
    api_key=settings.google_api_key,
    embedding_config=types.EmbedContentConfig( 
        output_dimensionality=3072,
    ),
    embed_batch_size=100,
)

print("Setting up Qdrant vector database")
def get_vector_store():
    """Locally scoped initialization for Qdrant"""
    client, aclient = get_qdrant_clients()
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=settings.qdrant_collection,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=16,
    )


def ingest_courses() -> List[Document]:
    documents = []
    courses_folder = Path("data/courses")
    for course in courses_folder.glob("*.md"):
        post = frontmatter.load(course)

        metadata = {
            k: (", ".join(v) if isinstance(v, list) else str(v))
            for k, v in post.metadata.items()
        }

        metadata["year"] = "2025-2026"
        metadata["file_name"] = course.name
        metadata["pdf_name"] = "dit-course-list.pdf"

        document = Document(
            id_=f"course_{course.name}",
            text=post.content,
            metadata=metadata,
            metadata_separator="\n",
            metadata_template="{key}: {value}",
            # formated as defined in official Gemini API 
            # https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
            text_template=(                       
                f"title: {course.name} | text: ΠΛΗΡΟΦΟΡΙΕΣ ΜΑΘΗΜΑΤΟΣ \n{{metadata_str}}\n\n"
                f"--- ΑΝΑΛΥΤΙΚΟ ΠΕΡΙΕΧΟΜΕΝΟ ---\n{{content}}"
            ),
            excluded_embed_metadata_keys=[
                "url",
                "file_name",
                "pdf_name",
                "ects",
                "header_path",
            ],
            excluded_llm_metadata_keys=["keywords", "file_name", "header_path"],
        )
        documents.append(document)

    print(f"Loaded {len(documents)} Course Documents.")
    return documents


def ingest_general_markdown_files() -> List[Document]:
    documents = []
    md_folder = Path("data/md_general")

    for md_file in md_folder.glob("*.md"):
        post = frontmatter.load(md_file)
        title_val = post.metadata.get("title", md_file.stem.replace("_", " "))
        document = Document(
            id_=f"doc_{md_file.stem}",
            text=post.content,
            metadata={
                "year": "2025-2026",
                "file_name": f"{md_file.stem}.pdf",
            },
            metadata_separator="\n",
            metadata_template="{key}: {value}",
            # formated as defined in official Gemini API 
            # https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
            text_template=(
                f"title: {title_val} | text: Metadata:\n{{metadata_str}}\n\nContent:\n{{content}}"
            ),
            excluded_embed_metadata_keys=["year", "file_name", "description"],
        )

        documents.append(document)

    print(f"Loaded {len(documents)} General Markdown Documents.")
    return documents

def ingest_website() -> List[Document]:
    documents = []
    json_file = Path("data/website_data/dit_website_enriched.jsonl")
    
    if not json_file.exists():
        print(f"File {json_file} does not exist.")
        return documents

    with json_file.open(mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            text_content = data.get("cleaned_content")
            if not text_content:
                continue

            keywords_str = ", ".join(data.get("keywords", [])) if isinstance(data.get("keywords"), list) else ""

            metadata = {
                "url": data.get("url", ""),
                "title": data.get("title", ""),
                "content_category": data.get("content_category", "Website"), 
                "last_modified": data.get("last_modified", "N/A"),
                "keywords": keywords_str,
                "summary": data.get("summary", ""),
            }

            document = Document(
                text=text_content,
                metadata=metadata,
                metadata_separator="\n",
                metadata_template="{key}: {value}",
                # formated as defined in official Gemini API 
                # https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
                text_template=(
                    f"title: {data.get('title', 'Ιστοσελίδα Τμήματος')} | text: "
                    f"--- ΙΣΤΟΣΕΛΙΔΑ ΤΜΗΜΑΤΟΣ ---\nΜΕΤΑΔΕΔΟΜΕΝΑ:\n{{metadata_str}}\n\nΚΕΙΜΕΝΟ:\n{{content}}"
                ),
                excluded_embed_metadata_keys=['url', 'last_modified', 'category'],
                excluded_llm_metadata_keys=['keywords']
            )
            documents.append(document)
    
    print(f"✅ Loaded {len(documents)} Enriched Website Documents.")
    return documents


def ingest_announcements() -> List[Document]:
    documents = []
    json_file = Path("data/website_data/dit_announcements.jsonl")

    if not json_file.exists():
        print(
            f"Το αρχείο {json_file} δεν υπάρχει. Τρέξε πρώτα το announcements_scraper.py."
        )
        return documents

    with json_file.open(mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            text_content = data.get("cleaned_content")
            if not text_content:
                continue

            metadata = {
                "url": data.get("url", ""),
                "title": data.get("title", ""),
                "content_category": data.get("content_category", "Ανακοινώσεις"),
                "last_modified": data.get("last_modified", "N/A"),
            }

            document = Document(
                id_=f"announcement_{data['title']}",
                text=text_content,
                metadata=metadata,
                metadata_separator="\n",
                metadata_template="{key}: {value}",
                # formated as defined in official Gemini API 
                # https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
                text_template=(
                    f"title: Ανακοίνωση {data.get('title', '')} | text: {{content}}"
                ),  # no need to inject metadata, they are on content too.
                excluded_embed_metadata_keys=[
                    "url",
                    "content_category",
                    "last_modified",
                ],
                excluded_llm_metadata_keys=[],
            )
            documents.append(document)

    print(f"✅ Loaded {len(documents)} Announcements.")
    return documents

def ingest_schedule() -> List[Document]:
    documents = []
    json_file = Path("data/schedule_data/dit_program_spring_2025-26_v2_2.jsonl")

    if not json_file.exists():
        print(f"⚠️ Το αρχείο {json_file} δεν υπάρχει.")
        return documents
    
    with json_file.open(mode='r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Natural Language text for the embedding model
            text_content = (
                f"Στο ωρολόγιο πρόγραμμα, κάθε {data['day']} από τις {data['start_time']} έως τις {data['end_time']}, "
                f"οι φοιτητές του {data['year']}ου Έτους έχουν το μάθημα '{data['course_name']}' "
                f"({data['course_type']}). Το μάθημα διδάσκεται από: {data['instructor']}, "
                f"στην αίθουσα: {data['classroom']}."
            )

            metadata = {
                "content_category": "Εαρινό Πρόγραμμα Μαθημάτων 2025-2026",
                "day": str(data['day']).strip(),
                "year": str(data['year']).strip(),
                "course_name": data['course_name'],
                "instructor": data['instructor']
            }

            document = Document(
                id_=f'schedule_{data['course_name']}_{data['day']}',
                text=text_content,
                metadata=metadata,
                metadata_separator='\n',
                metadata_template="{key}: {value}",
                # formated as defined in official Gemini API 
                # https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
                text_template=(
                    f"title: Πρόγραμμα - {data['course_name']} | text: {{content}}"
                ),
                excluded_embed_metadata_keys=list(metadata.keys()),
                excluded_llm_metadata_keys=[]
            )

            documents.append(document)
    
    print(f"Loaded {len(documents)} Schedule Events.")
    return documents


def run_ingestion():

    print("Setting up Qdrant vector database...")
    vector_store = get_vector_store()
    
    print("\n" + "=" * 50)
    print("🚀 DIT RAG Ingestion Pipeline (Production)")
    print("=" * 50)
    print("1. Ingest Courses (Ολόκληρα αρχεία)")
    print("2. Ingest General Markdown (Μεσαία chunks)")
    print("3. Ingest Website Enriched (Μικρά chunks)")
    print("4. Ingest Ανακοινώσεις (Μικρά chunks)")
    print("5. Ingest Πρόγραμμα Εξαμήνου")
    print("6. Ingest ΟΛΑ τα παραπάνω")
    print("0. Έξοδος")

    choice = input("\nΕπίλεξε τι θέλεις να περάσεις στην Qdrant (0-5): ")

    all_nodes = []

    if choice in ["1", "6"]:
        courses = ingest_courses()
        course_parser = SentenceSplitter(chunk_size=7000, chunk_overlap=200, paragraph_separator='\n\n', tokenizer=gemini_tokenizer)
        all_nodes.extend(
            course_parser.get_nodes_from_documents(courses, show_progress=True)
        )

    if choice in ["2", "6"]:
        md_docs = ingest_general_markdown_files()
        md_parser = MarkdownNodeParser.from_defaults(header_path_separator="->")
        all_nodes.extend(
            md_parser.get_nodes_from_documents(md_docs, show_progress=True)
        )

    if choice in ["3", "6"]:
        web_docs = ingest_website()
        web_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=200, paragraph_separator='\n\n', tokenizer=gemini_tokenizer)
        all_nodes.extend(
            web_parser.get_nodes_from_documents(web_docs, show_progress=True)
        )

    if choice in ["4", "6"]:
        announcement_docs = ingest_announcements()
        ann_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=200, paragraph_separator='\n\n', tokenizer=gemini_tokenizer)
        all_nodes.extend(
            ann_parser.get_nodes_from_documents(announcement_docs, show_progress=True)
        )
    if choice in ["5", "6"]:
        schedule_docs = ingest_schedule()
        schedule_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50, tokenizer=gemini_tokenizer)
        all_nodes.extend(
            schedule_parser.get_nodes_from_documents(schedule_docs, show_progress=True)
        )


    if choice == "0" or not all_nodes:
        print("Ακύρωση. Καμία ενέργεια δεν εκτελέστηκε.")
        return

    print(f"\n📦 Έτοιμα για ingestion: {len(all_nodes)} nodes συνολικά.")
    confirm = input("Θέλεις να προχωρήσεις στην αποθήκευση στην Qdrant; (y/n): ")

    if confirm.lower() == "y":
        print("\n⏳ Generating dense & sparse embeddings and indexing into Qdrant...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        valid_nodes =[]
        for node in all_nodes:
            if node.get_content() and len(node.get_content().strip()) > 5:
                valid_nodes.append(node)   

        index = VectorStoreIndex(
            nodes=[],
            embed_model=Settings.embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
        import time # Πρόσθεσέ το στα imports πάνω πάνω

        print("\n⏳ Ξεκινάει το ασφαλές Ingestion (Node by Node με Retry)...")
        successful_nodes = 0
        failed_nodes = 0
        
        for i, node in enumerate(valid_nodes):
            max_retries = 3 # Θα δοκιμάσει μέχρι 3 φορές
            success = False
            
            for attempt in range(max_retries):
                try:
                    index.insert_nodes([node])
                    success = True
                    break 
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Αν είναι timeout ή 503 (server overloaded)
                    if "timed out" in error_msg or "503" in error_msg or "500" in error_msg:
                        if attempt < max_retries - 1:
                            print(f"⏳ Timeout στο node, επανάληψη ({attempt + 1}/{max_retries}) σε 2 δευτ...")
                            time.sleep(2)
                        else:
                            print(f"❌ Οριστική αποτυχία μετά από {max_retries} προσπάθειες. Σφάλμα: {e}")
                    else:
                        print(f"⚠️ Προσπεράστηκε προβληματικό node {node.get_content()} \nΣφάλμα: {e}")
                        break 
            
            if success:
                successful_nodes += 1
                if successful_nodes % 20 == 0:
                    print(f"✅ Ενσωματώθηκαν επιτυχώς {successful_nodes}/{len(valid_nodes)} nodes...")
            else:
                failed_nodes += 1

        print("\n" + "=" * 50)
        print("🎉 Ingestion Completed!")
        print(f"Επιτυχίες: {successful_nodes}")
        print(f"Αποτυχίες: {failed_nodes}")
        print("=" * 50)
        print("Ingestion completed successfully!")
    else:
        print("Ακύρωση.")


if __name__ == "__main__":
    run_ingestion()
