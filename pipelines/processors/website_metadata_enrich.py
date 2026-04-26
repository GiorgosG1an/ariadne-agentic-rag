import asyncio
import json
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

INPUT_FILE = Path("website_data/dit_website.jsonl")
OUTPUT_FILE = Path("website_data/dit_website_enriched.jsonl")

CONCURRENT_LLM_REQUESTS = 3 

llm = GoogleGenAI(
    model='gemini-2.5-flash-lite',
    temperature=0.2
)

class PageEnrichment(BaseModel):
    content_category: str = Field(
        description="Η κατηγορία της σελίδας. Επίλεξε ΑΥΣΤΗΡΑ μία από: 'Προπτυχιακά', 'Μεταπτυχιακά', 'Προσωπικό', 'Εγκαταστάσεις', 'Έρευνα', 'Κανόνες/Πολιτικές', 'Γενικές Πληροφορίες'."
    )
    keywords: list[str] = Field(
        description="3 έως 5 λέξεις-κλειδιά στα Ελληνικά που χαρακτηρίζουν το κείμενο, και έχουν σημασιολογικό νόημα."
    )
    summary: str = Field(
        description="Μια σύντομη περίληψη 1-2 προτάσεων του περιεχομένου."
    )

async def enrich_single_page(data: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        text = data.get("cleaned_content", "")
        title = data.get("title", "Άγνωστο")

        if len(text) < 100:
            data["content_category"] = "Άλλο"
            data["summary"] = "Πολύ μικρό κείμενο."
            return data
        
        print(f"Αναλύεται: {title}...")

        prompt = f"""
        Είσαι βοηθός αρχειοθέτησης του Τμήματος Πληροφορικής & Τηλεπικοινωνιών.
        Διάβασε το παρακάτω κείμενο ιστοσελίδας και εξήγαγε την κατηγορία, 3-5 keywords, και μια μικρή περίληψη.

        ΤΙΤΛΟΣ: {title}
        ΚΕΙΜΕΝΟ:
        {text}"""

        prompt_tmpl = PromptTemplate(prompt)

        try:
            response: PageEnrichment = await llm.astructured_predict(
                PageEnrichment,
                prompt=prompt_tmpl
            )

            data["content_category"] = response.content_category
            data["keywords"] = response.keywords
            data["summary"] = response.summary
        except Exception as e:
            print(f"Σφάλμα LLM στο '{title}': {e}")
            data["content_category"] = "Γενικές Πληροφορίες"
            data["summary"] = "Αποτυχία παραγωγής περίληψης."

        await asyncio.sleep(1.5)

        return data
async def main():
    if not INPUT_FILE.exists():
        print(f"Το αρχείο {INPUT_FILE} δεν βρέθηκε!")
        return

    raw_pages = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_pages.append(json.loads(line))
                
    print(f"Φορτώθηκαν {len(raw_pages)} σελίδες για εμπλουτισμό.")

    semaphore = asyncio.Semaphore(CONCURRENT_LLM_REQUESTS)
    tasks = [enrich_single_page(page, semaphore) for page in raw_pages]
    
    enriched_pages = await asyncio.gather(*tasks)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for page in enriched_pages:
            f.write(json.dumps(page, ensure_ascii=False) + '\n')
            
    print(f"\nΤέλεια! Τα εμπλουτισμένα δεδομένα αποθηκεύτηκαν στο {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())