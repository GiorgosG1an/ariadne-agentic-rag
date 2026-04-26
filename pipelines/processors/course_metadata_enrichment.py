import os
import frontmatter

from dotenv import load_dotenv
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class CourseMetadataEnrichment(BaseModel):
    scientific_domains: List[str] = Field(description="Broad scientific fields in greek (e.g., 'Artificial Intelligence', 'Data Science').")
    keywords: List[str] = Field(description="Technical terms and technologies. Use English for industry-standard terms (e.g., 'Transformer', 'SQL'), Greek for others.")
    skills_learned: List[str] = Field(description="5 - 10 skills that are mentioned or learned by passing the course.")
    summary: str = Field(description="A 2-3 sentence semantic summary in GREEK, but keep technical terms in English where appropriate.")
    career_paths: List[str] = Field(description="Job titles or roles where this course's knowledge is applicable.")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash-lite"

def enrich_course(file_path: Path):
    post = frontmatter.load(file_path)
    content = post.content

    prompt = f"""
    You are an expert Data Engineer, with a task to enrich the provided content with metadata, that will boost the Retrieval in a RAG system\n 
    Analyze this syllabus for the course: {post.get('course_title')}
    
    Strict Language Rules:
    1. scientific_domains & career_paths: MUST be in English.
    2. summary: Write in Greek, but keep technical terminology in English (e.g. use 'Vector Space Model' instead of 'Διανυσματικό Μοντέλο Χώρου').
    3. keywords: Mix of Greek and English (prioritize English for tools/tech).
    4. skills: Prefer greek language, but keep english too

    Content to analyze:
    {content}
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=CourseMetadataEnrichment.model_json_schema(),
            temperature=0.2,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,
            )
        )
    )
    clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    metadata = CourseMetadataEnrichment.model_validate_json(clean_json)

    post['scientific_domains'] = metadata.scientific_domains
    post['keywords'] = metadata.keywords
    post['summary'] = metadata.summary
    post['skills_learned'] = metadata.skills_learned
    post['career_paths'] = metadata.career_paths

    with open(file_path, 'wb') as f:
        frontmatter.dump(post, f)
    print(f"Enriched: {file_path.name}")


def main():
    folder_path = Path("courses")
    files_not_enriched = []
    for md_file in folder_path.glob("*.md"):
        try:
            enrich_course(md_file)
        except Exception as e:
            print(f"Error processing file: {md_file.name}: {e}")
            files_not_enriched.append(md_file)

    # fallback for error code 503 UNAVAILABLE
    for file in files_not_enriched:
        try:
            enrich_course(file)
            files_not_enriched.remove(file)
        except Exception as e:
            print(f"Error processing file: {md_file.name}: {e}")

    print(files_not_enriched)

if __name__ == '__main__':
    main()