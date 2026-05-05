import json
import logging
import re

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import pymupdf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipelines.core.logger import get_logger

logger = get_logger(__name__)

class TableType(Enum):
    COURSES = "COURSES"
    STAFF = "STAFF"
    EDIP = "EDIP"
    REST = "REST"
    SECRETARY = "SECRETARY"
    ASSIGNED = "ASSIGNED"
    GENERIC = "GENERIC"
    UNKNOWN = "UNKNOWN"

@dataclass
class TableSchema:
    type_enum: TableType
    headers: List[str]
    keywords: List[str]
    expected_cols: int


class DITPDFParser:
    """
    DIT PDF Parser:

    A modular parser designed to extract text and tables from department PDFs (Latex)
    and format them into Markdown optimized for RAG document ingestion.

    Handles:
    - Heading Levels
    - Table Parsing
    """

    def __init__(self, margin_top: int = 50, margin_bottom: int = 50):
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.schemas = self._initialize_schemas()

    def _initialize_schemas(self) -> List[TableSchema]:
        """Encapsulates configuration to keep the global namespace clean."""
        return [
            TableSchema(
                type_enum=TableType.STAFF,
                headers=["Ονοματεπώνυμο", "Βαθμίδα", "Τηλέφωνο", "Email"],
                keywords=["βαθμίδα"],
                expected_cols=4,
            ),
            TableSchema(
                type_enum=TableType.EDIP,
                headers=["Ονοματεπώνυμο", "Τηλέφωνο", "Email"],
                keywords=["ονοματεπώνυμο", "email", "τηλέφωνο"],
                expected_cols=3,
            ),
            TableSchema(
                type_enum=TableType.REST,
                headers=["Ονοματεπώνυμο", "Ιδιότητα", "Email"],
                keywords=["ονοματεπώνυμο", "email", "ιδιότητα"],
                expected_cols=3,
            ),
            TableSchema(
                type_enum=TableType.SECRETARY,
                headers=["Ονοματεπώνυμο", "Θέση", "Τηλέφωνο", "Email"],
                keywords=["ονοματεπώνυμο", "θέση"],
                expected_cols=4,
            ),
            TableSchema(
                type_enum=TableType.ASSIGNED,
                headers=["Τίτλος Μαθήματος", "Κατηγορία", "ECTS", "Διδάσκων"],
                keywords=["τίτλος μαθήματος", "κατηγορία", "ects", "διδάσκων"],
                expected_cols=4,
            ),
            TableSchema(
                type_enum=TableType.COURSES,
                headers=[
                    "Τίτλος Μαθήματος",
                    "Εξάμηνο",
                    "Μονάδες ECTS",
                    "Ώρες (Θεωρία)",
                    "Ώρες (Εργαστήριο)",
                    "Ώρες (Φροντιστήριο)",
                    "Κατηγορία",
                ],
                keywords=["ects", "μονάδες"],
                expected_cols=7,
            ),
        ]

    @staticmethod
    def clean_text(text: str) -> str:
        """Standardizes text cleaning."""
        if not text:
            return ""
        text = text.replace('\n', ' ')
        return re.sub(r"\s+", " ", text).strip()

    def _detect_schema(
        self, table: pymupdf.table.Table
    ) -> Tuple[List[str], TableType, int]:
        """Dynamically matches table headers against registered schemas."""
        try:
            header_row = table.extract()[0]
            header_str = " ".join([str(x).lower() for x in header_row if x])
        except (IndexError, AttributeError):
            return [], TableType.UNKNOWN, 0

        # Find matching schema based on keywords
        for schema in self.schemas:
            if all(keyword in header_str for keyword in schema.keywords):
                return schema.headers, schema.type_enum, schema.expected_cols

        # Fallback to PDF's native headers
        headers = table.header.names if table.header and table.header.names else []
        return headers, TableType.GENERIC, len(headers)

    def _row_to_rag_card(self, headers: List[str], row_data: List[str]) -> str:
        """Transforms structured row data into a RAG-friendly Markdown card."""
        entity_name = row_data[0]
        if not entity_name:
            return ""

        card_md = f"\n#### {entity_name}\n"
        for i in range(1, len(row_data)):
            if i < len(headers):
                val = row_data[i]
                if val:
                    card_md += f"- **{headers[i]}**: {val}\n"
        return card_md

    def _process_table(self, page: pymupdf.Page, table: pymupdf.table.Table) -> str:
        """Processes a single table with schema detection and row normalization."""
        headers, table_type, expected_cols = self._detect_schema(table)
        if not headers or table_type == TableType.UNKNOWN:
            return ""

        md_output = ""
        for row in table.rows:
            row_data = [
                self.clean_text(page.get_text("text", clip=cell)) for cell in row.cells
            ]

            # 1. Skip Empty Rows
            if not any(row_data):
                continue

            # 2. Skip Repeated Header Rows inside the table body
            row_str = " ".join(row_data).lower()
            if "ονοματεπώνυμο" in row_str or "τίτλος μαθήματος" in row_str:
                continue
            if "θεωρία" in row_str and table_type == TableType.COURSES:
                continue

            # 3. Universal Data Normalization (Padding/Truncating)
            if expected_cols > 0:
                while len(row_data) < expected_cols:
                    row_data.append("")
                row_data = row_data[:expected_cols]

            md_output += self._row_to_rag_card(headers, row_data)

        return md_output

    def _process_text_block(
        self, block: Dict
    ) -> Optional[Dict[str, Union[float, str]]]:
        """Extracts and formats text blocks based on visual hierarchy (font size)."""
        if "lines" not in block:
            return None

        try:
            first_span = block["lines"][0]["spans"][0]
            font_size = round(first_span["size"], 1)

            raw_text = " ".join(
                [span["text"] for line in block["lines"] for span in line["spans"]]
            )

            text_content = self.clean_text(raw_text)

            if not text_content:
                return None

            # Map font sizes to Markdown Headings
            heading = text_content
            if font_size >= 20:
                heading = f"\n# {text_content}\n"
            elif font_size >= 13:
                heading = f"\n## {text_content}\n"
            elif font_size >= 11.8:
                heading = f"\n### {text_content}\n"
            elif font_size <= 8:
                return None  # Filter out tiny text (watermarks/footers)

            return {"y": block["bbox"][1], "text": heading}

        except (IndexError, KeyError) as e:
            logger.warning(f"Warning while processing text block: {e}\n")
            return None

    @staticmethod
    def clean_md_output(text: str) -> str:
        """
        Perform final post-processing on the assembled Markdown output
        to remove PDF artifacts and formatting noise.
        """
        if not text:
            return ""

        # Matches: #### [Text] [Number] -> Results in: #### [Text] that appears in course names that had superscript in the pdf
        text = re.sub(r"^(####\s+.+?)\s+\d+\s*$", r"\1", text, flags=re.MULTILINE)
        # 1. Fix Hyphenation (Cross-line and space-separated)
        # Using [^\W\d_] to match ONLY letters (Greek, English, etc.) and NOT numbers/symbols.
        # This prevents turning "1- 2" into "12", but fixes "Πανεπι- στήμιο" -> "Πανεπιστήμιο"
        text = re.sub(r"([^\W\d_]+)-\s*\n\s*([^\W\d_]+)", r"\1\2", text)
        text = re.sub(r"([^\W\d_]+)-\s+([^\W\d_]+)", r"\1\2", text)

        # 2. Remove TOC Dotted Lines (e.g., "Περιεχόμενα . . . . . 1")
        text = re.sub(r"^.*\. \. \..*$", "", text, flags=re.MULTILINE)

        # 3. Remove Stray Numbers (usually orphaned page numbers)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # 4. Remove Single Stray Letters (English AND Greek)
        # Matches a single letter, accounting for Greek vowels with accents
        text = re.sub(
            r"^\s*[a-zA-Zα-ωΑ-ΩάέήίόύώΆΈΉΊΌΎΏϊϋΪΫ]\s*$", "", text, flags=re.MULTILINE
        )

        # 5. Clean up excessive whitespace created by the deletions above
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def format_abbrevations(text: str) -> str:
        """
        Replace Greek abbreviations with their full forms in the given text.

        Converts abbreviated Greek terms commonly used in academic contexts:
        - (Θ) → Θεωρία
        - (Ε) → Εργαστήριο
        - (Φ) → Φροντιστήριο

        Example:
        ```python
        text = format_abbrevations("Διδάσκων (Θ, Ε)")
        print(text)
        # Output: "Διδάσκων (Θεωρία, Εργαστήριο)"
        ```
        """

        replacements = {
            r"\(Θ, Φ\)": "(Θεωρία, Φροντιστήριο)",
            r"\(Θ, Ε\)": "(Θεωρία, Εργαστήριο)",
            r"\(Θ\)": "(Θεωρία)",
            r"\(Φ\)": "(Φροντιστήριο)",
            r"\(Ε\)": "(Εργαστήριο)",
            r"\(Θ, Ε, Φ\)" : "(Θεωρία, Εργαστήριο, Φροντιστήριο)",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.UNICODE)
        
        return text

    @staticmethod
    def structural_cleanup(text: str) -> str:
        """
        Applies Markdown formatting to specific domain keywords.
        Applies structure to the course's descriptions safely.
        """
        keywords = [
            "Κατηγορία",
            "Μονάδες ECTS",
            "Εξάμηνο",
            "Προαπαιτούμενα",
            "Τύπος μαθήματος",
            "Γλώσσα διδασκαλίας",
            "Προσφέρεται σε φοιτητές Erasmus",
            "URL",
            "Διδακτικές δραστηριότητες",
            "Οργάνωση διδασκαλίας",
            "Μαθησιακά αποτελέσματα",
            "Μέθοδοι αξιολόγησης",
            "Γενικές ικανότητες που καλλιεργεί το μάθημα",
            "Περιεχόμενα",
            "Αξιολόγηση",
            "Βιβλιογραφία",
        ]

        # 1. Create a pattern that captures the keywords
        pattern = r"(" + "|".join(keywords) + r"):"

        # 2. Split the text.
        parts = re.split(pattern, text)

        # If no keywords are found, return the text untouched
        if len(parts) == 1:
            return text

        # parts[0] is everything BEFORE the first keyword. We keep it completely untouched.
        cleaned_parts = [parts[0]]

        # 3. Iterate through the split parts in pairs (keyword + content)
        # parts list structure: [pre_text, keyword1, content1, keyword2, content2...]
        for i in range(1, len(parts), 2):
            keyword = parts[i]
            content = parts[i + 1]

            if keyword == ("Μέθοδοι αξιολόγησης" or "Γενικές ικανότητες που καλλιεργεί το μάθημα"):
                # Extract text following 'Check-square' and ignore 'SQUARE'
                checked_items = re.findall(r"Check-square\s+([^SQUARE|Check\-square]+)", content)

                if checked_items:
                    formatted_list = "".join([f"\n    - {item.strip()}" for item in checked_items])
                    content = formatted_list
                else:
                    content = content.strip()
            else:

                # Apply the sub-bullet logic ONLY to the text belonging to these keywords.
                content = content.replace("• ", "\n    - ")

            # Reconstruct the string with the bold Markdown format
            cleaned_parts.append(f"\n- **{keyword}**:{content}")

        # 4. Rejoin the isolated chunks back into a single string
        cleaned_text = "".join(cleaned_parts)

        # 5. Cleanup redundant newlines
        cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)

        return cleaned_text

    @staticmethod
    def format_learning_outcomes(text: str) -> str:
        # 1. Ορισμός των κατηγοριών που πρέπει να γίνουν Heading 4
        categories = [
            "Γνώση και κατανόηση",
            "Εφαρμογή γνώσης και κατανόησης",
            "Κρίση",
            "Επικοινωνία",
        ]

        for cat in categories:
            # Μετατροπή της λέξης σε Heading 4 αν βρίσκεται στην αρχή γραμμής ή μετά από κενό
            text = re.sub(f"^{cat}", f"#### {cat}", text, flags=re.MULTILINE)

        # 2. Εντοπισμός του μοτίβου ΜΑ1.1, ΜΑ2.1 κλπ.
        # Το pattern ψάχνει για "ΜΑ", ακολουθούμενο από ψηφία, τελεία, ψηφία και τελεία/κενό
        # Παράδειγμα: "ΜΑ1.1. " -> "\n- **ΜΑ1.1.**: "
        text = re.sub(r"(ΜΑ\d+\.\d+\.?)", r"\n- **\1**", text)

        # 3. Καθαρισμός πολλαπλών κενών γραμμών
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def format_labs_section(text: str) -> str:
        """Create Heading 3 to the labs names and format the members"""
        text = re.sub(r'^(Εργαστήριο\s+[\w\s]+)$', r'### \1', text, flags=re.MULTILINE)
        text = re.sub(r'Μέλη:\s*(.*)', r'- **Μέλη**: \1', text)

        return text
    
    @staticmethod
    def format_weekly_schedule(text: str) -> str:
        text = re.sub(r'^####\s+(\d+(?:-\d+)?)\s*$', r'- **Εβδομάδα \1**', text, flags=re.MULTILINE)
        text = text.replace('#### εβδ.', '#### Εβδομαδιαίο Πρόγραμμα')
        text = re.sub(r'\n\s*\n', '\n', text)
        pattern = (
            r'#### Εβδομαδιαίο Πρόγραμμα\s*'
            r'-\s*\*\*Τίτλος ενότητας\*\*:\s*Τίτλος ενότητας\s*'
            r'-\s*\*\*Βιβλιογραφία\*\*:\s*Βιβλιογραφία\s*'
            r'-\s*\*\*Σύνδεσμος παρουσίασης\*\*:\s*Σύνδεσμος παρουσίασης'
        )

        text = re.sub(pattern, '', text)

        return text.strip()
        
    @staticmethod
    def get_instructors_map(text: str) -> Dict[str, List[str]]:
        """Return a dictionary with course name as key and a list of instructors as value"""
        mapping = {}
        appendix_match = re.search(r'# Α Αναθέσεις διδασκόντων.*', text, re.DOTALL)
        if not appendix_match:
            logger.warning("Warning: Appendix A not found.")
            return mapping
        appendix_text = appendix_match.group(0)
        
        course_blocks = re.finditer(r'#### (.*?)\n(.*?)(?=####|###|$)', appendix_text, re.DOTALL)

        for match in course_blocks:
            title = match.group(1).strip()
            body = match.group(2)
            
            # find instructor
            inst_match = re.search(r'- \*\*Διδάσκων\*\*: (.*)', body)
            if inst_match:
                raw_names = inst_match.group(1).strip()
                if raw_names in ['–', '-', '']:
                    continue
                # comma seperated if the comma is not inside parenthesis
                names_list = [n.strip() for n in re.split(r',\s*(?![^()]*\))', raw_names) if n]
                mapping[title] = names_list
            else:
                mapping[title] = ["N/A"]
        
        with open("instructors_map.json", 'w', encoding='utf-8') as file:
            json.dump(mapping, file, ensure_ascii=False, indent=2)

        return mapping
    


    @staticmethod
    def remove_unwanted_content(text: str, start_pattern: str, end_pattern: str, is_end_of_file: bool=False) -> str:
        """Remove content from the markdown, that is noise and does not add any value to the RAG."""
        if not is_end_of_file:
            regex = rf'({re.escape(start_pattern)}.*?)(?={re.escape(end_pattern)})'
            text = re.sub(regex, '', text, flags=re.DOTALL)
        else:
            regex = rf'{re.escape(start_pattern)}.*'
            text = re.sub(regex, '', text, flags=re.DOTALL)

        return text

    def parse(self, pdf_path: Union[str, Path]) -> str:
        """Core parsing orchestrator: processes layout, tables, and text."""
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            logger.error(f"File not found: {path_obj}")
            raise FileNotFoundError(f"File not found: {path_obj}")

        logger.info(f"Parsing PDF: {path_obj.name}")
        full_markdown = []

        try:
            with pymupdf.open(path_obj) as doc:
                for page in doc:
                    page_rect = page.rect
                    safe_area = pymupdf.Rect(
                        0,
                        self.margin_top,
                        page_rect.width,
                        page_rect.height - self.margin_bottom,
                    )

                    tables = page.find_tables(clip=safe_area, strategy="lines")
                    table_rects = [t.bbox for t in tables]
                    page_elements = []

                    # Process Tables
                    for table in tables:
                        table_md = self._process_table(page, table)
                        if table_md:
                            page_elements.append({"y": table.bbox[1], "text": table_md})

                    # Process Text Outside Tables
                    text_blocks = page.get_text(
                        "dict", clip=safe_area, flags=pymupdf.TEXT_PRESERVE_WHITESPACE
                    ).get("blocks", [])
                    for block in text_blocks:
                        b_rect = pymupdf.Rect(block.get("bbox", [0, 0, 0, 0]))

                        if any(b_rect.intersects(t_rect) for t_rect in table_rects):
                            continue

                        element = self._process_text_block(block)
                        if element:
                            page_elements.append(element)

                    # Stitch page vertically
                    page_elements.sort(key=lambda x: x["y"])
                    full_markdown.extend([str(el["text"]) for el in page_elements])
                    full_markdown.append("\n\n")

        except Exception as e:
            logger.error(f"Failed to parse {path_obj.name}: {e}")
            raise

        logger.info("Parsing complete.")
        return "\n".join(full_markdown)

    def parse_and_save(
        self, pdf_path: Union[str, Path], output_dir: Union[str, Path] = "md"
    ) -> Path:
        """Utility wrapper to parse and immediately save the output a single pdf file."""
        input_path = Path(pdf_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        md_output = self.parse(input_path)

        md_output = self.format_abbrevations(md_output)
        md_output = self.clean_md_output(md_output)
        md_output = self.structural_cleanup(md_output)
        md_output = self.format_learning_outcomes(md_output)
        md_output = self.format_labs_section(md_output)
        mapping = self.get_instructors_map(md_output)
        if "course-guide" in pdf_path.name:
            md_output = self.remove_unwanted_content(md_output, "### 3.1.3 Μαθησιακά αποτελέσματα", "### 3.1.4 ECTS")
            md_output = self.remove_unwanted_content(md_output, "## 3.6 Κατάλογος μαθημάτων", "## 3.7 Ενδεικτική κατανομή μαθημάτων σε εξάμηνα")
            md_output = self.remove_unwanted_content(md_output, "## 3.8 Προαπαιτούμενα μαθήματα", "## 3.9 Διάρκεια φοίτησης")
            md_output = self.remove_unwanted_content(md_output, "# 4 Περιγραφές μαθημάτων", "", is_end_of_file=True)
            

        if "course-list" in pdf_path.name:
            md_output = self.format_weekly_schedule(md_output)
            md_output = self.remove_unwanted_content(md_output, "# Περιεχόμενα", "# 2 Περιγραφές μαθημάτων")
            md_output = self.remove_unwanted_content(md_output, "- **Γενικές ικανότητες που καλλιεργεί το μάθημα**", "- **Περιεχόμενα**")
            md_output = self.remove_unwanted_content(md_output, "Μαθησιακά αποτελέσματα του προγράμματος σπουδών:", "- **Γενικές ικανότητες που καλλιεργεί το μάθημα**")
            md_output = self.remove_unwanted_content(md_output, "Μαθησιακά αποτελέσματα του προγράμματος σπουδών:", "- **Περιεχόμενα**")
            md_output = self.remove_unwanted_content(md_output, "# Α Μαθησιακά αποτελέσματα του προγράμματος σπουδών", "", is_end_of_file=True)

        output_file = out_dir / f"{input_path.stem}.md"
        output_file.write_text(md_output, encoding="utf-8", newline="\n")
        logger.info(f"Successfully saved Markdown to: {output_file}")

        return output_file

    def parse_directory(
        self, input_dir: Union[str, Path], output_dir: Union[str, Path] = "md"
    ):
        """Wrapper to parse an entire directory of PDFs.

        Calls the ``parse_and_save`` method for each PDF in the directory.

        Handles only pdf files that exist in the folder, all other files are skipped.
        """
        if not Path(input_dir).exists():
            logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        for pdf_file in Path(input_dir).glob("*.pdf"):
            self.parse_and_save(pdf_file, output_dir)


# --- Usage ---
if __name__ == "__main__":
    parser = DITPDFParser(margin_top=50, margin_bottom=50)

    parser.parse_directory("pdfs", "md")