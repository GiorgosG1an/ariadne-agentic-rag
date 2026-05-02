import json
import os
import re
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from google.genai import Client, types
from pydantic import ValidationError

from pipelines.core.logger import get_logger
from pipelines.schemas.documents import WeeklyScheduleModel, ClassScheduleEvent

load_dotenv()
logger = get_logger(__name__)

class ScheduleExtractor:
    def __init__(self, pdf_file_path: str, output_file: str, model_name: str = "gemini-2.5-flash"):
        self.pdf_file_path = Path(pdf_file_path)
        self.output_file = Path(output_file)
        self.output_file_cleaned = Path(str(output_file).replace(".jsonl", "_cleaned.jsonl"))
        self.model_name = model_name
        self.client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def extract_from_pdf(self):
        if not self.pdf_file_path.exists():
            logger.error(f"File {self.pdf_file_path} not found!")
            return

        logger.info("Uploading file to Google's FileAPI")
        uploaded_file = self.client.files.upload(file=str(self.pdf_file_path))

        prompt = """
        Είσαι ένας αλγόριθμος εξαγωγής δεδομένων υψηλής ακρίβειας. Σου δίνω το επίσημο Ωρολόγιο Πρόγραμμα ενός Πανεπιστημιακού Τμήματος.
        Το έγγραφο αποτελείται από πίνακες. Κάθε σελίδα (ή τμήμα) έχει τίτλο την Ημέρα (π.χ. 'Δευτέρα').
        Αριστερά είναι οι Ώρες. Πάνω είναι τα Έτη Σπουδών (1ο Έτος, 2ο Έτος, 3ο Έτος, 4ο Έτος).

        ΟΔΗΓΙΕΣ ΕΞΑΓΩΓΗΣ:
        1. Διάβασε προσεκτικά ΚΑΘΕ κελί του πίνακα.
        2. Αν ένα κελί εκτείνεται (merged) κάθετα σε πολλές ώρες (π.χ. από 10:00-11:00 έως 12:00-13:00), υπολόγισε σωστά το `start_time` (10:00) και το `end_time` (13:00).
        3. Αν μέσα στο ίδιο κελί υπάρχει το σύμβολο '&' (π.χ. δύο διαφορετικά μαθήματα ή εργαστήρια την ίδια ώρα), ΠΡΕΠΕΙ να φτιάξεις ΔΥΟ ΞΕΧΩΡΙΣΤΑ objects (events) στη λίστα σου!
        4. Βρες τον τύπο του μαθήματος. Αν λέει "Εργαστήριο" βάλε "Εργαστήριο", αν λέει "Φροντιστήριο" βάλε "Φροντιστήριο". Αλλιώς βάλε "Θεωρία".
        5. ΠΡΟΣΟΧΗ ΣΤΙΣ ΑΙΘΟΥΣΕΣ (CRITICAL): 
        - Δεν υπάρχουν αίθουσες που ξεκινάνε με το ψηφίο '1' (π.χ. 115, 114, 19). 
        - Αν δεις κάτι που μοιάζει με '115', είναι το γράμμα 'Ι' (Γιώτα) και ο αριθμός '15', δηλαδή 'Ι15'.
        - Αν δεις '19', είναι το 'Ι19'. 
        - Μετάτρεψε ΟΛΑ τα αρχικά '1' σε 'Ι' στις αίθουσες του Κάτω Κτιρίου. 
        - Σωστά παραδείγματα: 'Κάτω κτίριο - Ι15', 'Κάτω κτίριο - Ι14', 'Κάτω κτίριο - Ι19', 'Πάνω κτίριο - Β1'.
        6. Μην ξεχάσεις κανένα μάθημα! Κάνε εξαγωγή και για τις 5 εργάσιμες ημέρες.
        """

        logger.info("Gemini is parsing the tables (this will take a few minutes)...")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WeeklyScheduleModel,
                    temperature=0.0,
                ),
            )

            logger.info("Saving extracted data...")
            schedule_data = response.parsed

            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w", encoding="utf-8") as f:
                for event in schedule_data.events:
                    try:
                        validated_event = ClassScheduleEvent(**event.model_dump())
                        f.write(validated_event.model_dump_json() + "\n")
                    except ValidationError as ve:
                        logger.warning(f"Skipping invalid event due to schema error: {ve}")

            logger.info(f"Success! Extracted {len(schedule_data.events)} events to '{self.output_file}'.")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
        finally:
            try:
                self.client.files.delete(name=uploaded_file.name)
            except:
                pass

    @staticmethod
    def fix_classroom_name(name: str) -> str:
        name = re.sub(r"\b1(?=\d)", "Ι", name)
        name = name.replace("–", "-").replace("  ", " ")
        return name

    def merge_consecutive_classes(self):
        if not self.output_file.exists():
            logger.error(f"Cannot merge. File {self.output_file} not found.")
            return

        events = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line.strip()))

        grouped_events = {}
        for event in events:
            event["classroom"] = self.fix_classroom_name(event["classroom"])
            key = (
                event["day"].strip(),
                event["year"].strip(),
                event["course_name"].strip(),
                event["course_type"].strip(),
            )
            if key not in grouped_events:
                grouped_events[key] = []
            grouped_events[key].append(event)

        merged_events = []
        for key, group in grouped_events.items():
            group.sort(key=lambda x: x["start_time"])
            current_event = group[0]
            for next_event in group[1:]:
                if current_event["end_time"] == next_event["start_time"]:
                    current_event["end_time"] = next_event["end_time"]
                else:
                    merged_events.append(current_event)
                    current_event = next_event
            merged_events.append(current_event)

        days_order = {"Δευτέρα": 1, "Τρίτη": 2, "Τετάρτη": 3, "Πέμπτη": 4, "Παρασκευή": 5}
        merged_events.sort(key=lambda x: (days_order.get(x["day"], 99), x["start_time"]))

        with open(self.output_file_cleaned, "w", encoding="utf-8") as f:
            for event in merged_events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        logger.info(f"Merging complete! From {len(events)} records -> {len(merged_events)} cleaned records.")

    def run(self):
        self.extract_from_pdf()
        self.merge_consecutive_classes()

if __name__ == "__main__":
    PDF_FILE_PATH = "data/schedule_data/dit_program_spring_2025-26_v2.2.pdf"
    OUTPUT_FILE = "data/schedule_data/dit_program_spring_2025-26.jsonl"
    extractor = ScheduleExtractor(pdf_file_path=PDF_FILE_PATH, output_file=OUTPUT_FILE)
    extractor.run()
