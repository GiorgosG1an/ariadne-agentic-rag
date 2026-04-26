import json
import os
import re

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google.genai import Client, types
from pydantic import BaseModel, Field

load_dotenv()

PDF_FILE_PATH = "schedule_data/dit_program_spring_2025-26_v2.2.pdf"
OUTPUT_FILE = Path("schedule_data/dit_program_spring_2025-26.jsonl")
MODEL_NAME = "gemini-2.5-flash"


# Pydantic Schema for structured output
class ClassScheduleEvent(BaseModel):
    day: str = Field(description="Η ημέρα της εβδομάδας (π.χ. 'Δευτέρα', 'Τρίτη').")
    start_time: str = Field(description="Η ώρα έναρξης σε μορφή HH:MM (π.χ. '09:00').")
    end_time: str = Field(
        description="Η ώρα λήξης σε μορφή HH:MM (π.χ. '11:00'). Αν το κελί πιάνει πολλές ώρες, βάλε τη συνολική λήξη."
    )
    year: str = Field(description="Το έτος σπουδών ως ψηφίο: '1', '2', '3' ή '4'.")
    course_name: str = Field(description="Ο τίτλος του μαθήματος.")
    course_type: str = Field(
        description="Ο τύπος του μαθήματος: 'Θεωρία', 'Εργαστήριο', 'Φροντιστήριο', ή 'Πρακτική Άσκηση'. Αν δεν αναφέρεται κάτι, θεώρησε 'Θεωρία'."
    )
    instructor: str = Field(description="Το όνομα του διδάσκοντα ή των διδασκόντων.")
    classroom: str = Field(
        description="Το κτήριο και η αίθουσα(π.χ. 'Πάνω κτίριο - B1', 'Κάτω Κτήριο Ι14')."
    )


class WeeklySchedule(BaseModel):
    events: List[ClassScheduleEvent] = Field(
        description="Η πλήρης λίστα με ΟΛΑ τα μαθήματα της εβδομάδας."
    )


def extract_schedule_from_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Το αρχείο {pdf_path} δεν βρέθηκε!")
        return

    client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print("Uploading file to Google's FileAPI")
    uploaded_file = client.files.upload(file=pdf_path)

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
    5. Μην ξεχάσεις κανένα μάθημα! Κάνε εξαγωγή και για τις 5 εργάσιμες ημέρες.
"""

    print(
        "Το Gemini αναλύει τον πίνακα και εξάγει τα JSON (Αυτό θα πάρει μερικά λεπτά)..."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WeeklySchedule,
                temperature=0.0,
            ),
        )

        print("Αποθήκευση Δεδομένων...")
        schedule_data = response.parsed

        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for event in schedule_data.events:

                f.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")

        print(
            f"Επιτυχία! Εξήχθησαν {len(schedule_data.events)} μαθήματα/εργαστήρια στο '{OUTPUT_FILE}'."
        )

        client.files.delete(name=uploaded_file.name)

    except Exception as e:
        print(f"Σφάλμα κατά την εξαγωγή: {e}")


INPUT_FILE_MERGE = Path("schedule_data/dit_program_spring_2025-26.jsonl")
OUTPUT_FILE_MERGE = Path("schedule_data/dit_program_cleaned.jsonl")


def fix_classroom_name(name: str) -> str:
    # Αν βρει " 1" ακολουθούμενο από ψηφία (π.χ. 115, 19) το κάνει " Ι"
    name = re.sub(r"\b1(?=\d)", "Ι", name)
    name = name.replace("–", "-").replace("  ", " ")
    return name


def merge_consecutive_classes(input_path: Path, output_path: Path):

    events = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line.strip()))

    grouped_events = {}

    for event in events:
        event["classroom"] = fix_classroom_name(event["classroom"])
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

    # (Ημέρα -> Ώρα)
    days_order = {"Δευτέρα": 1, "Τρίτη": 2, "Τετάρτη": 3, "Πέμπτη": 4, "Παρασκευή": 5}
    merged_events.sort(key=lambda x: (days_order.get(x["day"], 99), x["start_time"]))

    with open(output_path, "w", encoding="utf-8") as f:
        for event in merged_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(
        f"Η συνένωση ολοκληρώθηκε! Από {len(events)} εγγραφές -> {len(merged_events)} καθαρές εγγραφές."
    )


if __name__ == "__main__":
    extract_schedule_from_pdf(PDF_FILE_PATH)
    merge_consecutive_classes(INPUT_FILE_MERGE, OUTPUT_FILE_MERGE)
