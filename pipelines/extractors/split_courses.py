import json
import re
from pathlib import Path
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipelines.core.logger import get_logger

logger = get_logger(__name__)

class CourseSplitter:
    def __init__(self, instructors_map_path: str = "instructors_map.json", input_md: str = "md/dit-course-list.md", output_dir: str = "data/courses"):
        self.instructors_map_path = Path(instructors_map_path)
        self.input_md = Path(input_md)
        self.output_dir = Path(output_dir)
        self.courses_data = {}
        self.load_instructors()

    def load_instructors(self):
        if self.instructors_map_path.exists():
            with open(self.instructors_map_path, "r", encoding="utf-8") as f:
                self.courses_data = json.load(f)
        else:
            logger.warning(f"Instructors map not found at {self.instructors_map_path}")

    def get_instructors(self, course_name):
        instructors = self.courses_data.get(course_name)
        if instructors == "-":
            instructors = "N/A"
        if instructors and isinstance(instructors, list):
            yaml_block = yaml.dump(
                instructors, allow_unicode=True, default_flow_style=False
            ).strip()
            yaml_instructors = "\n" + "\n".join(
                f"  {line}" for line in yaml_block.splitlines()
            )

            body_instructors = (
                "\n".join(f"    * {i}" for i in instructors) if instructors else "N/A"
            )
            return yaml_instructors, body_instructors

        return "[]", "N/A"

    @staticmethod
    def extract_section(pattern, text, default=""):
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else default

    @staticmethod
    def extract_weekly_schedule(block):
        start_match = re.search(r"- \*\*Εβδομάδα 1\*\*", block)
        if not start_match:
            return "-"

        content_from_start = block[start_match.start() :]

        end_markers = ["Τρόπος παράδοσης", "Χρήση τεχνολογιών"]
        end_index = len(content_from_start)
        for marker in end_markers:
            marker_pos = content_from_start.find(marker)
            if marker_pos != -1 and marker_pos < end_index:
                end_index = marker_pos

        relevant_content = content_from_start[:end_index].strip()

        lines = relevant_content.split("\n")
        formatted_weeks = []

        for line in lines:
            clean_line = line.strip()

            if not clean_line:
                continue

            if re.match(r"^- \*\*Εβδομάδα \d+(?:-\d+)?\*\*", clean_line):
                formatted_weeks.append(clean_line)
            elif clean_line.startswith("- **"):
                formatted_weeks.append("    " + clean_line)
            else:
                formatted_weeks.append("        " + clean_line)

        return "\n".join(formatted_weeks)

    def parse_and_split_courses(self):
        if not self.input_md.exists():
            logger.error(f"Input file {self.input_md} does not exist.")
            return

        with open(self.input_md, "r", encoding="utf-8") as file:
            input_text = file.read()

        course_blocks = re.split(r"### ", input_text)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for block in course_blocks:
            if not block.strip():
                continue

            lines = block.strip().split("\n")
            header_line = lines[0]
            match = re.search(r"(.*?) \[(.*?)\]", header_line)

            if not match:
                continue

            title = match.group(1).strip()
            code = match.group(2).strip()

            def get_field(field_name):
                return self.extract_section(
                    rf"- \*\*{field_name}\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block, "-"
                )

            ects = get_field("Μονάδες ECTS").strip()
            semester = get_field("Εξάμηνο").strip()
            pattern = r"^(\d+)\s+(\d+)$"
            match_sem = re.match(pattern, semester)
            if match_sem:
                primary_sem = match_sem.group(1)
                semester = f"- **Εξάμηνο**: {primary_sem} (Προσφέρεται επίσης και για μαθητές του {int(primary_sem) - 2} εξαμήνου)"
            else:
                primary_sem = semester
                semester = f"- **Εξάμηνο**: {primary_sem}"
            category = get_field("Κατηγορία").strip()
            url = get_field("URL").strip()
            prerequisites = get_field("Προαπαιτούμενα").strip()
            yaml_instructors, body_instructors = self.get_instructors(title)

            weeks_content = self.extract_weekly_schedule(block).strip()

            concepts_learned = self.extract_section(
                r"- \*\*Μαθησιακά αποτελέσματα\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
            ).strip()

            course_contents = self.extract_section(
                r"- \*\*Περιεχόμενα\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
            ).strip()

            evaluation_content = self.extract_section(
                r"- \*\*Αξιολόγηση\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
            ).strip()

            methods_content = self.extract_section(
                r"- \*\*Μέθοδοι αξιολόγησης\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
            ).strip()

            bibliography_content = self.extract_section(
                r"- \*\*Βιβλιογραφία\*\*:\s*\n(1\..*?)(\n- \*\*|\Z)", block
            ).strip()

            def clean_formatting(text):
                text = text.replace("Check-square", "- [x] ")
                text = text.replace("SQUARE", "- [ ] ")
                return text

            weeks_content = clean_formatting(weeks_content)
            evaluation_content = clean_formatting(evaluation_content)

            file_content = f"""---
course_title: "{title}"
course_code: "{code}"
ects: {ects}
semester: {primary_sem}
category: "{category}"
instructors: {yaml_instructors}
url: "{url}"
---

# {title}

## Βασικές Πληροφορίες
- **Όνομα Μαθήματος**: {title}
- **Κωδικός**: {code}
- **Διδάσκοντες**: 
{body_instructors}
- **Τύπος Μαθήματος**: {category}
- **ECTS**: {ects}
{semester}
- **Προαπαιτούμενα**: {prerequisites}
- **E-class url**: {url}

- **Μαθησιακά Αποτελέσματα**:
    {concepts_learned}

- **Περιεχόμενα**: {course_contents}

- **Αξιολόγηση**: {evaluation_content}

- **Μέθοδοι Αξιολόγησης**:
    {methods_content}

## Γενική Βιβλιογραφία
{bibliography_content}

## Εβδομαδιαίο Πρόγραμμα
{weeks_content}
"""
            filename = f"{code.replace('/', '-')}.md"
            with open(self.output_dir / filename, "w", encoding="utf-8") as f:
                f.write(file_content)
            count += 1
        logger.info(f"Generated {count} course markdown files in {self.output_dir}.")

if __name__ == "__main__":
    splitter = CourseSplitter()
    splitter.parse_and_split_courses()