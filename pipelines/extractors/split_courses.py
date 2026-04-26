"""Split courses into seperate markdown files, with frontmatter metadata"""

import json
import re
from pathlib import Path
import yaml

# Load your data once
with open("instructors_map.json", "r", encoding="utf-8") as f:
    courses_data = json.load(f)


def get_instructors(course_name):
    instructors = courses_data.get(course_name)
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


def extract_section(pattern, text, default=""):
    """Helper to extract a specific chunk of text based on a regex pattern."""
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else default


def extract_weekly_schedule(block):
    # 1. Εύρεση της αρχής
    start_match = re.search(r"- \*\*Εβδομάδα 1\*\*", block)
    if not start_match:
        return "-"

    content_from_start = block[start_match.start() :]

    # 2. Ορισμός τέλους
    end_markers = ["Τρόπος παράδοσης", "Χρήση τεχνολογιών"]
    end_index = len(content_from_start)
    for marker in end_markers:
        marker_pos = content_from_start.find(marker)
        if marker_pos != -1 and marker_pos < end_index:
            end_index = marker_pos

    relevant_content = content_from_start[:end_index].strip()

    # 3. Επεξεργασία γραμμών
    lines = relevant_content.split("\n")
    formatted_weeks = []

    for line in lines:
        clean_line = line.strip()

        # ΑΓΝΟΗΣΕ τις κενές γραμμές τελείως
        if not clean_line:
            continue

        # Regex για να πιάνει και το "Εβδομάδα 5-6" ή "Εβδομάδα 11-13"
        if re.match(r"^- \*\*Εβδομάδα \d+(?:-\d+)?\*\*", clean_line):
            formatted_weeks.append(clean_line)

        # Λεπτομέρειες (Τίτλος, Βιβλιογραφία κλπ)
        elif clean_line.startswith("- **"):
            formatted_weeks.append("    " + clean_line)

        # Overflow κείμενο
        else:
            formatted_weeks.append("        " + clean_line)

    return "\n".join(formatted_weeks)


def parse_and_split_courses(input_text):
    # Split by course title: ### Name [code]
    course_blocks = re.split(r"### ", input_text)

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

        # 1. Extract Metadata Variables
        def get_field(field_name):
            return extract_section(
                rf"- \*\*{field_name}\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block, "-"
            )

        ects = get_field("Μονάδες ECTS").strip()
        semester = get_field("Εξάμηνο").strip()
        pattern = r"^(\d+)\s+(\d+)$"
        match = re.match(pattern, semester)
        if match:
            primary_sem = match.group(1)
            secondary_sem = match.group(2)
            semester = f"- **Εξάμηνο**: {primary_sem} (Προσφέρεται επίσης και για μαθητές του {int(primary_sem) - 2} εξαμήνου)"
        else:
            primary_sem = semester
            semester = f"- **Εξάμηνο**: {primary_sem}"
        category = get_field("Κατηγορία").strip()
        url = get_field("URL").strip()
        prerequisites = get_field("Προαπαιτούμενα").strip()
        yaml_instructors, body_instructors = get_instructors(title)

        # 2. Extract specific Content Sections into separate variables

        # Extract Weekly Schedule (everything from the first 'Εβδομάδα' to the next major section)
        # We transform it immediately into headers
        weeks_content = extract_weekly_schedule(block).strip()

        concepts_learned = extract_section(
            r"- \*\*Μαθησιακά αποτελέσματα\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
        ).strip()

        course_contents = extract_section(
            r"- \*\*Περιεχόμενα\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
        ).strip()
        # Extract Evaluation
        evaluation_content = extract_section(
            r"- \*\*Αξιολόγηση\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
        ).strip()

        # Extract Methods
        methods_content = extract_section(
            r"- \*\*Μέθοδοι αξιολόγησης\*\*:\s*(.*?)(?=\n- \*\*|\Z)", block
        ).strip()

        # Extract Bibliography (looking for the numbered list pattern)
        bibliography_content = extract_section(
            r"- \*\*Βιβλιογραφία\*\*:\s*\n(1\..*?)(\n- \*\*|\Z)", block
        ).strip()

        # 3. Clean up Square Checkmarks for any variables that need it
        def clean_formatting(text):
            text = text.replace("Check-square", "- [x] ")
            text = text.replace("SQUARE", "- [ ] ")
            return text

        weeks_content = clean_formatting(weeks_content)
        evaluation_content = clean_formatting(evaluation_content)

        # --- File Assembly using the specific variables ---
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

        # Save to file
        filename = f"{code.replace('/', '-')}.md"
        output_dir = Path("courses")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / filename, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"Generated: {filename}")


if __name__ == "__main__":
    with open("md/dit-course-list.md", "r", encoding="utf-8") as file:
        content = file.read()

    parse_and_split_courses(content)