from pydantic import BaseModel, Field
from typing import List, Optional

class AnnouncementModel(BaseModel):
    url: str
    title: str
    content_category: str = "Ανακοινώσεις"
    last_modified: str
    cleaned_content: str

class WebsiteModel(BaseModel):
    url: str
    title: str
    content_category: str = ""
    last_modified: str
    keywords: List[str] = Field(default_factory=list)
    summary: str = ""
    language: str = "el"
    cleaned_content: str

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

class WeeklyScheduleModel(BaseModel):
    events: List[ClassScheduleEvent] = Field(
        description="Η πλήρης λίστα με ΟΛΑ τα μαθήματα της εβδομάδας."
    )
