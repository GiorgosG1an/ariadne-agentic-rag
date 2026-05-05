import pytest
from pydantic import ValidationError
from pipelines.schemas.documents import AnnouncementModel, WebsiteModel, ClassScheduleEvent

def test_announcement_model_valid():
    data = {
        "url": "https://example.com",
        "title": "Test Title",
        "last_modified": "2025-01-01",
        "cleaned_content": "Content"
    }
    model = AnnouncementModel(**data)
    assert model.url == "https://example.com"
    assert model.content_category == "Ανακοινώσεις"

def test_announcement_model_invalid():
    with pytest.raises(ValidationError):
        AnnouncementModel(url="https://example.com", title="Title") # Missing required fields

def test_website_model_valid():
    data = {
        "url": "https://example.com",
        "title": "Test Website",
        "last_modified": "2025-01-01",
        "cleaned_content": "Some site content"
    }
    model = WebsiteModel(**data)
    assert model.language == "el"
    assert model.keywords == []

def test_schedule_event_valid():
    data = {
        "day": "Δευτέρα",
        "start_time": "09:00",
        "end_time": "11:00",
        "year": "1",
        "course_name": "Math",
        "course_type": "Θεωρία",
        "instructor": "Dr. Smith",
        "classroom": "A1"
    }
    event = ClassScheduleEvent(**data)
    assert event.day == "Δευτέρα"

def test_schedule_event_invalid():
    with pytest.raises(ValidationError):
        ClassScheduleEvent(day="Δευτέρα")
