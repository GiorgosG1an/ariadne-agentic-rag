import pytest
import aiohttp
from unittest.mock import MagicMock, patch
from pipelines.scrapers.announcements_scraper import AnnouncementScraper

@pytest.fixture
def fake_html():
    return """
    <html>
        <body>
            <main role="main">
                <h1 class="page-header">Test Announcement | Category</h1>
                <time datetime="2025-01-01T12:00:00Z">2025-01-01</time>
                <p>This is a test announcement.</p>
                <a href="/announcement/1">Link 1</a>
                <a href="?page=1">Pagination</a>
            </main>
        </body>
    </html>
    """

@pytest.mark.asyncio
@patch.object(AnnouncementScraper, 'fetch_page')
async def test_announcement_scraper_fetch_and_parse(mock_fetch, fake_html):
    mock_fetch.return_value = fake_html
    scraper = AnnouncementScraper()
    
    session_mock = MagicMock(spec=aiohttp.ClientSession)
    import asyncio
    semaphore = asyncio.Semaphore(1)
    
    result = await scraper.fetch_and_parse_announcement("https://dit.uop.gr/test", session_mock, semaphore)
    
    assert result is not None
    assert result.title == "Test Announcement"
    assert result.last_modified == "2025-01-01"
    assert "This is a test announcement." in result.cleaned_content

@pytest.mark.asyncio
@patch.object(AnnouncementScraper, 'fetch_page')
async def test_announcement_scraper_discover_links(mock_fetch, fake_html):
    mock_fetch.side_effect = [fake_html, Exception("End")]
    scraper = AnnouncementScraper()
    
    session_mock = MagicMock(spec=aiohttp.ClientSession)
    
    links = await scraper.discover_announcement_links(session_mock, "https://dit.uop.gr/list")
    
    assert len(links) == 1
    assert "https://dit.uop.gr/announcement/1" in links
