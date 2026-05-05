import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Set, List
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from markdownify import ATX, markdownify as md
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from fake_useragent import UserAgent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pipelines.core.logger import get_logger
from pipelines.schemas.documents import AnnouncementModel

logger = get_logger(__name__)

class AnnouncementScraper:
    def __init__(self, base_url: str = "https://dit.uop.gr/", output_dir: str = "data/website_data"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "dit_announcements.jsonl"
        self.concurrent_requests = 5
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.announcement_urls = ["https://dit.uop.gr/all-announcements"]
        self.from_date = (datetime.now() - relativedelta(months=2)).strftime("%Y-%m-%d")
        self.ua = UserAgent()

    def get_headers(self):
        return {"User-Agent": self.ua.random}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url, timeout=15, headers=self.get_headers()) as response:
            response.raise_for_status()
            return await response.text()

    async def discover_announcement_links(self, session: aiohttp.ClientSession, base_list_url: str) -> Set[str]:
        found_links = set()
        page = 0
        logger.info(f"Looking announcements from: {self.from_date} to: {base_list_url}")

        IGNORE_PATHS = ["/en", "/user", "/taxonomy", "/rss", "/search"]

        while True:
            url = f"{base_list_url}?from={self.from_date}&page={page}"
            try:
                html = await self.fetch_page(session, url)
                soup = BeautifulSoup(html, "html.parser")
                main_content = soup.find("main", role="main")
                if not main_content:
                    break

                previous_link_count = len(found_links)

                for a_tag in main_content.find_all("a", href=True):
                    href = a_tag["href"]
                    if "?page=" in href or "?from=" in href or href.startswith("#"):
                        continue
                    if href == "/" or any(href.startswith(path) for path in IGNORE_PATHS):
                        continue
                    if href.startswith("/") and not href.endswith((".pdf", ".zip", ".jpg", ".png", ".doc", ".docx")):
                        full_url = urljoin(self.base_url, href)
                        if full_url not in self.announcement_urls:
                            found_links.add(full_url)

                if len(found_links) == previous_link_count:
                    logger.info(f"[End of list at page {page}]")
                    break

                page += 1
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error occurred at page {page}: {e}")
                break

        logger.info(f"Found {len(found_links)} unique announcements (until {self.from_date}).")
        return found_links

    async def fetch_and_parse_announcement(self, url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> AnnouncementModel | None:
        async with semaphore:
            try:
                html = await self.fetch_page(session, url)
                soup = BeautifulSoup(html, "html.parser")
                main_content = soup.find("main", role="main")
                if not main_content:
                    return None

                for a_tag in main_content.find_all("a", href=True):
                    href = a_tag["href"]
                    if href.lower().endswith(".pdf"):
                        full_pdf_url = urljoin(url, href)
                        a_tag.string = f"{a_tag.get_text(strip=True)} [{full_pdf_url}]"
                    else:
                        a_tag["href"] = urljoin(url, href)

                cruft_tags = (
                    main_content.find_all(class_=["breadcrumb", "tabs", "action-links", "pager"])
                    + main_content.find_all(class_="region region-sidebar-second")
                    + main_content.find_all("nav")
                    + main_content.find_all(attrs={"role": "navigation"})
                    + main_content.find_all("img")
                )
                for tag in cruft_tags:
                    tag.decompose()

                title_tag = soup.find("h1", class_="page-header") or soup.find("title")
                title = title_tag.get_text(strip=True).split(" |")[0] if title_tag else "Ανακοίνωση χωρίς τίτλο"

                date_tag = soup.find("time")
                if date_tag and date_tag.has_attr("datetime"):
                    post_date = date_tag["datetime"].split("T")[0]
                else:
                    post_date = datetime.now().strftime("%Y-%m-%d")

                markdown_text = md(str(main_content), heading_style=ATX)
                cleaned_lines = [line.strip() for line in markdown_text.split("\n")]
                cleaned_text = "\n".join(line for line in cleaned_lines if line)

                if len(cleaned_text) < 30:
                    return None

                final_text = f"ΑΝΑΚΟΙΝΩΣΗ: {title}\nΗΜΕΡΟΜΗΝΙΑ: {post_date}\n\n{cleaned_text}"

                # Pydantic validation
                announcement = AnnouncementModel(
                    url=url,
                    title=title,
                    last_modified=post_date,
                    cleaned_content=final_text
                )
                return announcement
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None

    async def run(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            pass

        async with aiohttp.ClientSession() as session:
            all_announcement_links = set()
            for list_url in self.announcement_urls:
                links = await self.discover_announcement_links(session, list_url)
                all_announcement_links.update(links)

            if not all_announcement_links:
                logger.info("No recent announcements found.")
                return

            semaphore = asyncio.Semaphore(self.concurrent_requests)
            tasks = [self.fetch_and_parse_announcement(url, session, semaphore) for url in all_announcement_links]

            logger.info(f"Starting scraping for {len(tasks)} announcements...")
            results = await asyncio.gather(*tasks)

            valid_results = 0
            with open(self.output_file, "a", encoding="utf-8") as f:
                for res in results:
                    if res:
                        f.write(res.model_dump_json() + "\n")
                        valid_results += 1

            logger.info(f"Completed! Saved {valid_results} announcements to {self.output_file}")

if __name__ == "__main__":
    scraper = AnnouncementScraper()
    asyncio.run(scraper.run())
