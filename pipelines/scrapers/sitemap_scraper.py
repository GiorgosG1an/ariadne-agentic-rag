import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from markdownify import ATX, markdownify as md
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from fake_useragent import UserAgent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pipelines.core.logger import get_logger
from pipelines.schemas.documents import WebsiteModel

logger = get_logger(__name__)

class SitemapScraper:
    def __init__(self, sitemap_url: str = "https://dit.uop.gr/sitemap.xml", output_dir: str = "data/website_data"):
        self.sitemap_url = sitemap_url
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "dit_website.jsonl"
        self.concurrent_requests = 5
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ua = UserAgent()

        self.exclude_url_patterns = [
            "/en", "/en/", "/user", "/user/", "/search", "404", "403",
            "oroi-hrisis", "politiki-cookies", "palaioi-odigoi-spoydon",
            "alumni", "mathimata", "staff", "proin-meli", "dilosi-prosbasimotitas",
        ]

    def get_headers(self):
        return {"User-Agent": self.ua.random}

    def should_crawl(self, url: str) -> bool:
        for pattern in self.exclude_url_patterns:
            if pattern in url:
                return False
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url, timeout=15, headers=self.get_headers()) as response:
            response.raise_for_status()
            return await response.text()

    async def get_urls_from_sitemap(self, session: aiohttp.ClientSession) -> dict:
        logger.info(f"Downloading sitemap from: {self.sitemap_url}")
        try:
            xml_content = await self.fetch_page(session, self.sitemap_url)
            soup = BeautifulSoup(xml_content, "xml")

            url_data = {}
            for url_node in soup.find_all("url"):
                loc_node = url_node.find("loc")
                lastmod_node = url_node.find("lastmod")

                if loc_node:
                    url: str = loc_node.text.strip()
                    if self.should_crawl(url):
                        url = url.replace("http:", "https:")
                        date_str = lastmod_node.text.strip().split("T")[0] if lastmod_node else "Άγνωστο"
                        url_data[url] = date_str

            logger.info(f"Found {len(url_data)} eligible URLs in the sitemap.")
            return url_data
        except Exception as e:
            logger.error(f"Failed to fetch sitemap: {e}")
            return {}

    async def fetch_and_parse(self, url: str, lastmod: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> WebsiteModel | None:
        async with semaphore:
            try:
                html = await self.fetch_page(session, url)
                soup = BeautifulSoup(html, "html.parser")

                main_content = soup.find("main", role="main")
                if not main_content:
                    return None

                cruft_tags = (
                    main_content.find_all(class_=["breadcrumb", "tabs", "action-links", "feed-icons"])
                    + main_content.find_all(class_="region-sidebar-second")
                    + main_content.find_all("nav")
                    + main_content.find_all(attrs={"role": "navigation"})
                    + main_content.find_all("img")
                )
                for tag in cruft_tags:
                    tag.decompose()

                title_tag = soup.find("h1", class_="page-header") or soup.find("title")
                title = title_tag.get_text(strip=True).split(" |")[0] if title_tag else "Χωρίς Τίτλο"

                for a_tag in main_content.find_all("a", href=True):
                    a_tag["href"] = urljoin(url, a_tag["href"])

                markdown_text = md(str(main_content), heading_style=ATX)
                cleaned_lines = [line.strip() for line in markdown_text.split("\n")]
                cleaned_text = "\n".join(line for line in cleaned_lines if line)

                if len(cleaned_text) < 50:
                    return None

                return WebsiteModel(
                    url=url,
                    title=title,
                    last_modified=lastmod,
                    cleaned_content=cleaned_text
                )
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None

    async def run(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            pass

        async with aiohttp.ClientSession() as session:
            url_data = await self.get_urls_from_sitemap(session)
            if not url_data:
                return

            semaphore = asyncio.Semaphore(self.concurrent_requests)
            tasks = [self.fetch_and_parse(url, lastmod, session, semaphore) for url, lastmod in url_data.items()]

            logger.info(f"Starting scraping for {len(tasks)} pages...")
            results = await asyncio.gather(*tasks)

            valid_results = 0
            with open(self.output_file, "a", encoding="utf-8") as f:
                for res in results:
                    if res:
                        f.write(res.model_dump_json() + "\n")
                        valid_results += 1

            logger.info(f"Completed! Saved {valid_results} pages to {self.output_file}")

if __name__ == "__main__":
    scraper = SitemapScraper()
    asyncio.run(scraper.run())