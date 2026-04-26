import asyncio
import json
from pathlib import Path
from typing import Dict
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from markdownify import ATX, markdownify as md

# ==========================================
#               Settings
# ==========================================
SITEMAP_URL = "https://dit.uop.gr/sitemap.xml"
OUTPUT_DIR = Path("website_data")
OUTPUT_FILE = OUTPUT_DIR / "dit_website.jsonl"
CONCURRENT_REQUESTS = 5  # parallel request

OUTPUT_DIR.mkdir(exist_ok=True)

# Words in URLs that we want to exclude
EXCLUDE_URL_PATTERNS = [
    "/en",
    "/en/",
    "/user",
    "/user/",
    "/search",
    "404",
    "403",
    "oroi-hrisis",
    "politiki-cookies",
    "palaioi-odigoi-spoydon",
    "alumni",
    "mathimata",
    # "members",
    "staff",
    "proin-meli",
    "dilosi-prosbasimotitas",
]


def should_crawl(url: str) -> bool:
    """Check if the url will be scraped, or it's on the excluded list"""
    for pattern in EXCLUDE_URL_PATTERNS:
        if pattern in url:
            return False
    return True


# ==========================================
#           SITEMAP PARSING
# ==========================================
async def get_urls_from_sitemap(session: aiohttp.ClientSession) -> dict:
    """Fetch and parse the sitemap to extract URLs and their last modified dates.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for the request.

    Returns:
        dict: A dictionary mapping URLs to their last modified dates (YYYY-MM-DD format), or 'Άγνωστο' if not available.
    """
    print(f"Downloading sitemap from: {SITEMAP_URL}")
    try:
        async with session.get(SITEMAP_URL) as response:
            xml_content = await response.text()
            soup = BeautifulSoup(xml_content, "xml")

            url_data = {}
            for url_node in soup.find_all("url"):
                loc_node = url_node.find("loc")
                lastmod_node = url_node.find("lastmod")

                if loc_node:
                    url: str = loc_node.text.strip()
                    if should_crawl(url):
                        url = url.replace("http:", "https:")
                        if lastmod_node:
                            date_str = lastmod_node.text.strip().split("T")[0]
                        else:
                            date_str = "Άγνωστο"

                        url_data[url] = date_str

            print(f"Found {len(url_data)} eligible URLs in the sitemap.")
            return url_data

    except Exception as e:
        print(f"Failed to fetch sitemap: {e}")
        return {}


# ==========================================
#           EXTRACTION
# ==========================================
async def fetch_and_parse(
    url: str, lastmod: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> Dict | None:
    """
    Fetch and parse the content of a given URL, extracting relevant data into a dictionary.

    Args:
        url (str): The URL to fetch and parse.
        lastmod (str): The last modified date of the URL.
        session (aiohttp.ClientSession): The HTTP session to use for the request.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
        dict or None: A dictionary containing parsed data (url, title, content_category, last_modified, keywords, summary, language, cleaned_content) if successful, otherwise None.
    """
    async with semaphore:
        try:
            async with session.get(url, timeout=15) as response:
                if response.status != 200:
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                main_content = soup.find("main", role="main")
                if not main_content:
                    return None

                cruft_tags = (
                    main_content.find_all(
                        class_=["breadcrumb", "tabs", "action-links", "feed-icons"]
                    )
                    + main_content.find_all(class_="region-sidebar-second")
                    + main_content.find_all("nav")
                    + main_content.find_all(attrs={"role": "navigation"})
                    + main_content.find_all("img")
                )
                for tag in cruft_tags:
                    tag.decompose()

                title_tag = soup.find("h1", class_="page-header") or soup.find("title")
                title = (
                    title_tag.get_text(strip=True).split(" |")[0]
                    if title_tag
                    else "Χωρίς Τίτλο"
                )

                for a_tag in main_content.find_all("a", href=True):
                    a_tag["href"] = urljoin(url, a_tag["href"])

                markdown_text = md(str(main_content), heading_style=ATX)

                cleaned_lines = [line.strip() for line in markdown_text.split("\n")]

                cleaned_text = "\n".join(line for line in cleaned_lines if line)

                if len(cleaned_text) < 50:
                    return None

                return {
                    "url": url,
                    "title": title,
                    "content_category": "",
                    "last_modified": lastmod,
                    "keywords": [],
                    "summary": "",
                    "language": "el",
                    "cleaned_content": cleaned_text,
                }

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None


async def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    async with aiohttp.ClientSession() as session:
        url_data = await get_urls_from_sitemap(session)
        if not url_data:
            return

        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        tasks = [
            fetch_and_parse(url, lastmod, session, semaphore)
            for url, lastmod in url_data.items()
        ]

        print(f"Starting scraping for {len(tasks)} pages...")
        results = await asyncio.gather(*tasks)

        valid_results = 0
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for res in results:
                if res:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    valid_results += 1

        print(
            f"\nCompleted! Saved {valid_results} pages to {OUTPUT_FILE}"
        )

if __name__ == "__main__":
    asyncio.run(main())