import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Set
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from markdownify import ATX, markdownify as md

# ==========================================
# Settings & URLS
# ==========================================
BASE_URL = "https://dit.uop.gr/"
OUTPUT_DIR = Path("website_data")
OUTPUT_FILE = OUTPUT_DIR / "dit_announcements.jsonl"
CONCURRENT_REQUESTS = 5

OUTPUT_DIR.mkdir(exist_ok=True)

ANNOUNCEMENT_URLS = [
    "https://dit.uop.gr/all-announcements",
]

months_ago = datetime.now() - relativedelta(months=2)
FROM_DATE = months_ago.strftime("%Y-%m-%d")


async def discover_announcement_links(
    session: aiohttp.ClientSession, base_list_url: str
) -> Set[str]:
    """Discover announcement links from the base list URL by applying the 'from' date filter.

    This function iterates through paginated pages of the announcement list,
    starting from the specified FROM_DATE, and collects unique URLs of announcements.

    Returns:
        Set: A set of unique full URLs to individual announcements.
    """
    found_links = set()
    page = 0

    print(f"\nLooking announcments from: {FROM_DATE} to: {base_list_url}")

    # list of paths to ignore
    IGNORE_PATHS = ["/en", "/user", "/taxonomy", "/rss", "/search"]

    while True:
        url = f"{base_list_url}?from={FROM_DATE}&page={page}"

        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    break

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                main_content = soup.find("main", role="main")
                if not main_content:
                    break

                previous_link_count = len(found_links)

                for a_tag in main_content.find_all("a", href=True):
                    href = a_tag["href"]

                    # Ignore pagination, anchors και filters
                    if "?page=" in href or "?from=" in href or href.startswith("#"):
                        continue

                    # Ignore (href="/") and known paths that dont include announcments
                    if href == "/" or any(
                        href.startswith(path) for path in IGNORE_PATHS
                    ):
                        continue

                    if href.startswith("/") and not href.endswith(
                        (".pdf", ".zip", ".jpg", ".png", ".doc", ".docx")
                    ):
                        full_url = urljoin(BASE_URL, href)
                        if full_url not in ANNOUNCEMENT_URLS:
                            found_links.add(full_url)

                if len(found_links) == previous_link_count:
                    print(f"   [End of list at page {page}]")
                    break

                page += 1
                await asyncio.sleep(0.5)

        except Exception as e:
            print(f"Error occured {page}: {e}")
            break

    print(f"Found {len(found_links)} unique announcments (until {FROM_DATE}).")
    return found_links


async def fetch_and_parse_announcement(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
):
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

                # PDF Links
                for a_tag in main_content.find_all("a", href=True):
                    href = a_tag["href"]
                    if href.lower().endswith(".pdf"):
                        full_pdf_url = urljoin(url, href)
                        a_tag.string = f"{a_tag.get_text(strip=True)} [{full_pdf_url}]"
                    else:
                        a_tag["href"] = urljoin(url, href)

                # remove unwanted tags
                cruft_tags = (
                    main_content.find_all(
                        class_=["breadcrumb", "tabs", "action-links", "pager"]
                    )
                    + main_content.find_all(class_="region region-sidebar-second")
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
                    else "Ανακοίνωση χωρίς τίτλο"
                )

                # date
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

                final_text = (
                    f"ΑΝΑΚΟΙΝΩΣΗ: {title}\nΗΜΕΡΟΜΗΝΙΑ: {post_date}\n\n{cleaned_text}"
                )

                return {
                    "url": url,
                    "title": title,
                    "content_category": "Ανακοινώσεις",
                    "last_modified": post_date,
                    "cleaned_content": final_text,
                }

        except Exception as e:
            print(f"Σφάλμα στο {url}: {e}")
            return None


async def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    async with aiohttp.ClientSession() as session:
        all_announcement_links = set()

        for list_url in ANNOUNCEMENT_URLS:
            links = await discover_announcement_links(session, list_url)
            all_announcement_links.update(links)

        if not all_announcement_links:
            print("Δεν βρέθηκαν πρόσφατες ανακοινώσεις.")
            return

        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        tasks = [
            fetch_and_parse_announcement(url, session, semaphore)
            for url in all_announcement_links
        ]

        print(f"\nΞεκινάει η λήψη κειμένου για {len(tasks)} ανακοινώσεις...")
        results = await asyncio.gather(*tasks)

        valid_results = 0
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for res in results:
                if res:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    valid_results += 1

        print(
            f"\nΟλοκληρώθηκε! Αποθηκεύτηκαν {valid_results} ΠΡΟΣΦΑΤΕΣ ανακοινώσεις στο {OUTPUT_FILE}"
        )


if __name__ == "__main__":
    asyncio.run(main())
