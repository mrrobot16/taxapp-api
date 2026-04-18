import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pathlib import Path

INDEX_URL = "https://www.irs.gov/downloads/irs-pdf"
TOTAL_PAGES = 62
DOWNLOAD_BASE = "https://www.irs.gov/pub/irs-pdf/"
OUTPUT_DIR = Path("./data/irs_forms")

async def fetch_page(session, page):
    url = f"{INDEX_URL}?page={page}"
    async with session.get(url) as resp:
        html = await resp.text()
    soup = BeautifulSoup(html, "html.parser")
    filenames = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".pdf"):
            filenames.add(urlparse(href).path.split("/")[-1])
    return filenames

async def fetch_all_pages(session):
    tasks = [fetch_page(session, page) for page in range(TOTAL_PAGES)]
    results = await asyncio.gather(*tasks)
    all_filenames = set()
    for page_filenames in results:
        all_filenames.update(page_filenames)
    return sorted(all_filenames)

async def download_pdf(session, filename, semaphore):
    async with semaphore:
        url = DOWNLOAD_BASE + filename
        dest = OUTPUT_DIR / filename
        if dest.exists():
            return
        try:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status == 200:
                    async with aiofiles.open(dest, "wb") as f:
                        await f.write(await resp.read())
                    print(f"✅ {filename}")
                else:
                    print(f"⚠️  {filename}: HTTP {resp.status}")
        except Exception as e:
            print(f"❌ {filename}: {e}")

async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(20)  # 20 concurrent downloads
    async with aiohttp.ClientSession() as session:
        pdfs = await fetch_all_pages(session)
        print(f"Found {len(pdfs)} PDFs")
        tasks = [download_pdf(session, pdf, semaphore) for pdf in pdfs]
        await asyncio.gather(*tasks)

asyncio.run(main())