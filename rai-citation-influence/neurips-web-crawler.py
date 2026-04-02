import os
import time
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

BASE_URL = "https://proceedings.neurips.cc"
YEARS_URL = f"{BASE_URL}/paper_files/paper"

OUTPUT_DIR = "./bibtex/neurips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUEST_DELAY = 0.5
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}


session = requests.Session()
session.headers.update({
    "User-Agent": "NeurIPS BibTeX Research Crawler"
})

# Track globally seen paper URLs to avoid duplicates
SEEN_PAPERS = set()


def get_soup(url):
    """Fetch page with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        time.sleep(2)
    return None


def get_year_links():
    print("Loading:", BASE_URL)

    response = requests.get(BASE_URL, headers=HEADERS, timeout=30)

    print("Status:", response.status_code)

    if response.status_code != 200:
        raise RuntimeError("Failed to load year index")

    soup = BeautifulSoup(response.text, "html.parser")

    year_links = []
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if "/paper_files/paper/" in href:
            year_links.append(BASE_URL + href)

    return sorted(year_links)


def get_paper_links(year_url):
    """Step 2: get all paper pages for a year."""

    print(f"Loading year page: {year_url}")

    try:
        response = session.get(year_url, timeout=30)
        if response.status_code != 200:
            print(f"Failed to load {year_url} (status {response.status_code})")
            return []
    except Exception as e:
        print(f"Request error for {year_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue

        if "Abstract" in href and "hash" in href:
            full_url = BASE_URL + href if href.startswith("/") else href

            if full_url in SEEN_PAPERS:
                continue

            SEEN_PAPERS.add(full_url)
            links.append(full_url)

    print(f"Found {len(links)} paper links for {year_url}")

    # rate limit to avoid hammering the server
    time.sleep(REQUEST_DELAY)

    return links


def get_bibtex(paper_url):
    """Step 3: extract bibtex from paper page."""
    soup = get_soup(paper_url)
    if soup is None:
        return None

    bib_link = None

    for a in soup.find_all("a"):
        if a.text.strip().lower() == "bibtex":
            bib_link = BASE_URL + a.get("href")
            break

    if not bib_link:
        print(f"No BibTeX found: {paper_url}")
        return None

    try:
        r = session.get(bib_link, timeout=30)
        time.sleep(REQUEST_DELAY)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass

    return None


def crawl_year(year_url):
    """Download all bibtex for a year."""
    year = year_url.split("/")[-1]
    output_file = f"{OUTPUT_DIR}/neurips_{year}.bib"

    existing_entries = set()

    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                if line.startswith("@"):
                    existing_entries.add(line.strip())

    paper_links = get_paper_links(year_url)

    print(f"\nYear {year}: {len(paper_links)} papers")

    with open(output_file, "a") as f:
        for i, paper_url in enumerate(paper_links):

            bib = get_bibtex(paper_url)
            if not bib:
                continue

            entry_id = bib.split("\n")[0]

            if entry_id in existing_entries:
                continue

            f.write(bib + "\n\n")
            existing_entries.add(entry_id)

            if i % 20 == 0:
                print(f"{year}: processed {i}/{len(paper_links)}")

            time.sleep(REQUEST_DELAY)


def main():

    year_links = get_year_links()

    print(f"Found {len(year_links)} years")

    START_YEAR = 2014

    for year_url in year_links:
        year = int(year_url.split("/")[-1])

        if year < START_YEAR:
            continue

        print(f"Starting crawl for year {year}")
        crawl_year(year_url)

    print("\nDone.")


if __name__ == "__main__":
    main()