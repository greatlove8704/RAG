import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import os
import time
import re
import urllib3 


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

target_urls = [
    "https://en.wikipedia.org/wiki/Pittsburgh",
    "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "https://pittsburghpa.gov/", 
    "https://www.britannica.com/place/Pittsburgh",
    "https://www.visitpittsburgh.com",
    "https://apps.pittsburghpa.gov/redtail/images/23255_2024_Operating_Budget.pdf",
    "https://www.cmu.edu/about/",
    "https://pittsburgh.events",
    "https://downtownpittsburgh.com/events/",
    "https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d",
    "https://events.cmu.edu",
    "https://www.cmu.edu/engage/alumni/events/campus/index.html",
    "https://www.pittsburghsymphony.org",
    "https://pittsburghopera.org",
    "https://trustarts.org",
    "https://carnegiemuseums.org",
    "https://www.heinzhistorycenter.org",
    "https://www.thefrickpittsburgh.org",
    "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh",
    "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
    "https://www.picklesburgh.com/",
    "https://www.pghtacofest.com/",
    "https://pittsburghrestaurantweek.com/",
    "https://littleitalydays.com",
    "https://bananasplitfest.com",
    "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
    "https://www.mlb.com/pirates",
    "https://www.steelers.com",
    "https://www.nhl.com/penguins/",
    "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/tax-forms/9622_amusement_tax_regulations.pdf",
    "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/tax-forms/9626_payroll_tax_regulations.pdf",
    "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/tax-forms/9623_isp_tax_regulations.pdf",
    "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/tax-forms/9624_local_services_tax_regulations.pdf",
    "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/tax-forms/9625_parking_tax_regulations.pdf",
    "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/tax-forms/9627_uf_regulations.pdf"
]

OUTPUT_DIR = "knowledge_base_raw"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def sanitize_filename(url):
    filename = re.sub(r'https?://', '', url)
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename[:100]

def fetch_html_content(url):
    print(f"Fetching HTML: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20, verify=True) # Start with True for HTML
        if url == "https://pittsburgh.events": 
             print("Trying https://pittsburgh.events with verify=False")
             response = requests.get(url, headers=HEADERS, timeout=20, verify=False)

        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style", "nav", "footer", "aside"]):
            script_or_style.decompose()
        text_lines = (line.strip() for line in soup.get_text().splitlines())
        text = "\n".join(line for line in text_lines if line)
        return text
    except requests.exceptions.SSLError as e:
        print(f"SSL Error fetching HTML {url}, trying with verify=False: {e}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=20, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(["script", "style", "nav", "footer", "aside"]):
                script_or_style.decompose()
            text_lines = (line.strip() for line in soup.get_text().splitlines())
            text = "\n".join(line for line in text_lines if line)
            return text
        except requests.exceptions.RequestException as e_retry:
            print(f"Error fetching HTML {url} even with verify=False: {e_retry}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HTML {url}: {e}")
        return None

def fetch_pdf_content(url):
    print(f"Fetching PDF: {url}")
    try:
        # Add verify=False for PDFs
        response = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        response.raise_for_status()
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)
        text = ""
        with open(temp_pdf_path, 'rb') as f_pdf:
            reader = PdfReader(f_pdf)
            for page in reader.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + "\n"
        os.remove(temp_pdf_path)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDF {url}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing PDF {url}: {e}")
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        return None

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    processed_urls_file = os.path.join(OUTPUT_DIR, "_processed_urls.txt")
    processed_urls = set()

    if os.path.exists(processed_urls_file):
        with open(processed_urls_file, 'r', encoding='utf-8') as f:
            processed_urls = set(line.strip() for line in f)

    for url in target_urls:
        if url in processed_urls:
            print(f"Skipping already processed: {url}")
            continue

        filename_base = sanitize_filename(url)
        output_path = os.path.join(OUTPUT_DIR, filename_base + ".txt")
        content = None

        if url.lower().endswith(".pdf"):
            content = fetch_pdf_content(url)
        else:
            content = fetch_html_content(url)

        if content and content.strip():
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Saved: {output_path}")
                with open(processed_urls_file, 'a', encoding='utf-8') as f_processed:
                    f_processed.write(url + "\n")
                processed_urls.add(url)
            except Exception as e:
                print(f"Error writing file for {url}: {e}")
        else:
            print(f"No content for {url}")

        time.sleep(1.5) 
    print("finished.")