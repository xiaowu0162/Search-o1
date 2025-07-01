import os
from pathlib import Path
import random
import json
from urllib.parse import urlparse
from langchain_community.utilities import GoogleSerperAPIWrapper   # new dependency
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_text_from_url(url, use_jina=False, jina_api_key=None, jina_api_key_file=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            if jina_api_key_file:
                jina_api_key = first_jina_key_with_balance(jina_api_key_file)
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, jina_api_key_file=None,
                       snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, jina_api_key_file, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns None or raises an exception if the request times out.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()  # Raise exception if the request failed
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query}")
        return {}  # Or you can choose to raise an exception
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Bing Web Search request: {e}")
        return {}


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    
    return useful_info


#############################################################################
# New functions to replace bing search with google search
#############################################################################

def first_jina_key_with_balance(
    key_file: str | Path,
    threshold: int = 50_000,
    timeout: int | float = 5,
    verbose: bool = False
) -> Optional[str]:
    """
    Return the first Jina AI API key (after shuffling) whose remaining-token
    balance exceeds *threshold*.

    Parameters
    ----------
    key_file : str | Path
        Text file containing one API key per line.
    threshold : int, default 50_000
        Minimum number of remaining tokens required.
    timeout : int | float, default 5
        Per-request network timeout (seconds).

    Returns
    -------
    str | None
        The first qualifying key, or *None* if none pass the threshold.
    """
    # 1) Read keys
    keys = [ln.strip() for ln in Path(key_file).read_text().splitlines() if ln.strip()]
    if not keys:
        raise ValueError("No keys found in the given file.")

    # 2) Shuffle
    random.shuffle(keys)

    # 3) Query balances one-by-one
    url = "https://r.jina.ai/"
    balance_re = re.compile(r"\[Balance left\]\s+(\d+)", re.I)

    for key in keys:
        headers = {
            "Authorization": f"Bearer {key}",
            "User-Agent": "jina-balance-checker/0.1",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as err:
            # Network / auth errors → skip to next key
            if verbose:
                print(f"⚠️  Failed for {key[:6]}…{key[-6:]}: {err}")
            continue

        m = balance_re.search(resp.text)
        if not m:
            if verbose:
                print(f"⚠️  Balance line not found for {key[:6]}…{key[-6:]}")
            continue

        balance = int(m.group(1))
        if verbose:
            print(f"🔎 {key[:6]}… → balance {balance}")
        if balance > threshold:
            if verbose:
                print(f"✅  Using key {key[:6]}…{key[-6:]} (balance {balance})")
            return key

    if verbose:
        print("❌  No key met the threshold.")
    return None

def google_serper_search(
    query: str,
    *,
    k: int = 10,
    gl: str = "us",
    hl: str = "en",
    serper_api_key: str | None = None,
    **kwargs,
) -> dict:
    """
    Run a Google search via Serper.dev using LangChain's wrapper.

    Args:
        query: Search query string.
        k:     Max number of organic results to return (Serper default=10).
        gl:    Country code for geolocated results (e.g. 'us', 'uk').
        hl:    UI language (e.g. 'en', 'zh-CN').
        serper_api_key:  Optionally pass the key directly; otherwise set
                         `SERPER_API_KEY` in the environment.

    Returns:
        Raw Serper JSON.
    """
    try:
        search = GoogleSerperAPIWrapper(
            serper_api_key=serper_api_key, k=k, gl=gl, hl=hl, **kwargs
        )
        return search.results(query)
    except:
        return {}


def extract_relevant_info_serper(search_results: dict) -> list[dict]:
    useful_info = []
    for idx, hit in enumerate(search_results.get("organic", [])):
        try:
            url = hit.get("link", "")
            site_name = urlparse(url).netloc.replace("www.", "")
            info = {
                "id": idx + 1,
                "title": hit.get("title", ""),
                "url": url,
                "site_name": site_name,
                "date": hit.get("date", ""),         # may be absent → empty string
                "snippet": hit.get("snippet", ""),
                "context": "",
            }
            useful_info.append(info)
        except:
            # exception is very rare and is often caused by invalid urls
            continue
    return useful_info


# ----------------------- Example Usage (Bing Search) -----------------------
# if __name__ == "__main__":
#     # Example usage
#     # Define the query to search
#     query = "Structure of dimethyl fumarate"
    
#     # Subscription key and endpoint for Bing Search API
#     BING_SUBSCRIPTION_KEY = "YOUR_BING_SUBSCRIPTION_KEY"
#     if not BING_SUBSCRIPTION_KEY:
#         raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
#     bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
#     # Perform the search
#     print("Performing Bing Web Search...")
#     search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    
#     print("Extracting relevant information from search results...")
#     extracted_info = extract_relevant_info(search_results)

#     print("Fetching and extracting context for each snippet...")
#     for info in tqdm(extracted_info, desc="Processing Snippets"):
#         full_text = extract_text_from_url(info['url'], use_jina=True)  # Get full webpage text
#         if full_text and not full_text.startswith("Error"):
#             success, context = extract_snippet_with_context(full_text, info['snippet'])
#             if success:
#                 info['context'] = context
#             else:
#                 info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
#         else:
#             info['context'] = f"Failed to fetch full text: {full_text}"

#     # print("Your Search Query:", query)
#     # print("Final extracted information with context:")
#     # print(json.dumps(extracted_info, indent=2, ensure_ascii=False))



# ----------------------- Example Usage (Google Search) -----------------------
if __name__ == "__main__":
    
    query = "Structure of dimethyl fumarate"
    # Make sure your key is visible to the wrapper
    os.environ["SERPER_API_KEY"] = open('/fsx-comem/diwu0162/Search-o1/serper_api_key.txt').readline().strip()
    
    print("Running Google Search (Serper)…")
    raw = google_serper_search(query, k=10)     # <- new call
    print("Extracting metadata …")
    extracted_info = extract_relevant_info_serper(raw)
    print(json.dumps(extracted_info, indent=4))

    snippet_map = {item["url"]: item["snippet"] for item in extracted_info}
    print('Fetching page contents...')
    page_bodies = fetch_page_content(
        urls=list(snippet_map),
        max_workers=10,          # tweak to suit your network / CPU
        use_jina=True,           # set False if you prefer your own scraper
        jina_api_key=None,
        jina_api_key_file='/fsx-comem/diwu0162/Search-o1/jina_api_key.txt',
        snippets=snippet_map,    # each worker gets the snippet tip
    )

    print('Extracting snippets with context...')
    for info in extracted_info:
        raw_text = page_bodies.get(info["url"], "")
        if not raw_text or raw_text.startswith("Error"):
            info["context"] = f"Failed to fetch text: {raw_text[:200]}"
            continue
        success, ctx = extract_snippet_with_context(raw_text, info["snippet"])
        info["context"] = ctx if success else raw_text[:8000]

    out_path = Path("test_serper_search_results_with_context.json")
    out_path.write_text(json.dumps(extracted_info, indent=4, ensure_ascii=False))

