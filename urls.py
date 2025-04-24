import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json

def crawl_web(seed_url, max_pages=10, output_file='crawl_results.json'):
    # List to keep track of visited URLs
    visited = []
    # List to store URLs to be crawled
    queue = [seed_url]

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url not in visited:
            try:
                response = requests.get(url)
                visited.append(url)
                print(f"Crawling {url}")

                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract links from the page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Construct full URL if it's a relative URL
                    full_url = urljoin(url, href)
                    # Check if the URL belongs to the same domain
                    if urlparse(full_url).netloc == urlparse(seed_url).netloc:
                        # Add to the queue if it hasn't been visited and not already in queue
                        if full_url not in visited and full_url not in queue:
                            queue.append(full_url)

            except Exception as e:
                print(f"Error crawling {url}: {e}")

    print(f"Visited {len(visited)} pages.")

    # Save visited URLs to JSON file
    save_to_json(visited, output_file)

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage:
crawl_web("https://www.isfcr.pes.edu/", max_pages=1000, output_file='crawl_results.json')
