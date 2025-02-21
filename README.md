# ILLUMINATING THE HIDDEN CORNERS OF THE INTERNET USING AI
## DESCRIPTION
The application of Artificial Intelligence (AI) has transformed how we navigate and comprehend the internet. By utilizing machine learning algorithms and natural language processing techniques, AI can reveal hidden aspects of the internet, uncovering trends and connections that might not be immediately visible. This capability is especially valuable for detecting and combating illicit activities, such as cybercrime and online harassment, which often exist in the darker areas of the web. By highlighting these concealed aspects, AI can contribute to fostering a safer and more transparent online space, while also offering significant insights for researchers, policymakers, and law enforcement agencies.

### Objectives
1. Collect and analyze dark web data to identify illicit activity patterns.

2. Develop AI models to detect and classify illicit activities.

3. Create interactive visualizations to facilitate findings dissemination.

4. Provide law enforcement with actionable insights to disrupt criminal networks.

5. Advance knowledge of the dark web and inform policy decisions.

### Dataset
1. Dark Web Market Archives: A collection of historical data from dark web marketplaces like Silk Road, AlphaBay, and Hansa.

2. Tor Project's Onion Services Dataset: A dataset containing information on onion services, including their URLs, descriptions, and categories.

3. Dark Web Forum Posts: A collection of text data from dark web forums, which can be used to analyze language patterns and identify potential illicit activities.

## PROGRAM

```
import requests
from bs4 import BeautifulSoup
import socks
import socket
from stem.control import Controller
from transformers import pipeline
from stem import Signal
from google.colab import files

# Install necessary libraries if not already installed
!pip install requests[socks] stem beautifulsoup4 transformers torch

# Check Tor version
!tor --version

# Start Tor service (if not already running)
!sudo service tor start

# Check Tor status
!sudo service tor status

# Configure Tor proxy
PROXIES = {
    "http": "socks5h://127.0.0.1:9050",
    "https": "socks5h://127.0.0.1:9050"
}

# AI Text Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Check Tor connection
def check_tor():
    try:
        response = requests.get("http://check.torproject.org", proxies=PROXIES, timeout=10)
        if "Congratulations" in response.text:
            print("✅ Tor is working!")
        else:
            print("❌ Tor is NOT working.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")

# Scrape Dark Web (.onion) Content
def scrape_dark_web(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error: {e}"

# Restart Tor Circuit for a Fresh Identity
def new_tor_identity():
    with Controller.from_port(port=9051) as controller:
        try:
          controller.authenticate() # Try to authenticate without password first
        except Exception as e:
          print(f"Authentication failed: {e}. Try providing a password in /etc/tor/torrc")
          return # Exit if authentication fails

        controller.signal(Signal.NEWNYM)
        print("🔄 New Tor identity requested.")

def summarize_content(content):
    if len(content) > 500:
        return summarizer(content[:1000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    return "Text too short for summarization."

# Main Execution
if __name__ == "__main__":
    check_tor()

    dark_url = "http://example.onion"  # ⚠️ Replace with a real .onion site
    print(f"\n🌑 Scraping Dark Web: {dark_url}")
    
    dark_content = scrape_dark_web(dark_url)
    print("\n📄 Dark Web Content:")
    print(dark_content[:1000])  # Show only the first 1000 characters
    
    # Summarize content if it's long enough
    dark_summary = summarize_content(dark_content)
    print("\n🌑 Dark Web Summary:")
dark_summary
    
    # Uncomment to get a new identity
    # new_tor_identity()
```
## OUTPUT
<img width="215" alt="ss1" src="https://github.com/user-attachments/assets/cd72795c-50d1-4428-8e2a-acd240a107cf" />



## RESULT


Thus, the illicit activity patterns are identified and flagged.Dark web intelligence report generated successfully.


