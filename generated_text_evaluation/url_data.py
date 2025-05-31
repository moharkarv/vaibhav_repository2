from newspaper import Article
import requests

def extract_text_from_url(url):
    try:
        # Check if URL is reachable before proceeding
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Use newspaper3k to extract title and text
        article = Article(url)
        article.download()
        article.parse()

        title = article.title
        text = article.text

        # Return combined text (title + body), truncated to 5000 chars
        full_text = f"{title}\n\n{text}"
        return full_text[:5000]

    except requests.exceptions.RequestException as e:
        return f"Error fetching the URL: {e}"
    except Exception as e:
        return f"Error processing the article: {e}"

# Example usage:
# url = "https://example.com/article"
# extracted_text = extract_text_from_url(url)
# print(extracted_text_
