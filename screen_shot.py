from playwright.sync_api import sync_playwright
from pathlib import Path
import uuid


def capture_screenshot(url: str) -> str:
    """Captures a screenshot of the webpage and returns the file path."""
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    screenshot_path = screenshot_dir / f"{uuid.uuid4()}.png"

    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=10000)  # Wait for page to load
        page.screenshot(path=screenshot_path, full_page=True)
        browser.close()

    return screenshot_path
from playwright.sync_api import sync_playwright
from pathlib import Path
import uuid


def capture_screenshot(url: str) -> str:
    """Captures a screenshot of the webpage and returns the file path."""
    screenshot_path = f"screenshots/{uuid.uuid4()}.png"
    Path("screenshots").mkdir(exist_ok=True)  # Ensure directory exists
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=10000)  # Wait for page to load
        page.screenshot(path=screenshot_path, full_page=True)
        browser.close()

    return screenshot_path
if __name__ == "__main__":
    url = "https://www.bbc.com/"  # Change to the desired webpage URL
    screenshot_file = capture_screenshot(url)
    print(f"Screenshot saved at: {screenshot_file}")
    
