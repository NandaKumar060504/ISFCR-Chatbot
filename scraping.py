import logging
from typing import TYPE_CHECKING, List, Literal, Optional, Union
if TYPE_CHECKING:
    from selenium.webdriver import Chrome
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import re

logger = logging.getLogger(__name__)

class SeleniumURLLoader(BaseLoader):
    """Load `HTML` pages with `Selenium` and parse with `Unstructured`.

    This is useful for loading pages that require javascript to render.

    Attributes:
        urls (List[str]): List of URLs to load.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        browser (str): The browser to use, either 'chrome' or 'firefox'.
        binary_location (Optional[str]): The location of the browser binary.
        executable_path (Optional[str]): The path to the browser executable.
        headless (bool): If True, the browser will run in headless mode.
        arguments [List[str]]: List of arguments to pass to the browser.
    """
    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        browser: Literal["chrome", "firefox"] = "chrome",
        binary_location: Optional[str] = None,
        executable_path: Optional[str] = None,
        headless: bool = True,
        arguments: List[str] = [],
    ):
        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.browser = browser
        self.binary_location = binary_location
        self.executable_path = executable_path
        self.headless = headless
        self.arguments = arguments


    def _get_driver(self) -> Chrome:
        """Create and return a Chrome WebDriver instance with specified options."""
        chrome_options = ChromeOptions()
        for arg in self.arguments:
            chrome_options.add_argument(arg)
        if self.headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
        if self.binary_location:
            chrome_options.binary_location = self.binary_location
        service = Service(executable_path=self.executable_path) if self.executable_path else None
        return Chrome(options=chrome_options, service=service)

        
    def _build_metadata(self, url: str, driver: Union["Chrome"]) -> dict:
        """Build metadata based on the contents of the webpage"""
        metadata = {
            "source": url,
            "title": "No title found.",
            "description": "No description found.",
            "language": "No language found.",
        }
        if title := driver.title:
            metadata["title"] = title
        try:
            if description := driver.find_element(
                By.XPATH, '//meta[@name="description"]'
            ):
                metadata["description"] = (
                    description.get_attribute("content") or "No description found."
                )
        except NoSuchElementException:
            pass
        try:
            if html_tag := driver.find_element(By.TAG_NAME, "html"):
                metadata["language"] = (
                    html_tag.get_attribute("lang") or "No language found."
                )
        except NoSuchElementException:
            pass
        return metadata

    def load(self) -> List[Document]:
        """Load the specified URLs using Selenium and create Document instances."""
        docs: List[Document] = []
        driver = self._get_driver()

        try:
            for url in self.urls:
                try:
                    driver.get(url)
                    clean_text = self._parse_page(driver, url)
                    metadata = self._build_metadata(url, driver)
                    docs.append(Document(page_content=clean_text, metadata=metadata))
                except Exception as e:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                    if not self.continue_on_failure:
                        raise e
        finally:
            driver.quit()

        return docs

    def _parse_page(self, driver: Chrome, url: str) -> str:
        """Parse the page content based on the URL."""
        # Example refactored logic for different URL patterns
        if url.split("/")[-2] == "team":
            main = driver.find_element(By.TAG_NAME, "main")
            elements = main.find_element(By.CSS_SELECTOR, '[data-testid="mesh-container-content"]')
            name = elements.find_elements(By.TAG_NAME, 'h1')
            elements = elements.find_elements(By.TAG_NAME, 'p')
            text = "\n".join([el.text.strip() for el in elements])
            text2 = "\n".join([el.text.strip() for el in name])
            text = text2 + '\n'+text
        else:
            main = driver.find_element(By.TAG_NAME, "main")
            elements = main.find_element(By.CSS_SELECTOR, '[data-testid="mesh-container-content"]')
            elements = elements.find_elements(By.CSS_SELECTOR, '[data-testid="richTextElement"]')
            text = "\n".join([el.text.strip() for el in elements])

        return re.sub(r'\n\s*\n+', '\n\n', text)
