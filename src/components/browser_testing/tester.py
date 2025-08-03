
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class BrowserTester:
    """
    Automates browser interactions for web testing.
    """

    def __init__(self, browser: str = "chrome"):
        """
        Initializes the browser tester.

        Args:
            browser: The browser to use (chrome or firefox).
        """
        if browser == "chrome":
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            self.driver = webdriver.Chrome(options=chrome_options)
        elif browser == "firefox":
            firefox_options = webdriver.FirefoxOptions()
            firefox_options.add_argument("--headless")
            self.driver = webdriver.Firefox(options=firefox_options)
        else:
            raise ValueError(f"Unsupported browser: {browser}")

    def navigate_to_url(self, url: str):
        """
        Navigates to a URL.

        Args:
            url: The URL to navigate to.
        """
        self.driver.get(url)

    def get_page_title(self) -> str:
        """
        Gets the title of the current page.

        Returns:
            The page title.
        """
        return self.driver.title

    def close_browser(self):
        """
        Closes the browser.
        """
        self.driver.quit()

