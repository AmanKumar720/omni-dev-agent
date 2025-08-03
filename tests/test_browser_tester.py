
import unittest
from src.components.browser_testing.tester import BrowserTester

class TestBrowserTester(unittest.TestCase):

    def test_chrome_navigation(self):
        """Test navigation with Chrome."""
        try:
            tester = BrowserTester(browser="chrome")
            tester.navigate_to_url("https://www.google.com")
            title = tester.get_page_title()
            tester.close_browser()
            self.assertEqual(title, "Google")
        except Exception as e:
            # Catching WebDriver exceptions if Chrome is not found
            self.fail(f"Chrome navigation test failed with an exception: {e}")

    @unittest.skip("Skipping Firefox test for now. Please ensure Firefox and geckodriver are installed.")
    def test_firefox_navigation(self):
        """Test navigation with Firefox."""
        try:
            tester = BrowserTester(browser="firefox")
            tester.navigate_to_url("https://www.mozilla.org")
            title = tester.get_page_title()
            tester.close_browser()
            self.assertIn("Mozilla", title)
        except Exception as e:
            self.fail(f"Firefox navigation test failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()

