"""
Playwright-based product page scraper.

Orchestrates browser automation and coordinates the three parsing layers
(JSON-LD → Embedded JSON → HTML fallback) to extract product data from
individual product pages.
"""

from __future__ import annotations

import asyncio
import logging
import httpx
from playwright.async_api import async_playwright, BrowserContext, Page

from .jsonld_parser import extract_jsonld_product, _find_product_in_jsonld, _flatten_product, _flatten_product_group
from .embedded_json_parser import extract_embedded_product, find_product_data
from .html_parser import extract_html_product

logger = logging.getLogger(__name__)

# Realistic user agent to reduce bot detection
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Resource types to block for faster page loads
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


def merge_product_data(*dicts: dict[str, Any] | None) -> dict[str, Any]:
    """Merge multiple product dicts left-to-right.

    Later dicts override earlier ones, but only for non-empty values.
    A value is considered "empty" if it is None, '', or [].
    """
    merged: dict[str, Any] = {}
    for d in dicts:
        if d is None:
            continue
        for key, value in d.items():
            if value is None or value == "" or value == []:
                continue
            merged[key] = value
    return merged


def extract_product_from_jsonld_strings(script_texts: list[str]) -> dict[str, Any] | None:
    """Extract and flatten product data from raw JSON-LD text strings using standard library json.loads."""
    import json
    import re
    for text in script_texts:
        if not text or not text.strip():
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        product = _find_product_in_jsonld(data)
        if product:
            ld_type = product.get("@type", "")
            if isinstance(ld_type, list):
                type_list = [t.lower() for t in ld_type if isinstance(t, str)]
            elif isinstance(ld_type, str):
                type_list = [ld_type.lower()]
            else:
                type_list = []

            if "productgroup" in type_list:
                return _flatten_product_group(product)
            else:
                return _flatten_product(product)
    return None


def parse_playwright_embedded(next_data_text: str | None, embedded_js: dict | None) -> dict[str, Any] | None:
    """Extract and parse product data from embedded Next.js props, window states, or dataLayer structures."""
    import json
    
    if next_data_text:
        try:
            data = json.loads(next_data_text)
            page_props = data.get("props", {}).get("pageProps", {})
            prod = find_product_data(page_props) or find_product_data(data)
            if prod:
                return prod
        except Exception:
            pass

    if embedded_js:
        for key in ['__INITIAL_STATE__', '__PRELOADED_STATE__', '__APP_STATE__', 'productData', 'product']:
            val_str = embedded_js.get(key)
            if val_str:
                try:
                    data = json.loads(val_str)
                    prod = find_product_data(data)
                    if prod:
                        return prod
                except Exception:
                    pass
        
        dl_str = embedded_js.get('dataLayer')
        if dl_str:
            try:
                data = json.loads(dl_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            ecommerce = item.get("ecommerce", {})
                            products = (
                                ecommerce.get("detail", {}).get("products")
                                or ecommerce.get("items")
                                or ecommerce.get("products")
                            )
                            if isinstance(products, list) and products:
                                prod = products[0]
                                if isinstance(prod, dict):
                                    return prod
            except Exception:
                pass
                
    return None


class ProductScraper:
    """Async Playwright-based product page scraper.

    Launches a headless browser, navigates to product pages concurrently
    (limited by semaphore), and extracts product data through a three-layer
    parsing pipeline.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 15000,
        max_concurrent: int = 3,
        dom_wait_ms: int = 1500,
        http_retry_count: int = 1,
    ) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._dom_wait_ms = dom_wait_ms
        self._http_retry_count = http_retry_count
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def scrape_products(
        self, product_results: list[dict[str, Any]], browser: Any = None
    ) -> list[dict[str, Any]]:
        """Scrape product data from a list of URLs.

        Args:
            product_results: List of dicts, each with at least a 'url' key
                             and optional metadata (title, price, source, thumbnail).
            browser: Optional shared Playwright Browser instance. If provided,
                     the browser lifecycle is managed by the caller.

        Returns:
            List of raw product dicts (Nones filtered out).
        """
        if not product_results:
            return []

        scraped: list[dict[str, Any]] = []

        async with httpx.AsyncClient(
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
            timeout=10.0,
        ) as http_client:
            if browser is not None:
                context = await browser.new_context(
                    user_agent=_USER_AGENT,
                    viewport={"width": 1280, "height": 800},
                    java_script_enabled=True,
                )
                try:
                    tasks = [
                        self._scrape_with_semaphore(context, result, http_client)
                        for result in product_results
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning("Scraping task failed: %s", result)
                            continue
                        if result is not None:
                            scraped.append(result)
                finally:
                    await context.close()
            else:
                async with async_playwright() as p:
                    browser_instance = await p.chromium.launch(headless=self._headless)
                    context = await browser_instance.new_context(
                        user_agent=_USER_AGENT,
                        viewport={"width": 1280, "height": 800},
                        java_script_enabled=True,
                    )
                    try:
                        tasks = [
                            self._scrape_with_semaphore(context, result, http_client)
                            for result in product_results
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        for result in results:
                            if isinstance(result, Exception):
                                logger.warning("Scraping task failed: %s", result)
                                continue
                            if result is not None:
                                scraped.append(result)
                    finally:
                        await context.close()
                        await browser_instance.close()

        logger.info(
            "Scraped %d products from %d URLs",
            len(scraped),
            len(product_results),
        )
        return scraped

    async def _scrape_with_semaphore(
        self, context: BrowserContext, result: dict[str, Any], http_client: httpx.AsyncClient | None = None
    ) -> dict[str, Any] | None:
        """Wrap single-page scraping with concurrency control."""
        async with self._semaphore:
            return await self._scrape_single_product(context, result, http_client)

    async def _scrape_single_product(
        self, context: BrowserContext, result: dict[str, Any], http_client: httpx.AsyncClient | None = None
    ) -> dict[str, Any] | None:
        """Scrape a single product page.

        First attempts to fetch HTML via an HTTP client and parse it.
        If that fails, falls back to Playwright.
        """
        import time
        import json
        import re
        url = result.get("url", "")
        if not url:
            return None

        overall_start = time.time()
        
        # Timing/state variables
        http_duration = 0.0
        status_code = "N/A"
        html_size = 0
        jsonld_duration = 0.0
        embedded_duration = 0.0
        html_parsing_duration = 0.0
        total_http_duration = 0.0
        playwright_fallback = False
        method = "failed"
        merged = None

        run_http = (http_client is not None) and (not result.get("skip_http", False))

        if run_http:
            logger.info("HTTP request start for URL: %s", url)
            http_start_time = time.time()
            try:
                response = None
                max_attempts = max(1, self._http_retry_count + 1)
                for attempt in range(max_attempts):
                    try:
                        response = await http_client.get(url)
                        if response.status_code == 200:
                            break
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise e
                        await asyncio.sleep(0.2)

                if response is not None:
                    status_code = response.status_code
                    http_duration = time.time() - http_start_time
                    
                    if status_code == 200:
                        html_content = response.text
                        html_size = len(response.content)
                        
                        # 1. JSON-LD parsing
                        jsonld_start = time.time()
                        jsonld_data = None
                        try:
                            jsonld_data = extract_jsonld_product(html_content)
                        except Exception as e:
                            logger.debug("HTTP JSON-LD extraction error: %s", e)
                        jsonld_duration = time.time() - jsonld_start

                        if jsonld_data and jsonld_data.get("name") and str(jsonld_data["name"]).strip():
                            merged = jsonld_data
                            method = "http_jsonld"
                        else:
                            # 2. Embedded JSON parsing
                            embedded_start = time.time()
                            embedded_data = None
                            try:
                                embedded_data = extract_embedded_product(html_content)
                            except Exception as e:
                                logger.debug("HTTP Embedded JSON extraction error: %s", e)
                            embedded_duration = time.time() - embedded_start

                            if embedded_data and embedded_data.get("name") and str(embedded_data["name"]).strip():
                                merged = embedded_data
                                method = "http_embedded_json"
                            else:
                                # 3. HTML parsing (BeautifulSoup DOM) fallback
                                html_start = time.time()
                                html_data = None
                                try:
                                    html_data = extract_html_product(html_content, url)
                                except Exception as e:
                                    logger.debug("HTTP HTML DOM extraction error: %s", e)
                                html_parsing_duration = time.time() - html_start

                                if html_data and html_data.get("name") and str(html_data["name"]).strip():
                                    merged = html_data
                                    method = "http_html_fallback"
                                else:
                                    logger.info("HTTP extraction failed to find valid name. Falling back to Playwright.")
                                    playwright_fallback = True
                        
                        total_http_duration = time.time() - http_start_time
                    else:
                        logger.warning("HTTP status %s for %s. Falling back to Playwright.", status_code, url)
                        playwright_fallback = True
                else:
                    playwright_fallback = True
            except Exception as exc:
                http_duration = time.time() - http_start_time
                status_code = f"Error: {type(exc).__name__}"
                logger.warning("HTTP request failed for %s: %s. Falling back to Playwright.", url, exc)
                playwright_fallback = True
        else:
            playwright_fallback = True

        pw_duration = 0.0
        if playwright_fallback:
            pw_start = time.time()
            logger.info("Playwright fallback start for URL: %s", url)
            
            page: Page | None = None
            try:
                page = await context.new_page()

                # Block heavy resources for faster loading
                await page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type in _BLOCKED_RESOURCE_TYPES
                        else route.continue_()
                    ),
                )

                nav_committed = False
                try:
                    await page.goto(url, timeout=self._timeout_ms, wait_until="commit")
                    nav_committed = True
                    logger.info("Navigation committed for URL: %s in %.2fs", url, time.time() - pw_start)
                except Exception as exc:
                    logger.warning(
                        "Navigation timeout/error for %s (commit failed): %s. Attempting extraction on whatever loaded.",
                        url,
                        exc,
                    )

                # Wait for basic content or tags to load in the DOM
                try:
                    await page.wait_for_function(
                        """() => {
                            const hasJsonLd = Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
                                .some(s => s.textContent.includes('"Product"') || s.textContent.includes('"ProductGroup"'));
                            if (hasJsonLd) return true;

                            if (document.getElementById('__NEXT_DATA__') || 
                                window.__INITIAL_STATE__ || 
                                window.__PRELOADED_STATE__ || 
                                window.__APP_STATE__ || 
                                window.productData || 
                                window.product) {
                                return true;
                            }

                            if (document.querySelector('h1')) {
                                return true;
                            }

                            return false;
                        }""",
                        timeout=self._dom_wait_ms
                    )
                except Exception:
                    pass

                # Playwright loads the webpage; extract the fully rendered HTML and parse with BeautifulSoup pipeline
                logger.info("Playwright loaded webpage; extracting rendered HTML for BeautifulSoup parsing.")
                content = await page.content()

                # 1. JSON-LD parsing on rendered HTML
                try:
                    jsonld_data = extract_jsonld_product(content)
                    if jsonld_data and jsonld_data.get("name") and str(jsonld_data["name"]).strip():
                        merged = jsonld_data
                        method = "playwright_rendered_jsonld"
                except Exception as e:
                    logger.debug("Playwright rendered JSON-LD extraction failed: %s", e)

                # 2. Embedded JSON parsing on rendered HTML
                if not merged:
                    try:
                        embedded_data = extract_embedded_product(content)
                        if embedded_data and embedded_data.get("name") and str(embedded_data["name"]).strip():
                            merged = embedded_data
                            method = "playwright_rendered_embedded_json"
                    except Exception as e:
                        logger.debug("Playwright rendered embedded extraction failed: %s", e)

                # 3. HTML parsing (BeautifulSoup) on rendered HTML
                if not merged:
                    try:
                        html_data = extract_html_product(content, url)
                        if html_data and html_data.get("name") and str(html_data["name"]).strip():
                            merged = html_data
                            method = "playwright_rendered_beautifulsoup"
                    except Exception as e:
                        logger.debug("Playwright rendered BeautifulSoup parsing failed: %s", e)

                if not merged:
                    logger.warning("All extraction methods failed on Playwright rendered content for %s", url)
                    method = "playwright_failed"

                pw_duration = time.time() - pw_start

            except Exception as exc:
                logger.warning("Failed to scrape %s via Playwright: %s", url, exc)
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass

        total_duration = time.time() - overall_start

        # Detailed timing logs
        logger.info("Product URL: %s", url)
        if run_http:
            logger.info("HTTP fetch: %.2fs", http_duration)
            logger.info("HTTP status: %s", status_code)
            logger.info("HTML size: %.2f MB", html_size / (1024 * 1024))
            
            # Show path details
            if method == "http_jsonld":
                logger.info("JSON-LD extraction: %.2fs", jsonld_duration)
            elif method == "http_embedded_json":
                logger.info("JSON-LD: failed (%.2fs)", jsonld_duration)
                logger.info("Embedded JSON extraction: %.2fs", embedded_duration)
            elif method == "http_html_fallback":
                logger.info("JSON-LD: failed (%.2fs)", jsonld_duration)
                logger.info("Embedded JSON: failed (%.2fs)", embedded_duration)
                logger.info("HTML parsing: %.2fs", html_parsing_duration)
            else:
                logger.info("HTTP extraction failed")

            logger.info("Total HTTP extraction: %.2fs", total_http_duration)
        
        if playwright_fallback:
            logger.info("Playwright fallback: Yes (duration: %.2fs)", pw_duration)
        else:
            logger.info("Playwright fallback: No")

        logger.info("Method: %s", method)
        logger.info("Total: %.2fs", total_duration)

        if not merged:
            logger.warning("No product data extracted from %s", url)
            return None

        # Attach URL and SERP metadata
        merged["product_url"] = url
        merged["serp_meta"] = {
            "title": result.get("title"),
            "price": result.get("price") or result.get("extracted_price"),
            "source": result.get("source"),
            "thumbnail": result.get("thumbnail"),
        }

        return merged

def scrape_product_pages(product_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synchronous wrapper for ProductScraper.scrape_products.

    Loads configuration and runs the async scraper in a new event loop.

    Args:
        product_results: List of result dicts from search result parser.

    Returns:
        List of raw product data dicts.
    """
    # Import config here to avoid circular imports at module level
    from config import get_config

    config = get_config()
    scraper = ProductScraper(
        headless=config.PLAYWRIGHT_HEADLESS,
        timeout_ms=config.PLAYWRIGHT_TIMEOUT_MS,
    )

    # Handle running inside an existing event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (e.g. FastAPI) — create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, scraper.scrape_products(product_results))
            return future.result()
    else:
        return asyncio.run(scraper.scrape_products(product_results))
