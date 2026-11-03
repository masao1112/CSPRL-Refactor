"""Check the static research page against a running local HTTP server."""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[2]
options = Options()
options.binary_location = os.environ.get("CHROME_BINARY", "/opt/google/chrome/chrome")
for argument in ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--window-size=1440,1100"]:
    options.add_argument(argument)
options.set_capability("goog:loggingPrefs", {"browser": "ALL", "performance": "ALL"})
driver = webdriver.Chrome(service=Service(os.environ["CHROMEDRIVER"]), options=options)
wait = WebDriverWait(driver, 12)
url = os.environ.get("DEMO_URL", "http://127.0.0.1:8765")


def el(id):
    return driver.find_element(By.ID, id)


try:
    driver.get(url)
    driver.execute_script("document.documentElement.style.scrollBehavior='auto'")
    wait.until(lambda _: el("motion-toggle").get_attribute("aria-pressed") == "true")
    wait.until(lambda _: el("process-animation").get_property("naturalWidth") == 1200)
    assert len(driver.find_elements(By.TAG_NAME, "button")) == 1
    assert not driver.find_elements(By.CSS_SELECTOR, "input, select")
    assert len(driver.find_elements(By.TAG_NAME, "table")) == 3
    assert "224,7 ± 0,3" in el("comparison-table").text
    assert "225,4 ± 1,7" in el("ablation-table").text
    assert len(driver.find_elements(By.CSS_SELECTOR, ".metric-guide > div")) == 7
    assert len(driver.find_elements(By.CSS_SELECTOR, ".parameter-list > div")) == 13
    assert "90,1" in el("output").text
    assert "thấp hơn mốc hiện hữu" in el("metrics").text
    assert "60.000" in driver.find_element(By.CSS_SELECTOR, ".parameter-groups").text
    assert "Về trang minh họa" not in driver.find_element(By.TAG_NAME, "body").text
    assert driver.execute_script("return document.querySelector('#results').compareDocumentPosition(document.querySelector('#method')) & Node.DOCUMENT_POSITION_FOLLOWING")
    assert el("process-animation").get_attribute("src").endswith(".gif")
    # Pointer access, including WebDriver's automatic scroll to the control.
    el("motion-toggle").click()
    assert el("process-animation").get_attribute("src").endswith("planning-poster.png")
    el("motion-toggle").click()
    assert el("process-animation").get_attribute("src").endswith(".gif")

    # Respect the OS preference, and still allow explicit opt-in.
    driver.execute_cdp_cmd("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]})
    wait.until(lambda _: el("motion-toggle").get_attribute("aria-pressed") == "false")
    driver.refresh()
    wait.until(lambda _: el("motion-toggle").is_displayed())
    assert el("process-animation").get_attribute("src").endswith("planning-poster.png")

    for image in driver.find_elements(By.TAG_NAME, "img"):
        source = image.get_attribute("src")
        with urlopen(source) as response:
            assert response.status == 200
    with urlopen(urljoin(url + "/", "assets/paper.pdf")) as response:
        assert response.read(4) == b"%PDF"
    for source in (ROOT / "res").glob("*.png"):
        target = ROOT / "demo/assets/figures" / source.name
        assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
    with Image.open(ROOT / "demo/assets/planning-process.gif") as animation:
        assert animation.is_animated
        assert animation.n_frames == 49
        assert animation.size == (1200, 660)

    driver.execute_script("document.documentElement.style.scrollBehavior='auto';window.scrollTo(0,0)")
    driver.save_screenshot("/tmp/csprl-research-desktop.png")
    for image in driver.find_elements(By.CSS_SELECTOR, ".map-grid img"):
        assert image.size["width"] <= 210
    assert driver.find_element(By.CSS_SELECTOR, ".teaser").size["width"] <= 720
    for width in [320, 390, 768, 1024, 1440]:
        driver.set_window_size(width, 1100)
        assert not driver.execute_script("return document.documentElement.scrollWidth>innerWidth"), f"Overflow at {width}"
    driver.set_window_size(390, 1100)
    driver.execute_script("window.scrollTo(0,0)")
    driver.save_screenshot("/tmp/csprl-research-mobile.png")
    driver.set_window_size(1440, 1100)
    driver.execute_script("document.querySelector('#metrics').scrollIntoView({behavior:'instant'})")
    driver.save_screenshot("/tmp/csprl-research-metrics.png")
    driver.execute_script("document.querySelector('#parameters').scrollIntoView({behavior:'instant'})")
    driver.save_screenshot("/tmp/csprl-research-parameters.png")
    driver.execute_script("document.querySelector('#illustration').scrollIntoView({behavior:'instant'})")
    driver.save_screenshot("/tmp/csprl-research-animation.png")

    # Static figures and evidence must be readable with JS completely disabled.
    driver.execute_cdp_cmd("Emulation.setScriptExecutionDisabled", {"value": True})
    driver.refresh()
    assert len(driver.find_elements(By.TAG_NAME, "table")) == 3
    assert not el("motion-toggle").is_displayed()
    assert el("process-animation").get_attribute("src").endswith("planning-poster.png")
    assert "Tóm tắt" in el("abstract").text
    assert "224,7" in el("output").text
    assert len(driver.find_elements(By.CSS_SELECTOR, ".metric-guide dd")) == 7
    assert "120.000" in driver.find_element(By.CSS_SELECTOR, ".parameter-groups").text
    requests = []
    for entry in driver.get_log("performance"):
        message = json.loads(entry["message"])["message"]
        if message["method"] == "Network.requestWillBeSent":
            requests.append(message["params"]["request"]["url"])
    assert all(request.startswith(url) or request.startswith("data:") for request in requests)
    assert not any(any(ext in request for ext in ["simulation.mjs", "roads.json", ".onnx", ".zip", ".pt"]) for request in requests)
    errors = [entry for entry in driver.get_log("browser") if entry["level"] == "SEVERE"]
    assert not errors, errors
    print("PASS: output-first layout, 7 metric explanations, 13 parameter explanations, compact figures, original assets, GIF pause/play, reduced motion, no-JS content, paper tables, 5 viewport widths, no runtime simulation/model or external requests")
finally:
    driver.quit()
