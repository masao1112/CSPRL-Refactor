"""Browser checks for the offline planning workflow. Start the local HTTP server first."""
import csv
import io
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select, WebDriverWait

download_dir=Path(tempfile.mkdtemp(prefix="csprl-csv-"))
options=Options()
options.binary_location=os.environ.get("CHROME_BINARY","/opt/google/chrome/chrome")
for value in ["--headless=new","--no-sandbox","--disable-dev-shm-usage","--window-size=1440,1100"]:
    options.add_argument(value)
options.add_experimental_option("prefs",{"download.default_directory":str(download_dir),"download.prompt_for_download":False})
options.set_capability("goog:loggingPrefs",{"browser":"ALL","performance":"ALL"})
driver=webdriver.Chrome(service=Service(os.environ["CHROMEDRIVER"]),options=options)
wait=WebDriverWait(driver,25)
url=os.environ.get("DEMO_URL","http://127.0.0.1:8765")

def el(id):
    return driver.find_element(By.ID,"planner-"+id)

def click(id):
    driver.execute_script("arguments[0].scrollIntoView({block:'center',behavior:'instant'})",el(id))
    el(id).click()

def scrub(step):
    driver.execute_script("arguments[0].value=arguments[1];arguments[0].dispatchEvent(new Event('input',{bubbles:true}))",el("timeline"),step)

def keyboard_marker(selector):
    marker=driver.find_element(By.CSS_SELECTOR,selector)
    driver.execute_script("arguments[0].focus()",marker)
    ActionChains(driver).send_keys(Keys.ENTER).perform()

try:
    driver.get(url)
    driver.execute_script("document.documentElement.style.scrollBehavior='auto'")
    disclosure=driver.find_element(By.ID,"planning-demo")
    assert not disclosure.get_attribute("open")
    assert driver.find_element(By.CSS_SELECTOR,".paper-header .authors").is_displayed()
    assert len(driver.find_elements(By.CSS_SELECTOR,".metric-guide > div"))==7
    assert len(driver.find_elements(By.CSS_SELECTOR,".parameter-list > div"))==13
    assert driver.execute_script("return getComputedStyle(document.body).backgroundColor")=="rgb(255, 255, 255)"
    paper_table=driver.find_element(By.ID,"comparison-table").get_attribute("innerHTML")
    assert not driver.find_elements(By.CSS_SELECTOR,".topbar, .intro")
    initial_requests=[json.loads(e["message"])["message"] for e in driver.get_log("performance")]
    assert not any(m["method"]=="Network.requestWillBeSent" and "planning-data.json" in m["params"]["request"]["url"] for m in initial_requests)
    disclosure.find_element(By.TAG_NAME,"summary").click()
    wait.until(lambda _:el("play").is_enabled())
    assert el("metric-stations").text=="5"
    assert len(driver.find_elements(By.CSS_SELECTOR,"#planner-main-map .station-marker"))==5
    assert len(driver.find_elements(By.CSS_SELECTOR,".charger-grid > div"))==12
    assert "639" in el("dataset-status").text
    assert "900.000 triệu đồng = 900 tỷ đồng" in el("budget-summary").text
    assert "515,0 m" in el("inspector-content").text
    assert "triệu đồng" in el("inspector-content").text
    assert "Chi phí mặt bằng" in el("inspector-content").text
    assert "Chi phí thiết bị" in el("inspector-content").text
    assert "đv" not in driver.find_element(By.ID,"planning-demo").text
    assert len(driver.find_elements(By.CSS_SELECTOR,"#planner-main-map [data-layer=demand] circle"))>500
    click("step")
    assert el("metric-stations").text=="6"
    keyboard_marker('#planner-main-map [data-station-id="S06"]')
    assert "S06" in el("inspector-content").text
    assert driver.execute_script("return document.activeElement.dataset.mapFocus")=="main-station-S06"
    # The first charger addition must change the drawn service radius, and rewind restores it.
    click("layer-service")
    scrub(4)
    target=re.search(r"tại (S\d+)",el("current-action").text).group(1)
    expanded=float(driver.find_element(By.CSS_SELECTOR,f'#planner-main-map [data-service-station="{target}"]').get_attribute("r"))
    scrub(3)
    previous=float(driver.find_element(By.CSS_SELECTOR,f'#planner-main-map [data-service-station="{target}"]').get_attribute("r"))
    assert expanded>previous
    click("layer-service")
    scrub(6)
    assert "Di dời" in el("current-action").text
    scrub(0)
    assert el("metric-stations").text=="5"
    assert el("station-select").get_attribute("value")=="S01"
    Select(el("speed")).select_by_value("200")
    click("play")
    wait.until(lambda _:int(el("timeline").get_attribute("value"))>0)
    click("play")
    paused=el("timeline").get_attribute("value")
    time.sleep(.35)
    assert el("timeline").get_attribute("value")==paused
    for method in ["ga","benefit","demand","dqn"]:
        Select(el("method")).select_by_value(method)
        assert el("metric-stations").text=="5"
        assert el("timeline").get_attribute("value")=="0"
        click("step")
        assert el("metric-stations").text=="6"
    best_step=int(re.search(r"bước (\d+)",el("best-step-note").text).group(1))
    click("best")
    assert int(el("timeline").get_attribute("value"))==best_step
    selected_count=int(el("metric-stations").text)
    click("export")
    wait.until(lambda _:len(list(download_dir.glob("*.csv")))==1)
    path=next(download_dir.glob("*.csv"))
    rows=list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    assert len(rows)==selected_count
    assert all(int(row["step"])==best_step for row in rows)
    assert all(row["data_mode"]=="illustrative_no_model" for row in rows)
    powers=[3,7,11,20,22,30,60,80,120,150,180,250]
    for row in rows:
        assert int(row["power_kw"])==sum(int(row[f"chargers_{power}_kw"])*power for power in powers)
        assert row["cost_unit"]=="million_vnd"
        assert float(row["budget_total_million_vnd"])==900000
        assert abs(float(row["station_cost_million_vnd"])-float(row["land_cost_million_vnd"])-float(row["equipment_cost_million_vnd"]))<1e-6
        assert abs(float(row["radius_km"])-1/(1+math.exp(-float(row["power_kw"])/1000)))<1e-9
    before=el("step-label").text
    Select(el("compare-a")).select_by_value("benefit")
    Select(el("compare-b")).select_by_value("demand")
    assert el("step-label").text==before
    assert len(driver.find_elements(By.CSS_SELECTOR,"#planner-comparison-body tr"))==8
    keyboard_marker('#planner-compare-map-a [data-station-id="S01"]')
    assert "Phương án A" in el("inspector-source").text
    assert "S01" in el("compare-detail").text
    click("export-b")
    wait.until(lambda _:len(list(download_dir.glob("*.csv")))==2)
    second=next(p for p in download_dir.glob("*.csv") if p!=path)
    compare_rows=list(csv.DictReader(io.StringIO(second.read_text(encoding="utf-8-sig"))))
    assert all(row["method"]=="demand" for row in compare_rows)
    click("layer-demand")
    assert not driver.find_elements(By.CSS_SELECTOR,"[data-layer=demand] circle")
    click("layer-service")
    assert driver.find_elements(By.CSS_SELECTOR,"#planner-main-map [data-layer=service] circle")
    click("layer-grid")
    assert not driver.find_elements(By.CSS_SELECTOR,".bus-marker")
    click("layer-grid")
    keyboard_marker("#planner-main-map .bus-marker")
    assert "thanh cái" in el("inspector-title").text
    assert "MW" in el("inspector-content").text
    click("layer-demand")
    click("layer-service")
    click("reset")
    assert el("metric-stations").text=="5"
    driver.execute_script("window.scrollTo(0,0)")
    driver.save_screenshot("/tmp/csprl-planner-desktop.png")
    driver.execute_script("document.querySelector('#planner-comparison').scrollIntoView({behavior:'instant'})")
    driver.save_screenshot("/tmp/csprl-planner-comparison.png")
    for width in [320,390,768,1024,1440]:
        driver.set_window_size(width,1100)
        assert not driver.execute_script("return document.documentElement.scrollWidth>innerWidth"),f"Overflow at {width}"
    driver.set_window_size(390,1100)
    driver.execute_script("window.scrollTo(0,0)")
    driver.save_screenshot("/tmp/csprl-planner-mobile.png")
    assert driver.find_element(By.ID,"comparison-table").get_attribute("innerHTML")==paper_table
    # Closing the optional demo stops replay; the scientific article remains visible.
    click("play")
    disclosure.find_element(By.TAG_NAME,"summary").click()
    time.sleep(.25)
    paused=el("timeline").get_attribute("value")
    time.sleep(.3)
    assert el("timeline").get_attribute("value")==paused
    assert driver.find_element(By.ID,"comparison-table").is_displayed()
    errors=[e for e in driver.get_log("browser") if e["level"]=="SEVERE"]
    assert not errors,errors
    requests=[]
    for entry in driver.get_log("performance"):
        message=json.loads(entry["message"])["message"]
        if message["method"]=="Network.requestWillBeSent":
            requests.append(message["params"]["request"]["url"])
    assert all(r.startswith(url) or r.startswith(("data:","blob:")) for r in requests),requests
    assert not any(any(token in r for token in [".onnx",".zip",".pth","huggingface","tile.openstreetmap"]) for r in requests)
    driver.execute_cdp_cmd("Network.enable",{})
    driver.execute_cdp_cmd("Network.setBlockedURLs",{"urls":["*planning-data.json*"]})
    driver.refresh()
    disclosure=driver.find_element(By.ID,"planning-demo")
    if not disclosure.get_attribute("open"):
        disclosure.find_element(By.TAG_NAME,"summary").click()
    wait.until(lambda _:"Không tải được" in el("load-status").text)
    assert not el("play").is_enabled()
    driver.execute_cdp_cmd("Network.setBlockedURLs",{"urls":[]})
    el("load-status").find_element(By.TAG_NAME,"button").click()
    wait.until(lambda _:el("play").is_enabled())
    assert el("metric-stations").text=="5"
    print("PASS: original research layout and paper table preserved, lazy optional demo, close-to-pause, 5 real initial stations, all layers, 4 methods, play/pause/step/rewind/best, 12-level inspector, keyboard, comparison, actual CSV downloads, 5 viewports, local-only requests, data failure/retry")
finally:
    driver.quit()
