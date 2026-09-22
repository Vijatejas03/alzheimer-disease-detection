"""
Capture 'How to Read This Explanation' user guide and Attribution Color Legend on MRI Analysis page.
"""

import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "reports" / "screenshots"

def capture_mri_legend():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--window-size=1440,1200')
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get("http://localhost:8502")
        time.sleep(3)
        
        # Navigate to MRI Analysis
        labels = driver.find_elements(By.CSS_SELECTOR, 'section[data-testid="stSidebar"] div[role="radiogroup"] label')
        for lbl in labels:
            if "MRI Analysis" in lbl.text:
                lbl.click()
                time.sleep(2)
                break
                
        # Select "Load Curated Research Sample"
        radios = driver.find_elements(By.CSS_SELECTOR, 'div[role="radiogroup"] label')
        for r in radios:
            if "Load Curated Research Sample" in r.text:
                r.click()
                time.sleep(3)
                break
                
        # Scroll section[data-testid="stMain"] down to the User Guide & Legend
        driver.execute_script("""
            var main = document.querySelector('section[data-testid="stMain"]');
            if (main) {
                main.scrollTop = 2600;
            }
        """)
        time.sleep(2)
            
        out_path = SCREENSHOT_DIR / "page_mri_analysis_user_guide_legend.png"
        driver.save_screenshot(str(out_path))
        print(f"Captured MRI Analysis user guide & legend screenshot: {out_path}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    capture_mri_legend()
