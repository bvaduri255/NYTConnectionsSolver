from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

import json
import pandas as pd

DRIVER_PATH = r"chromedriver-win64\chromedriver-win64\chromedriver.exe"


# options = Options()
# options.headless = True
# options.add_argument("--window-size=1920,1200")
service = Service(executable_path=DRIVER_PATH)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service = service, options = options)

template_URL = "https://connections.swellgarfo.com/nyt/"

total_puzzles = 304
puzzles = []
descriptions = []

for puzzle in range(1,total_puzzles + 1):
    driver.get(template_URL + str(puzzle))

    #print(driver.page_source)
    elem = driver.find_element(By.ID, "__NEXT_DATA__")
    jsontext = json.loads(elem.get_attribute('innerHTML'))
    ans_array = jsontext["props"]["pageProps"]["answers"]

    temp_data = []

    for i in range(len(ans_array)):
        temp = ans_array[i]["words"]
        temp_data.append(temp)
        temp_description = ans_array[i]["description"]
        descriptions.append(temp_description)

    puzzles.append(temp_data)

df = pd.DataFrame(puzzles)
df.to_csv("connections.csv", sep=',', encoding='utf-8')