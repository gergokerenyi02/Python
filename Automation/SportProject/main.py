from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Locating a website
website = "https://www.thesun.co.uk/sport/football"

# Path to the chromedriver
#path = "./chromedriver"

# Setting up the driver
#service = Service(executable_path=path)


driver = webdriver.Chrome() # Added chromedriver to usr/local/bin

# Opening the website
driver.get(website)




# Waiting for the elements to be present
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@class, "teaser__copy-container")]')))


containers = driver.find_elements(by="xpath", value='//div[@class="teaser__copy-container"]')

titles = []
subtitles = []
links = []




for container in containers:
    container_html = container.get_attribute('outerHTML')
    #print(f"Container HTML: {container_html}")
    try:
        
        #title = container.find_element(by="xpath", value='./h3').text
        #subtitle = container.find_element(By.XPATH, './span').text
        link = container.find_element(by="xpath", value='./a').get_attribute("href")
    except Exception as e:
        print("Error occured: ", e)
        continue
    

    #titles.append(title)
    #subtitles.append(subtitle)
    links.append(link)


print(titles)
print(subtitles)
print(links)

driver.quit()