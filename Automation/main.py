from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Locating a website
website = "https://www.thesun.co.uk/sport/football"

# Path to the chromedriver
path = "./chromedriver"

# Setting up the driver
service = Service(executable_path=path)
driver = webdriver.Chrome(service=service)

# Opening the website
driver.get(website)

containers = driver.find_elements(by="xpath", value='//div[@class="teaser__copy-container"]/a/h2')

for container in containers:
    title = container.find_element(by="xpath", value="./a/h2").text
    subtitle = container.find_element(by="xpath", value="./a/p").text
    link = container.find_element(by="xpath", value="./a").get_attribute("href")