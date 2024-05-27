import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Locating a website
website = "https://books.toscrape.com/"

# Path to the chromedriver
#path = "./chromedriver"

# Setting up the driver
#service = Service(executable_path=path)


driver = webdriver.Chrome() # Added chromedriver to usr/local/bin

# Opening the website
driver.get(website)



wait = WebDriverWait(driver, 10)


containers = driver.find_elements(by="xpath", value='//article[@class="product_pod"]')

titles = []
prices = []
links = []
stocks = []

for container in containers:
    try:
        title = container.find_element(by="xpath", value='./h3/a').get_attribute("title")
        link = container.find_element(by="xpath", value='./h3/a').get_attribute("href")
        price = container.find_element(by="xpath", value='./div/p[@class="price_color"]').text
        stock = container.find_element(by="xpath", value='./div/p[@class="instock availability"]').text
        
        titles.append(title)
        links.append(link)
        prices.append(price)
        stocks.append(stock)
    except Exception as e:
        print("Error occured: ", e)
        continue
    


#print(titles)
#print(prices)
#print(links)
#print(stocks)

pd.DataFrame({
    'Title': titles,
    'Price': prices,
    'Link': links,
    'Stock': stocks
}).to_csv("books.csv")


driver.quit()