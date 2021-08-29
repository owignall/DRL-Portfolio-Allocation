from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains

from bs4 import BeautifulSoup

from time import sleep

# PATH = "C:/Program Files (x86)/Google/ChromeDriver/chromedriver.exe"
PATH = "other\chromedriver\chromedriver.exe"
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1200x600')
# options.add_experimental_option("excludeSwitches", ['enable-automation']);
# chromeDriverService = ChromeDriverService.CreateDefaultService()
# chromeDriverService.HideCommandPromptWindow = True
driver = webdriver.Chrome(PATH, chrome_options=options)



driver.get('https://www.google.co.uk/search?q=Apple&tbs=cdr:1,cd_min:01/01/2009,cd_max:01/01/2010&tbm=nws&start=0')



# button = EC.presence_of_element_located((By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div[1]/div/button'))
# WebDriverWait(driver, 5).until(button)
# form = driver.find_elements_by_xpath("/html/body/div[2]/div[3]/form")
# sleep(2)
WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div[1]/div/button')))
button = driver.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div[1]/div/button')

# print(form.text)
# sleep(1)

# form.submit()
button.click()



WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="hdtbMenus"]/span[2]/g-popup/div[1]')))
driver.find_element_by_xpath('//*[@id="hdtbMenus"]/span[2]/g-popup/div[1]').click()

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="lb"]/div/g-menu/g-menu-item[8]')))
driver.find_element_by_xpath('//*[@id="lb"]/div/g-menu/g-menu-item[8]').click()

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="OouJcb"]')))
from_date_box = driver.find_element_by_xpath('//*[@id="OouJcb"]')
from_date_box.clear()
from_date_box.send_keys("01/01/2016")

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="rzG2be"]')))
to_date_box = driver.find_element_by_xpath('//*[@id="rzG2be"]')
to_date_box.clear()
to_date_box.send_keys("12/31/2016")

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="T3kYXe"]/g-button')))
driver.find_element_by_xpath('//*[@id="T3kYXe"]/g-button').click()


sleep(5)

WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="lst-ib"]')))
search_box = driver.find_element_by_xpath('//*[@id="lst-ib"]')
search_box.clear()
search_box.send_keys("Microsoft")
search_box.submit()



for i in range(50):
    # sleep(1)
    # next_button = driver.find_element_by_xpath('//*[@id="pnnext"]')

    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="pnnext"]')))
    next_button = driver.find_element_by_xpath('//*[@id="pnnext"]')


    next_button.click()

    page = driver.page_source


    soup = BeautifulSoup(page,'html.parser')
    articles_div = soup.find('div', id='rso')
    if articles_div == None:
        pass
    else:
        a_divs = articles_div.find_all('div', class_="dbsr")
        for a_div in a_divs:
            # print(a_div)
            article = dict()
            article['heading'] = a_div.find('div', role="heading").text
            article['link'] = a_div.find('a')['href']
            article['date'] = a_div.find('span', class_="WG9SHc").text
            # articles.append(article)
            print(article['date'], "-", article['heading'])


sleep(100)

driver.quit()