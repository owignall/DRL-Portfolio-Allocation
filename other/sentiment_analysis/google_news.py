import requests
from bs4 import BeautifulSoup

GOOGLE_HEADERS = {
    'cookie': 'CONSENT=YES+cb.20210727-07-p1.en+FX+232; 1P_JAR=2021-07-29-20; NID=220=IejjYDNZoeiOPhcOHqIC6lQoOU1wblHSv2dlAMLzR9daG7JFjh43rzykeiGRH-mZZ8ZCOPwCXeOxli1CxVkHN6jser1LxE41gkABFLccZsNaGgnajJPsD9LN_4-byjEdPJsAv_k1gnzFzXtjukkroowbISEIDnSQiplLI-h97Us1abCR',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55'
}

def get_articles(search, from_date, to_date, max_pages=1000):
    articles = []
    for i in range(max_pages):
        url = f"https://www.google.co.uk/search?q={search}&tbs=cdr:1,cd_min:{from_date},cd_max:{to_date}&tbm=nws&start={10 * i}"
        print(url)
        page = requests.get(url, headers=GOOGLE_HEADERS)
        soup = BeautifulSoup(page.content,'html.parser')
        articles_div = soup.find('div', id='rso')
        if articles_div == None:
            break
        else:
            a_divs = articles_div.find_all('div', class_="dbsr")
            for a_div in a_divs:
                article = dict()
                article['heading'] = a_div.find('div', role="heading").text
                article['link'] = a_div.find('a')['href']
                article['date'] = a_div.find('span', class_="WG9SHc").text
                articles.append(article)
    return articles


search = "Apple"
from_date = "10/01/2015"
to_date = "11/01/2015"
page_number = 1

# url = f"https://www.google.co.uk/search?q={search}&tbs=cdr:1,cd_min:{from_date},cd_max:{to_date}&tbm=nws&start=0"
# print(url)

articles = get_articles(search, from_date, to_date, max_pages=1)

for a in articles:
    print(a['date'], a['heading'])


