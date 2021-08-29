from constants import *
from storage import *
# General
from datetime import datetime
import numpy as np
import pandas as pd
import math
import time
# Scraping
import requests
from bs4 import BeautifulSoup
import threading
import re
import json
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Sentiment
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Article:
    def __init__(self, a_title, a_link, a_author):
        self.title = a_title
        self.link = a_link
        self.author = a_author
        self.content = None
    
    def __str__(self):
        return f"{self.title}\nBy {self.author}"
    
    def extract_content(self):
        try:
            if self.link[0] != "/":
                raise ValueError("Link for this article is not a url extension")
            else:
                url = f"https://uk.investing.com{self.link}"
                page = requests.get(url, headers=STANDARD_HEADERS)
                soup = BeautifulSoup(page.content,'html.parser')

                left_col = soup.find('section', id='leftColumn')
                article_page = left_col.find('div', class_='WYSIWYG articlePage')

                content = ""
                for p in article_page.find_all('p'):
                    text = p.text
                    if len(text) > 2:
                        content += text

                self.content = content
        except Exception as e:
            print(e)

class InvestmentRanking:
    """DEPRICATED"""
    def __init__(self, a_firm, a_action, a_from_grade, a_to_grade):
        self.firm = a_firm
        self.action = a_action
        self.from_grade = a_from_grade
        self.to_grade = a_to_grade

class DailyData:
    """DEPRICATED"""
    def __init__(self, a_date, a_open, a_high, a_low, a_close, a_adj_close, a_volume):
        # Historical data
        self.date = a_date
        self.open = float(a_open)
        self.high = float(a_high)
        self.low = float(a_low)
        self.close = float(a_close)
        self.adj_close = float(a_adj_close)
        self.volume = int(a_volume)
        # Technical indicators
        self.ema12 = None
        self.ema26 = None
        self.macd = None
        self.signal_line = None
        self.close_change = None
        self.up_sma14 = None
        self.down_sma14 = None
        self.rsi = None
        self.normalized_rsi = None
        self.sma20 = None
        self.std_dev20 = None
        self.std_devs_out = None
        self.bb_upper = None
        self.bb_lower = None
        self.vol_sma60 = None
        self.relative_vol = None
        # Qualitative data
        self.articles = []
        self.investment_rankings = []
    
    def __str__(self):
        return f"{self.date}\t{self.close}\t{len(self.articles)} Articles\t{len(self.investment_rankings)} Investment Rankings"

class OldStock:
    def __init__(self, a_name, a_code, a_ic_name):
        self.name = a_name
        self.code = a_code
        self.ic_name = a_ic_name
        self.data_start = int(time.time() - 8 * E_YEAR)
        self.data_end = int(time.time())
        self.rs_decay = 0.9
        self.data = []

        self.investing_link =f"https://uk.investing.com/equities/{self.ic_name}"
        self.yahoo_link = f"https://uk.finance.yahoo.com/quote/{self.code}"

    def __str__(self):
        rep = f"{self.name} ({self.code}) - {len(self.data)} Data points"
        return rep
    
    def print_data_overview(self):
        print(f"\nDate\tOpen\tArticles")
        for d in self.data:
            print(d)
    
    def print_all_articles(self):
        for d in self.data:
            for a in d.articles:
                print(f"{d.date}\t{a.title}")
    
    def extract_price_data(self):
        period1 = str(self.data_start)
        period2 = str(self.data_end)
        interval = "1d"
        file_link = f"https://query1.finance.yahoo.com/v7/finance/download/{self.code}?period1={period1}&period2={period2}&interval={interval}"
        request = requests.get(file_link, headers=STANDARD_HEADERS)
        content = str(request.content).replace("'", "").split("\\n")
        for i in range(1, len(content)):
            new_point = DailyData(*content[i].split(","))
            self.data.append(new_point)
        if len(self.data) == 0:
            raise Exception("No data was retrieved by the extraction function")

    def extract_financial_data(self):
        # MIGHT NOT BE ABLE TO FIND ENOUGH DATA FOR THIS
        # This code gives annual financials for last 4 years or last 4 months.
        # REF
        # https://www.mattbutton.com/how-to-scrape-stock-upgrades-and-downgrades-from-yahoo-finance/
        """
        url = f"https://uk.finance.yahoo.com/quote/GOOG/financials"
        page = requests.get(url, headers=STANDARD_HEADERS)
        soup = BeautifulSoup(page.content,'html.parser')
        script = soup.find('script', text=re.compile(r'root\.App\.main'))
        json_text = re.search(r'^\s*root\.App\.main\s*=\s*({.*?})\s*;\s*$', script.string, flags=re.MULTILINE).group(1)
        data = json.loads(json_text)
        financials = data['context']['dispatcher']['stores']['QuoteSummaryStore']
        """

    def extract_inv_news_data(self, threads=5, extract_content=True):     
        def _convert_date(news_date):
            # Converts format of date e.g. Apr 14, 2021 -> 2021-04-14
            if "ago" in news_date:
                return datetime.today().strftime('%Y-%m-%d')
            
            date_componets = news_date.replace(",", "").split(" ")
            return f"{date_componets[2]}-{MONTHS[date_componets[0].lower()]}-{date_componets[1]}"
        
        def _extract_content_function(article):
            article.extract_content()

        def _extract_from_page(articles_dict, n, i, valid):
            # Adds 
            url = f"https://uk.investing.com/equities/{self.ic_name}-news/{str(n)}"
            page = requests.get(url, headers=STANDARD_HEADERS)
            soup = BeautifulSoup(page.content,'html.parser')

            # Check you haven't looped
            p_number = soup.find('div', id='paginationWrap').find('a', class_='pagination selected').text
            
            if n != int(p_number):
                valid[i] = False
            else:
                articles_section = soup.find('section', id='leftColumn')
                articles = articles_section.find_all('article')
                for article in articles:
                    d = article.find('div')
                    a = d.find('a')
                    details_sec = d.find(class_= 'articleDetails')
                    details = details_sec.find_all('span')

                    title = a.text
                    link = a['href']
                    author = details[0].text[3:]
                    date = _convert_date(details[1].text[3:])

                    new_article = Article(title, link, author)
                    if date in articles_dict:
                        articles_dict[date].append(new_article)
                    else:
                        articles_dict[date] = [new_article]
                valid[i] = True
        
        print("Extracting Articles")
        # Populate a dictionary with scraped articles
        articles_dict = dict()
        # Use threading to request pages and extract articles
        start_page = 1
        searching = True
        while searching:
            threads_list = []
            valid = [None] * threads
            for n in range(start_page, start_page + threads):
                t = threading.Thread(target=_extract_from_page, args=(articles_dict, n, n - 1 - start_page, valid))
                t.start()
                threads_list.append(t)
            for t in threads_list:
                t.join()
            if not all(valid):
                searching = False
            else:
                start_page += threads

        # Move articles to respective elements of data attribute
        extractable_articles_list = []
        for i in range(len(self.data)):
            if self.data[i].date in articles_dict:
                self.data[i].articles = articles_dict[self.data[i].date]
                extractable_articles_list += [a for a in articles_dict[self.data[i].date] if a.link[0] == "/"]
            else:
                self.data[i].articles = []
        
        print("Extracting Content")
        # Extracting article content
        if extract_content:
            # Extract using threading
            start_index = 0
            extracting = True
            while extracting:
                threads_list = []
                for i in range(start_index, start_index + threads):
                    print(i)
                    if i < len(extractable_articles_list):
                        # print("Extracting article")
                        t = threading.Thread(target=_extract_content_function, args=(extractable_articles_list[i],))
                        t.start()
                        threads_list.append(t)
                    else:
                        extracting = False
                for t in threads_list:
                    t.join()
                start_index += threads

    def extract_investment_ranking_data(self):
        # REFERENCE
        # https://www.mattbutton.com/how-to-scrape-stock-upgrades-and-downgrades-from-yahoo-finance/
        url = f"https://uk.finance.yahoo.com/quote/{self.code}/analysis"
        page = requests.get(url, headers=STANDARD_HEADERS)
        soup = BeautifulSoup(page.content,'html.parser')
        script = soup.find('script', text=re.compile(r'root\.App\.main'))
        json_text = re.search(r'^\s*root\.App\.main\s*=\s*({.*?})\s*;\s*$', script.string, flags=re.MULTILINE).group(1)
        data = json.loads(json_text)
        rankings_scraped = data['context']['dispatcher']['stores']['QuoteSummaryStore']['upgradeDowngradeHistory']['history']
        
        # OLD APPROACH
        # Create dictionary of InvestmentRanking object
        rankings_dict = dict()
        for r in rankings_scraped:
            investment_ranking = InvestmentRanking(r['firm'], r['action'], r['fromGrade'], r['toGrade'])
            date = datetime.fromtimestamp(r['epochGradeDate']).strftime('%Y-%m-%d')
            if date in rankings_dict:
                rankings_dict[date].append(investment_ranking)
            else:
                rankings_dict[date] = [investment_ranking]
        # Add InvestmentRanking objects to data attribute
        for i in range(len(self.data)):
            if self.data[i].date in rankings_dict:
                self.data[i].investment_rankings = rankings_dict[self.data[i].date]

    def extract_all_data(self, verbose=False):
        if verbose: print("Extracting price data")
        self.extract_price_data()
        # self.extract_financial_data()
        if verbose: print("Extracting investment ranking data")
        self.extract_investment_ranking_data()
        if verbose: print("Extracting news data")
        self.extract_inv_news_data()

    def calculate_technical_indicators(self):
        """Iterates through daily data calculating technical indicators."""
        smoothing = 2
        # MACD
        # Calculating EMAs and MACD
        self.data[11].ema12 = sum([d.close for d in self.data[:12]]) / 12
        self.data[25].ema26 = sum([d.close for d in self.data[:26]]) / 26
        for i in range(12, len(self.data)):
            if i > 11: # EMA 12
                self.data[i].ema12 = (self.data[i].close * (smoothing / 13)) + (self.data[i-1].ema12 * (1 - (smoothing / 13)))
            if i > 25: # EMA 26
                self.data[i].ema26 = (self.data[i].close * (smoothing / 27)) + (self.data[i-1].ema26 * (1 - (smoothing / 27)))
            if i > 24: # MACD
                self.data[i].macd = self.data[i].ema12 - self.data[i].ema26
        # Calculating Signal Line
        self.data[33].signal_line = sum([d.macd for d in self.data[25:34]]) / 9
        for i in range(34, len(self.data)):
            self.data[i].signal_line = (self.data[i].macd * (smoothing / 10)) + (self.data[i-1].signal_line * (1 - (smoothing / 10)))
        
        # RSI
        # Calculating close changes
        for i in range(1, len(self.data)):
            self.data[i].close_change = self.data[i].close - self.data[i-1].close
        # Up and Down SMA 14
        for i in range(14, len(self.data)):
            self.data[i].up_sma14 = sum([d.close_change if d.close_change > 0 else 1e-8 for d in self.data[i-13:i+1]]) / 14
            self.data[i].down_sma14 = sum([abs(d.close_change) if d.close_change < 0 else 1e-8 for d in self.data[i-13:i+1]]) / 14
            self.data[i].rsi = 100 - (100 / (1 + (self.data[i].up_sma14 / self.data[i].down_sma14)))
            self.data[i].normalized_rsi = self.data[i].rsi / 100
        
        # Bollinger Bands
        for i in range(19, len(self.data)):
            self.data[i].sma20 = sum([d.close for d in self.data[i-19:i+1]]) / 20
            self.data[i].std_dev20 = math.sqrt(sum([(d.close - self.data[i].sma20)**2 for d in self.data[i-19:i+1]]) / 20)
            self.data[i].bb_upper = self.data[i].sma20 + (2 * self.data[i].std_dev20)
            self.data[i].bb_lower = self.data[i].sma20 - (2 * self.data[i].std_dev20)
            self.data[i].std_devs_out = (self.data[i].close - self.data[i].sma20) / self.data[i].std_dev20

        # On Balance Volume (COULD ADD)

        # Relative Volume
        for i in range(59, len(self.data)):
            self.data[i].vol_sma60 = sum([d.volume for d in self.data[i-59:i+1]]) / 60
            self.data[i].relative_vol = self.data[i].volume / self.data[i].vol_sma60

    def extract_and_calculate_all(self):
        self.extract_all_data()
        self.calculate_technical_indicators()

    def extract_and_calculate_technical(self):
        self.extract_price_data()
        self.calculate_technical_indicators()




# old method from new approach

# def extract_financial_data(self):
#     # MIGHT NOT BE ABLE TO FIND ENOUGH DATA FOR THIS
#     # This code gives annual financials for last 4 years or last 4 months.
#     # REF
#     # https://www.mattbutton.com/how-to-scrape-stock-upgrades-and-downgrades-from-yahoo-finance/
#     """
#     url = f"https://uk.finance.yahoo.com/quote/GOOG/financials"
#     page = requests.get(url, headers=STANDARD_HEADERS)
#     soup = BeautifulSoup(page.content,'html.parser')
#     script = soup.find('script', text=re.compile(r'root\.App\.main'))
#     json_text = re.search(r'^\s*root\.App\.main\s*=\s*({.*?})\s*;\s*$', script.string, flags=re.MULTILINE).group(1)
#     data = json.loads(json_text)
#     financials = data['context']['dispatcher']['stores']['QuoteSummaryStore']
#     """