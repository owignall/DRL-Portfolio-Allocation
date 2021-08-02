"""
TO DO

COULD DO
    - Look at papers which use financial data for DRL
    - Extraction of financial data
    - Consider other article content extraction
"""

from constants import *

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import pandas as pd
import time
import threading
import re
import json
import math

E_MIN = 60
E_HOUR = 60 * E_MIN
E_DAY = 24 * E_HOUR
E_YEAR = 365 * E_DAY

MONTHS = {'jan': "01", 'feb': "02", 'mar': "03", 'apr': "04", 'may': "05", 'jun': "06", 'jul': "07", 'aug': "08", 'sep': "09",  'oct': "10", 'nov': "11", 'dec': "12"}

HEADER = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36'}

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
                page = requests.get(url, headers=HEADER)
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
        request = requests.get(file_link, headers=HEADER)
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
        page = requests.get(url, headers=HEADER)
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
            page = requests.get(url, headers=HEADER)
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
        page = requests.get(url, headers=HEADER)
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

# NEW APPROACH

class Stock:
    def __init__(self, name, code, ic_name):
        self.name = name
        self.code = code
        self.ic_name = ic_name
        self.data_start = int(time.time() - 8 * E_YEAR)
        self.data_end = int(time.time())
        self.df = self._initialize_df()

        # Parameters
        self.rs_decay = 0.9

        # Links
        self.investing_link =f"https://uk.investing.com/equities/{self.ic_name}"
        self.yahoo_link = f"https://uk.finance.yahoo.com/quote/{self.code}"
    
    def __str__(self):
        rep = f"{self.code} - {self.name}\n{self.df.describe()}"
        return rep

    def _initialize_df(self):
        """Extract price data and use this to create a new dataframe for the stock"""
        def _convert_type(value):
            data_types = [int, float, str]
            for d in data_types:
                try:
                    return d(value)
                except ValueError:
                    pass
        price_data_points = []
        period1 = str(self.data_start)
        period2 = str(self.data_end)
        interval = "1d"
        file_link = f"https://query1.finance.yahoo.com/v7/finance/download/{self.code}?period1={period1}&period2={period2}&interval={interval}"
        request = requests.get(file_link, headers=HEADER)
        content = str(request.content).replace("'", "").split("\\n")
        cols = content[0].split(",")
        for i in range(1, len(content)):
            new_point = map(_convert_type, content[i].split(","))
            price_data_points.append(new_point)
        if len(price_data_points) == 0:
            raise Exception("No data was retrieved by the extraction function")
        return pd.DataFrame(data=price_data_points, columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])

    def extract_financial_data(self):
        # MIGHT NOT BE ABLE TO FIND ENOUGH DATA FOR THIS
        # This code gives annual financials for last 4 years or last 4 months.
        # REF
        # https://www.mattbutton.com/how-to-scrape-stock-upgrades-and-downgrades-from-yahoo-finance/
        """
        url = f"https://uk.finance.yahoo.com/quote/GOOG/financials"
        page = requests.get(url, headers=HEADER)
        soup = BeautifulSoup(page.content,'html.parser')
        script = soup.find('script', text=re.compile(r'root\.App\.main'))
        json_text = re.search(r'^\s*root\.App\.main\s*=\s*({.*?})\s*;\s*$', script.string, flags=re.MULTILINE).group(1)
        data = json.loads(json_text)
        financials = data['context']['dispatcher']['stores']['QuoteSummaryStore']
        """

    def extract_investment_ranking_data(self):
        # REFERENCE
        # https://www.mattbutton.com/how-to-scrape-stock-upgrades-and-downgrades-from-yahoo-finance/
        url = f"https://uk.finance.yahoo.com/quote/{self.code}/analysis"
        page = requests.get(url, headers=HEADER)
        soup = BeautifulSoup(page.content,'html.parser')
        script = soup.find('script', text=re.compile(r'root\.App\.main'))
        json_text = re.search(r'^\s*root\.App\.main\s*=\s*({.*?})\s*;\s*$', script.string, flags=re.MULTILINE).group(1)
        data = json.loads(json_text)
        rankings_scraped = data['context']['dispatcher']['stores']['QuoteSummaryStore']['upgradeDowngradeHistory']['history']
        
        # DF APPROACH
        rankings_dict = dict()
        for r in rankings_scraped:
            investment_ranking = {'action':r['action'], 'from':r['fromGrade'], 'to': r['toGrade']}
            date = datetime.fromtimestamp(r['epochGradeDate']).strftime('%Y-%m-%d')
            if date in rankings_dict:
                rankings_dict[date].append(investment_ranking)
            else:
                rankings_dict[date] = [investment_ranking]
        # Rankings by date list
        rankings_by_date_list = []
        ranking_scores = []
        previous_score = 0
        for i in range(len(self.df)):
            if self.df.iloc[i]['date'] in rankings_dict:
                rankings = rankings_dict[self.df.iloc[i]['date']]
                values = [RANKING_VALUES[r['to']] if r['to'] in RANKING_VALUES else 0 for r in rankings]
                score = (self.rs_decay * previous_score) + sum(values)
                # print(score, values)
            else:
                rankings = []
                score = self.rs_decay * previous_score
                # print(score)
            previous_score = score
            rankings_by_date_list.append(rankings)
            ranking_scores.append(score)    
        self.df['rankings'] = rankings_by_date_list
        self.df['ranking_score'] = ranking_scores

    def extract_news_data(self, investing=True, google=True, threads=5):
        
        def _convert_date(news_date):
            # Converts format of date e.g. Apr 14, 2021 -> 2021-04-14
            if "ago" in news_date:
                return datetime.today().strftime('%Y-%m-%d')
            
            date_componets = news_date.replace(",", "").split(" ")
            return f"{date_componets[2]}-{MONTHS[date_componets[0].lower()]}-{date_componets[1]}"
        
        def _extract_from_investing_page(articles_dict, n, i, valid):
            # Adds 
            url = f"https://uk.investing.com/equities/{self.ic_name}-news/{str(n)}"
            page = requests.get(url, headers=HEADER)
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

                    new_article = {'title': title, 'link': link, 'author': author}
                    if date in articles_dict:
                        articles_dict[date].append(new_article)
                    else:
                        articles_dict[date] = [new_article]
                valid[i] = True
        
        if investing:
            print("Extracting Investing Articles")
            # Populate a dictionary with scraped articles
            articles_dict = dict()
            # Use threading to request pages and extract articles
            start_page = 1
            searching = True
            while searching:
                threads_list = []
                valid = [None] * threads
                for n in range(start_page, start_page + threads):
                    t = threading.Thread(target=_extract_from_investing_page, args=(articles_dict, n, n - 1 - start_page, valid))
                    t.start()
                    threads_list.append(t)
                for t in threads_list:
                    t.join()
                if not all(valid):
                    searching = False
                else:
                    start_page += threads
            
            articles = []
            for i in range(len(self.df)):
                if self.df.loc[i,'date'] in articles_dict:
                    articles.append(articles_dict[self.df.loc[i,'date']])
                else:
                    articles.append([])
        
        if google:
            # MIGHT BE BETTER SOURCE THAN INVESTING.COM
            pass

        # Add articles to DataFrame
        self.df['articles'] = articles
        
    def calculate_technical_indicators(self):
        """Calculates technical indicators and adds them to the DataFrame."""

        # MACD
        smoothing = 2
        ema12 = [None for _ in range(11)]
        ema26 = [None for _ in range(25)]
        macd = [None for _ in range(25)]
        ema12.append(sum(self.df.iloc[:12]['close'])/12)
        ema26.append(sum(self.df.iloc[:26]['close'])/26)
        for i in range(12, len(self.df)):
            if i > 11: # EMA 12
                ema12.append((self.df.iloc[i]['close'] * (smoothing / 13)) + (ema12[i-1] * (1 - (smoothing / 13))))
            if i > 25: # EMA 26
                ema26.append((self.df.iloc[i]['close'] * (smoothing / 27)) + (ema26[i-1] * (1 - (smoothing / 27))))
            if i > 24: # MACD
                macd.append(ema12[i] - ema26[i])
        signal_line = [None for _ in range(33)]
        signal_line.append(sum(macd[25:34]) / 9)
        for i in range(34, len(self.df)):
            signal_line.append((macd[i] * (smoothing / 10)) + (signal_line[i-1] * (1 - (smoothing / 10))))
        
        # RSI
        # Calculating close changes
        close_change = [None]
        for i in range(1, len(self.df)):
            close_change.append(self.df.iloc[i]['close'] - self.df.iloc[i-1]['close'])
        up_sma14 = [None for _ in range(14)]
        down_sma14 = [None for _ in range(14)]
        rsi = [None for _ in range(14)]
        normalized_rsi = [None for _ in range(14)]
        # Up and Down SMA 14
        for i in range(14, len(self.df)):
            # ups = list(filter(lambda x: x > 0, close_change[i-13:i+1]))
            ups = [x for x in close_change[i-13:i+1] if x > 0]
            up_sma14.append(1e-8 if len(ups) == 0 else sum(ups) / 14)
            # downs = list(map(lambda x:abs(x), filter(lambda x: x < 0, close_change[i-13:i+1])))
            downs = [abs(x) for x in close_change[i-13:i+1] if x < 0]
            down_sma14.append(1e-8 if len(downs) == 0 else sum(downs) / 14)
            rsi.append(100 - (100 / (1 + (up_sma14[i] / down_sma14[i]))))
            normalized_rsi.append(rsi[i] / 100)
        
        # Bollinger Bands
        sma20 = [None for _ in range(19)]
        std_dev20 = [None for _ in range(19)]
        bb_upper = [None for _ in range(19)]
        bb_lower = [None for _ in range(19)]
        std_devs_out = [None for _ in range(19)]
        for i in range(19, len(self.df)):
            sma20.append(sum(self.df.iloc[i-19:i+1]['close']) / 20)
            std_dev20.append(math.sqrt(sum([(d - sma20[i])**2 for d in self.df.loc[:,'close'].iloc[i-19:i+1]]) / 20))
            bb_upper.append(sma20[i] + (2 * std_dev20[i]))
            bb_lower.append(sma20[i] - (2 * std_dev20[i]))
            std_devs_out.append((self.df.iloc[i]['close'] - sma20[i]) / std_dev20[i])

        # Relative Volume
        vol_sma60 = [None for _ in range(59)]
        relative_vol = [None for _ in range(59)]
        for i in range(59, len(self.df)):
            vol_sma60.append(sum(self.df.loc[i-59:i,'volume']) / 60)
            relative_vol.append(self.df.iloc[i]['volume'] / vol_sma60[i])

        generated_lists = {
            'ema12': ema12, 'ema26': ema26, 'macd': macd, 'signal_line': signal_line, 'close_change': close_change, 
            'up_sma14': up_sma14, 'down_sma14': down_sma14, 'rsi': rsi, 'normalized_rsi': normalized_rsi, 'sma20': sma20,
            'std_dev20': std_dev20, 'bb_upper': bb_upper, 'bb_lower': bb_lower, 'std_devs_out': std_devs_out, 
            'vol_sma60': vol_sma60, 'relative_vol': relative_vol}
        
        # Check that the list lengths are correct
        if not all([len(self.df) == len(l) for l in generated_lists.values()]):
            raise Exception("One of the lists in technical indicators calculation was wrong length")
        
        # Add technical indicator lists to the dataframe
        for key, value in generated_lists.items():
            self.df[key] = value
    
    def extract_and_calculate_all(self, verbose=True):
        if verbose: print("Extracting investment ranking data")
        self.extract_investment_ranking_data()
        if verbose: print("Extracting news data")
        self.extract_news_data()
        if verbose: print("Calculating technical indicators")
        self.calculate_technical_indicators()
    
    def extract_and_calculate_basic(self, verbose=True):
        if verbose: print("Extracting investment ranking data")
        self.extract_investment_ranking_data()
        if verbose: print("Calculating technical indicators")
        self.calculate_technical_indicators()

    def save_as_excel(self):
        self.df.to_excel(f"{self.code}.xlsx")



if __name__ == "__main__":
    s = Stock("Apple", "AAPL", "apple-computer-inc")
    # s.extract_price_data()
    s.calculate_technical_indicators()
    # s.extract_news_data()
    # print(s.loc[:,'inv_articles'])
    # for a in s.df['inv_articles']:
    

    






# # CODE TO CHECK DF AND DAILY DATA MATCHES
# for i in range(len(self.data)):
#     # print(f"{getattr(self.data[i], 'rsi')} ... {self.df.loc[i,'rsi']}")
#     try:
#         for k in generated_lists:
#             if (getattr(self.data[i], k) != self.df.loc[i,k]): # and k not in ['rsi', 'normalized_rsi', 'down_sma14', 'up_sma14']:
#                 if not (math.isnan(self.df.loc[i,k]) and getattr(self.data[i], k) == None):
#                     if abs(getattr(self.data[i], k) - self.df.loc[i,k]) > 1e-4:
#                         print(k, "does not match.")
#                         print(f"{getattr(self.data[i], k)} != {self.df.loc[i,k]}")

#                         raise ValueError("Values don't match")

#         for k in ['close', 'volume', 'adj_close', 'date']:
#             if (getattr(self.data[i], k) != self.df.loc[i,k]) and k not in ['rsi', 'normalized_rsi', 'down_sma14', 'up_sma14']:
#                 if not (math.isnan(self.df.loc[i,k]) and getattr(self.data[i], k) == None):
#                     print(k, "does not match.")
#                     print(f"{getattr(self.data[i], k)} != {self.df.loc[i,k]}")

#     except ValueError as e:
#         print(k)
#         print(f"{getattr(self.data[i], k)} != {self.df.loc[i,k]}")
#         raise e