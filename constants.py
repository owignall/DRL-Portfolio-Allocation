import os

E_MIN = 60
E_HOUR = 60 * E_MIN
E_DAY = 24 * E_HOUR
E_YEAR = 365 * E_DAY

# Data Scraping

MONTHS = {'jan': "01", 'feb': "02", 'mar': "03", 'apr': "04", 'may': "05", 'jun': "06", 'jul': "07", 'aug': "08", 'sep': "09",  'oct': "10", 'nov': "11", 'dec': "12"}

STANDARD_HEADERS = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36'}

GOOGLE_HEADERS = {
    'cookie': 'CONSENT=YES+cb.20210727-07-p1.en+FX+232; 1P_JAR=2021-07-29-20; NID=220=IejjYDNZoeiOPhcOHqIC6lQoOU1wblHSv2dlAMLzR9daG7JFjh43rzykeiGRH-mZZ8ZCOPwCXeOxli1CxVkHN6jser1LxE41gkABFLccZsNaGgnajJPsD9LN_4-byjEdPJsAv_k1gnzFzXtjukkroowbISEIDnSQiplLI-h97Us1abCR',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55'
}

folder_contents = os.listdir("other/chromedriver")
files = [file for file in folder_contents if file.startswith("chromedriver")][0]
file_name = files[0] if len(files) > 0 else ""
WEBDRIVER_PATH = f"other/chromedriver/{file_name}"

# Sentiment Analysis

RANKING_VALUES = {
    "Cautious": -1,
    "Conviction Buy": 1,
    "Below Average": -1,
    "Above Average": 1,
    "Underperformer": -1,
    "Overperformer": 1,
    "Fair Value": 0,
    "": 0,
    "Mixed": 0,
    "Reduce": -1,
    "Hold Neutral": 0,
    "Long-Term Buy": 1,
    "Equal-Weight": 0,
    "Buy": 1,
    "Strong Buy": 1,
    "Hold": 0,
    "In-Line": 0,
    "Perform": 0,
    "Negative": -1,
    "Positive": 1,
    "Overweight": 1,
    "Sector Outperform": 1,
    "Top Pick": 1,
    "Accumulate": 1,
    "Market Perform": 0,
    "Market Outperform": 1,
    "Sector Perform": 0,
    "In-line": 0,
    "Equal-weight": 0,
    "Neutral": 0,
    "Average": 0,
    "Peer Perform": 1,
    "Underperform": -1,
    "Market Underperform": -1,
    "Sector Weight": 0,
    "Sector Underperform": -1,
    "Market Weight": 0,
    "Long-term Buy": 1,
    "Underweight": -1,
    "Sell": -1,
    "Outperform": 1
}

CHANGE_VALUES = {
    'init': 0, 
    'reit': 0, 
    'down': -1, 
    'up': 1, 
    'main': 0
}

HF_LABEL_VALUES = {
    'NEGATIVE' : -1,
    'POSITIVE' : 1
}

# Stock Arguments

SNP_500_TOP_100 = [
    ['Apple', 'AAPL', 'apple-computer-inc'],
    ['Microsoft', 'MSFT', 'microsoft-corp'],
    ['Amazon', 'AMZN', 'amazon-com-inc'],
    ['Alphabet', 'GOOG', 'google-inc-c'],
    ['Facebook', 'FB', 'facebook-inc'],
    ['Berkshire Hathaway', 'BRK-B', 'berkshire-hathaway'],     
    ['Tesla Motors', 'TSLA', 'tesla-motors'],
    ['Visa', 'V', 'visa-inc'],
    ['Nvidia', 'NVDA', 'nvidia-corp'],
    ['J P Morgan', 'JPM', 'jp-morgan-chase'],
    ['Johnson and Johnson', 'JNJ', 'johnson-johnson'],
    ['Wal-Mart', 'WMT', 'wal-mart-stores'],
    ['United Health', 'UNH', 'united-health-group'],
    ['Mastercard', 'MA', 'mastercard-cl-a'],
    ['Bank of America', 'BAC', 'bank-of-america'],
    ['Procter and Gamble', 'PG', 'procter-gamble'],
    ['Home Depot', 'HD', 'home-depot'],
    ['Walt Disney', 'DIS', 'disney'],
    ['Adobe', 'ADBE', 'adobe-sys-inc'],
    ['Comcast', 'CMCSA', 'comcast-corp-new'],
    ['Exxon Mobil', 'XOM', 'exxon-mobil'],
    ['Coca-Cola', 'KO', 'coca-cola-co'],
    ['Verizon Communications', 'VZ', 'verizon-communications'],
    ['Salesforce', 'CRM', 'salesforce-com'],
    ['Intel', 'INTC', 'intel-corp'],
    ['Netflix', 'NFLX', '"netflix,-inc."'],
    ['Oracle', 'ORCL', 'oracle-corp'],
    ['Cisco', 'CSCO', 'cisco-sys-inc'],
    ['Pfizer', 'PFE', 'pfizer'],
    ['Eli Lilly', 'LLY', 'eli-lilly-and-co'],
    ['AT&T', 'T', 'at-t'],
    ['NIKE', 'NKE', 'nike'],
    ['PepsiCo', 'PEP', 'pepsico'],
    ['AbbVie', 'ABBV', 'abbvie-inc'],
    ['Chevron', 'CVX', 'chevron'],
    ['Abbott Labs', 'ABT', 'abbott-laboratories'],
    ['Merck', 'MRK', 'merck---co'],
    ['Broadcom', 'AVGO', 'avago-technologies'],
    ['Thermo Fisher Scientific', 'TMO', 'thermo-fisher-sc'],
    ['Danaher', 'DHR', 'danaher-corp'],
    ['T-Mobile US', 'TMUS', 'metropcs-communications'],
    ['Accenture', 'ACN', 'accenture-ltd'],
    ['Wells Fargo', 'WFC', 'wells-fargo'],
    ['United Parcel Service', 'UPS', 'united-parcel'],
    ["McDonald's", 'MCD', 'mcdonalds'],
    ['Texas Instruments', 'TXN', 'texas-instru'],
    ['Costco', 'COST', 'costco-whsl-corp-new'],
    ['Medtronic', 'MDT', 'medtronic'],
    ['Morgan Stanley', 'MS', 'morgan-stanley'],
    ['Philip Morris International', 'PM', 'philip-morris-intl'],
    ['Qualcomm', 'QCOM', 'qualcomm-inc'],
    ['Bristol-Myers Squibb', 'BMY', 'bristol-myer-squiib'],
    ['Honeywell International', 'HON', 'honeywell-intl'],
    ['Linde', 'LIN', 'linde-plc'],
    ['NextEra Energy', 'NEE', 'nextera-energy-inc'],
    ['Union Pacific', 'UNP', 'union-pacific'],
    ['Citigroup', 'C', 'citigroup'],
    ['Boeing', 'BA', 'boeing-co'],
    ['Amgen', 'AMGN', 'amgen-inc'],
    ['Lowes Co.', 'LOW', 'lowes-companies'],
    ['Charles Schwab', 'SCHW', 'charles-schwab'],
    ['Raytheon Technologies', 'RTX', 'united-tech'],
    ['Intuit', 'INTU', 'intuit'],
    ['Charter Communications', 'CHTR', 'charter-communications'],
    ['Starbucks', 'SBUX', 'starbucks-corp'],
    ['BlackRock', 'BLK', '"blackrock,-inc.-c"'],
    ['IBM', 'IBM', 'ibm'],
    ['American Express', 'AXP', 'american-express'],
    ['American Tower REIT', 'AMT', 'amer-tower-corp'],
    ['Applied Materials', 'AMAT', 'applied-matls-inc'],
    ['Goldman Sachs', 'GS', 'goldman-sachs-group'],
    ['Caterpillar', 'CAT', 'caterpillar'],
    ['Target', 'TGT', 'target'],
    ['General Electric', 'GE', 'general-electric'],
    ['3M', 'MMM', '3m-co'],
    ['CVS Health', 'CVS', 'cvs-corp'],
    ['Estee Lauder Companies', 'EL', 'estee-lauder'],
    ['Lockheed Martin', 'LMT', 'lockheed-martin'],
    ['ServiceNow', 'NOW', 'servicenow-inc'],
    ['Intuitive Surgical', 'ISRG', 'intuitive-surgical-inc'],
    ['Advanced Micro Devices', 'AMD', 'adv-micro-device'],
    ['Deere and Co.', 'DE', 'deere---co'],
    ['Stryker', 'SYK', 'stryker'],
    ['S&P Global', 'SPGI', 'mcgraw-hill'],
    ['Booking Holdings', 'BKNG', 'priceline.com-inc'],
    ['Anthem', 'ANTM', 'wellpoint-inc'],
    ['Fidelity National Information', 'FIS', 'fidelity-natl-in'],
    ['Prologis', 'PLD', 'prologis'],
    ['Zoetis', 'ZTS', 'zoetis-inc'],
    ['Lam Research', 'LRCX', 'lam-research-corp'],
    ['Mondelez', 'MDLZ', 'mondelez-international-inc'],
    ['Micron', 'MU', 'micron-tech'],
    ['Altria', 'MO', 'altria-group'],
    ['US Bancorp', 'USB', 'us-bancorp'],
    ['General Motors', 'GM', 'gen-motors'],
    ['Crown Castle Intl', 'CCI', 'crown-castle-int'],
    ['Gilead Sciences', 'GILD', 'gilead-sciences-inc'],
    ['Automatic Data Processing', 'ADP', 'auto-data-process']
]