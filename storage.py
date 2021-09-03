"""
These functions are used to store and retrieve Stock objects which may contain stock data. A
function is also included to extract all stock objects from a given folder.
"""

import dill
from datetime import datetime
from os import listdir


def save_dill_object(my_object, file_name):
    with open(file_name, "wb") as file:
        dill.dump(my_object, file)

def retrieve_dill_object(file_name):
    with open(file_name, "rb") as file:
        my_object = dill.load(file)
    return my_object

def save_stock(stock, folder):
    code = stock.code
    date = datetime.today().strftime('%Y-%m-%d')
    save_dill_object(stock, f"{folder}/{code}_{date}.dill")

def retrieve_stocks_from_folder(folder):
    stocks = []
    files = listdir(folder)
    for stock_file in files:
        s = retrieve_dill_object(f"{folder}/{stock_file}")
        stocks.append(s)
    return stocks