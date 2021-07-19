from stock_data_extraction import *
from stock_storage import *

with open("SNP500_stocks.txt", "r") as file:
    table = [stock.split("\t") for stock in file.read().split("\n")]
    headers = table[0]
    stock_arguments = table[1:]


# s = Stock(*stock_arguments[5])

# s.extract_price_data()
# print(s.data)
# s.extract_and_calculate_technical()

for i in range(len(stock_arguments)):
    s = Stock(*stock_arguments[i])
    s.extract_and_calculate_technical()
    save_stock(s, folder="saved_stocks_technical_only")
    # stocks.append(s)
    print(s)

# for i in range(100):
#     d = stocks[0].data[i]
#     print(d.date, d.sma20 ,d.std_dev20, d.bb_upper, d.bb_lower)
#     # print(i, d.close_change, d.up_sma14, d.down_sma14, d.rsi)
#     # print(i, d.ema12, d.ema26, d.macd, d.signal_line)

