import csv
import os
from datetime import datetime


class TradeLogger:
    def __init__(self, filename='trades.csv'):
        self.filename = filename
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['时间', '交易对', '方向', '信号值', '数量', '价格', '资金', '是否违规'])

    def log(self, time, symbol, action, signal, quantity, price, capital, violation=False):
        record = [
            time,
            symbol,
            action,
            round(signal, 4) if signal is not None else '',
            round(quantity, 6),
            round(price, 2),
            round(capital, 2),
            '是' if violation else '否'
        ]
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(record)
