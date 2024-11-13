# EvoInvest

A tool that uses a multiobjective genetic algorithm to optimize an investment portfolio's expected return andassociated risk. Built with [streamlit](https://pypi.org/project/streamlit/).

## Methodology

* Extract information from 96 selected tickers (see Tickers section) from Yahoo! Finance using [yfinance](https://pypi.org/project/yfinance/) and [pandas](https://pypi.org/project/pandas/).

* Compute the log returns and covariance matrix using [numpy](https://pypi.org/project/numpy/).

* Run Multiobjective Genetic Algorithm to find a group of solutions that approximate the Pareto front. The objective functions are as follows:
  * $\text{max} \sum_{i=n}^{n}w_i \mu_i$
    * $w_i$: Weight associated to stock $i$
    * $\mu_i$: Log return from stock $i$

  * $\text{min } {\sigma = W \Sigma W^T}$
    * $\sigma$: Standard deviation (risk) associated to portfolio
    * $W$: Weight vector of the portfolio
    * $\Sigma$: Covariance matrix of the stocks
   
* Plot the pareto front and weights of an individual solution using [matplotlib](https://pypi.org/project/matplotlib/).

## Disclaimer

This project is intended for educational purposes only and is not for profit. The information and tools provided should be used with caution, as the results may not always be accurate or precise. Users are encouraged to verify the outputs and apply their own judgement before making any decisions based on the results. The creators of this tool cannot, in any way, be held liable for any losses users may suffer. This tool is not intended to replace professional financial advice.

## Tickers: 

```
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ',
       'V', 'UNH', 'JPM', 'PG', 'HD', 'MA', 'BAC', 'XOM', 'PFE', 'KO', 'MRK',
       'PEP', 'ABBV', 'CSCO', 'AVGO', 'TMO', 'LLY', 'NFLX', 'COST', 'ORCL',
       'NKE', 'DIS', 'ABT', 'CRM', 'ACN', 'TXN', 'VZ', 'ADBE', 'MCD', 'HON',
       'WMT', 'INTC', 'QCOM', 'UPS', 'MS', 'LIN', 'CVX', 'T', 'UNP', 'LOW',
       'SBUX', 'NEE', 'PM', 'AMGN', 'DHR', 'BA', 'BMY', 'SPGI', 'CAT', 'GS',
       'RTX', 'BLK', 'SCHW', 'AMT', 'IBM', 'MDLZ', 'ISRG', 'NOW', 'PLD',
       'INTU', 'DE', 'BKNG', 'MO', 'MDT', 'CI', 'CB', 'LMT', 'MMC', 'TGT',
       'GE', 'PYPL', 'ELV', 'ZTS', 'ADI', 'SYK', 'DUK', 'EW', 'SO', 'CL',
       'VRTX', 'FIS', 'ICE', 'APD', 'ETN', 'SHW', 'CCI', 'GM']
```
