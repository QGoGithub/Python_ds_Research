# A Python module for retrieving stock data using the
# yahoo finance API

# requires the following installations:
#pip install cvxopt
#pip install numpy
#pip install scipy
#pip install pandas
#pip install patsy
#pip install statsmodels

import csv
import urllib2
from collections import defaultdict
import numpy
from statsmodels.stats.correlation_tools import cov_nearest

# Retrieves the stock quote for the given symbol
# from Yahoo Finance as a float.
# Input:  symbol - stock symbol as a string
# Output: price  - latest trade price from yahoo finance

def get_stock_quote(symbol):
	BASE_URL = 'http://download.finance.yahoo.com/d/quotes.csv?s='
	ID = symbol
	close_prop = '&f=l1'
	SUFFIX = '&e=.csv'
	url = "%s%s%s%s" % (BASE_URL, ID, close_prop, SUFFIX)
	price = float(urllib2.urlopen(url).read().strip())
	return price

# Downloads the stock history for the given symbol,
# for the given date range, as a csv file.
# Input: symbol   - stock symbol as a string
#        start    - start date in the form 'mm/dd/yyyy'
#        end      - end date in the form 'mm/dd/yyyy'
#        outfile  - output filename, e.g. 'out.csv'
#        interval - trading interval; either d, w, m (daily, weekly, monthl7)

def csv_quote_history(symbol, start, end, outfile, interval='d'):
	response = _quote_history(symbol, start, end, interval)
	with open(outfile, 'wb') as f:
		csv_reader = csv.reader(response)
		csv_writer = csv.writer(f)
		for row in csv_reader:
			csv_writer.writerow(row)

# Gives the stock history for the given symbol,
# for the given date range, as a dictionary.
# Output: keys: ['High', 'Adj Close', 'Volume', 'Low', 'Date', 'Close', 'Open']
#         values: list
def quote_history_dict(symbol, start, end, interval='m'):
	history = defaultdict(lambda: [])
	response = _quote_history(symbol, start, end, interval)
	dreader = csv.DictReader(response)
	for row in dreader:
		for key in row.iterkeys():
			history[key].insert(0, row[key])
	return history

def _quote_history(symbol, start, end, interval):
	BASE_URL = 'http://ichart.yahoo.com/table.csv?s='
	ID = symbol
	sm, sd, sy = start.split('/')
	em, ed, ey = end.split('/')
	url = "%s%s&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=%s" % (BASE_URL, ID, (int(sm)-1), int(sd), int(sy), (int(em)-1), int(ed), int(ey), interval)
	response = urllib2.urlopen(url)
	return response

def get_prices(symbol, start, end, interval='m'):
	history = quote_history_dict(symbol, start, end, interval)
	prices = map(lambda x: round(float(x),2), history['Close'])
	prices[0] = round(float(history['Open'][0]),2)
	return prices

def get_returns(symbol, start, end, interval='m'):
	history = quote_history_dict(symbol, start, end, interval)
	prices = map(lambda x: round(float(x),2), history['Close'])
	prices[0] = round(float(history['Open'][0]),2)
	returns = map(lambda (x, y): (y/x)-1, zip(prices[0:-1], prices[1:]))
	return returns

def get_yr_returns(symbol, start, end):
	history = quote_history_dict(symbol, start, end, 'm')
	prices = map(lambda x: round(float(x),2), history['Close'])
	prices[0] = round(float(history['Open'][0]),2)
	prices.insert(0, prices[0])
	returns = map(lambda (x, y): (y/x)-1, zip(prices[0::12][:-1], prices[12::12]))
	return returns

def avg_return(symbol, start, end, interval='m'):
	if interval=='y':
		return numpy.mean(get_yr_returns(symbol, start, end))
	else:
		return numpy.mean(get_returns(symbol, start, end, interval))

def cov_matrix(symbols, start, end, interval='m'):
	if interval=='y':
		data = [numpy.array(get_yr_returns(s, start, end)) for s in symbols]
	else:
		data = [numpy.array(get_returns(s, start, end, interval)) for s in symbols]
	x = numpy.array(data)
	return cov_nearest(numpy.cov(x))
