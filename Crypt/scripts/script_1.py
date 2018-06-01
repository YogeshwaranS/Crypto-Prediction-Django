# import json
import simplejson
import numpy as np
import os
import pandas as pd
import urllib2
import httplib

httplib.HTTPConnection._http_vsn = 10
httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'
# connect to poloniex's API
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1356998100&end=9999999999&period=300'

# parse json returned from the API to Pandas DF
openUrl = urllib2.urlopen(url)
r = openUrl.read()
httplib.HTTPConnection._http_vsn = 11
httplib.HTTPConnection._http_vsn_str = 'HTTP/1.1'
openUrl.close()
d = simplejson.loads(r.decode())
df = pd.DataFrame(d)

original_columns = [u'close', u'date', u'high', u'low', u'open']
new_columns = ['Close', 'Timestamp', 'High', 'Low', 'Open']
df = df.loc[:, original_columns]
df.columns = new_columns
folder = os.getcwd() + '/Crypt/scripts/data/'
df.to_csv(folder + 'bitcoin2015to2017.csv', index=None)


def init_me():
    print ("script 1 running")
