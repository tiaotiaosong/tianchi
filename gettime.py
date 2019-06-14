import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time

wifi = []

data = pd.read_csv('m_690data.csv')
for array in data.values:
    array = array[2].split()
    d = array[0].split('-')
    t = array[1].split(':')
    da1 = d+t
    da = [int(x) for x in da1]
    print datetime(da[0],da[1],da[2],da[3],da[4])
    break
print datetime.now()
   # wifi.append(datetime(d.append(t)))
#print wifi
