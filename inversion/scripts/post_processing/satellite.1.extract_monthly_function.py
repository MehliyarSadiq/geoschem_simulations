#!/usr/bin/env python
# coding: utf-8
from tools_new import *

year = 2017
month = 5
# file names and paths
input_file = '/geos/u73/msadiq/satellite/oco-2/OCO2_b91_10sec_GOOD_r24.nc4'

for month in range(1,13):
    extract_oco2_monthly(year, month, input_file)
    flatten_oco2_monthly(year, month)

