# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:02:03 2014

@author: Prathmesh
"""
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import requests
from pprint import pprint
import csv
from time import sleep

s1='http://fantasy.premierleague.com/web/api/elements/'

players = []
for player_link in range(1,624,1):
    link = s1+""+str(player_link)
    r = requests.get(link)
    player =r.json() 
    players.append(player)
    sleep(1)
    
with open('/Users/Prathmesh/Documents/Data-Science-Course/Project/dec15_players.csv', 'wb') as f:  # Just use 'w' mode in 3.x
     w = csv.DictWriter(f,player.keys())
     w.writeheader()
     for player in players:        
         w.writerow(player)
