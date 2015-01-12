# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 20:09:42 2014

@author: Prathmesh
"""
#Importing Various Python modules 

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import requests
from pprint import pprint
import csv
from time import sleep
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error




#Sample API for one Player
r = requests.get('http://fantasy.premierleague.com/web/api/elements/266/')
top = r.text    # unicode text string
top = r.json()  # dictionary
pprint(top)



#for loop for scrapping api for all the players
#Reviewers Don't need to Run this Step as I've already got the data in CSV
#This is a time consuming process as this step is accessing web api

s1='http://fantasy.premierleague.com/web/api/elements/'

players = []
for player_link in range(1,624,1):
    link = s1+""+str(player_link)
    r = requests.get(link)
    player =r.json() 
    players.append(player)
    sleep(1)

#Writing the Data from the API into a CSV file
with open('/Users/Prathmesh/Documents/Data-Science-Course/Project/dict_output1.csv', 'wb') as f:  # Just use 'w' mode in 3.x
     w = csv.DictWriter(f,player.keys())
     w.writeheader()
     for player in players:        
         w.writerow(player)

# Reading CSV into Pandas DataFrame
# dict_output Refers to Player data from a November Week
players_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/dict_output1.csv',index_col='web_name', na_filter=False)

#Observing the Data for Exploration
players_df.head()
players_df.tail()
players_df.dtypes
players_df.iloc[0,:]

#Some random plotting to determine which attributes to use in the final model
# I did this step for various variables but it didn't reveal much to me.
plt.scatter(players_df.value_form, players_df.points_per_game, alpha=0.3)
plt.scatter(players_df.value_season, players_df.points_per_game, alpha=0.3) 
plt.scatter(forwards_df.bps, forwards_df.points_per_game, alpha=0.3) 
plt.xlabel("BPS")
plt.ylabel("Points Per Game")

#Filtering the data only for Forwards
players_df[players_df.type_name=='Forward'].to_csv('../data/players_updated.csv')

forwards_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/players_updated.csv',index_col='web_name', na_filter=False)

#Exploring the data in the forwards data frame
forwards_df.describe()
forwards_df.head()
forwards_df.tail()
forwards_df.dtypes
forwards_df.points_per_game.describe()
forwards_df.points_per_game.value_counts()
forwards_df.isnull()

#Creating  Initial Linear Model for Forwards

forwards_model = smf.ols(formula='event_total ~ selected_by + value_form + value_season + form + ea_index + bps', data=forwards_df).fit()
forwards_model.summary()

# Exploring Multi-collinearity between Variables
columns = ['event_total', 'selected_by', 'value_form', 'value_season', 'form','ea_index','bps']
pd.scatter_matrix(forwards_df[columns])

corr_matrix = np.corrcoef(forwards_df[columns].T)
sm.graphics.plot_corr(corr_matrix, xnames=columns)

# Its obvious from the Correlation Matrix that there is correlation between bps - ea_index & form - value_form
# Hence removing bps & value_from model and exploring
forwards_model = smf.ols(formula='event_total ~ selected_by + form + value_season + ea_index', data=forwards_df).fit()
forwards_model.summary()


#Trying Interaction terms to handle Correlation
# Interaction using * between form & value_form ; form and value_season
interaction_model = smf.ols(formula='event_total ~ selected_by + value_form*form + value_season*form + ea_index', data=forwards_df).fit()
interaction_model.summary()

# Removing players who have played 0 minutes till now
forwards_df[forwards_df.minutes> 0].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards.csv')
regular_forwards_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards.csv',index_col='web_name', na_filter=False)

# Running all features models for data set without fringe players
# This inculdes all the possible features, just to study the importance of each feature
all_features_model = smf.ols(formula='event_total ~ selected_by + total_points+ chance_of_playing_this_round + value_form + value_season + form + transfers_out_event+ transfers_in_event + points_per_game + minutes +ea_index + bps', data=regular_forwards_df).fit()
all_features_model.summary()


# Performing Cross Evaluation of the Model

# Creating two interaction terms based on what we learned from multi-collinearity matrix
regular_forwards_df['interaction_term1'] = regular_forwards_df.value_form * regular_forwards_df.form
regular_forwards_df['interaction_term2'] = regular_forwards_df.value_season * regular_forwards_df.form


cols = ['points_per_game','now_cost','selected_by', 'interaction_term1','interaction_term2', 'ea_index']


X = regular_forwards_df[cols]

y = regular_forwards_df.event_total

lm = LinearRegression()

# Also tried RandomForestClassifier but it didn't produce better model as compared to Linear Regression
#rf = RandomForestClassifier(n_estimators=100) 
scores = cross_val_score(lm, X, y, cv=5, scoring='mean_squared_error')

#Calculating Root Mean Squared Error
np.sqrt(-scores)
np.mean(np.sqrt(-scores)) #RMSE for Nov Model 2.0596


# Creating the Same Model for Dec 1 Data Set

players_dec1_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/dec1_players.csv',index_col='web_name', na_filter=False)

players_dec1_df[players_dec1_df.type_name=='Forward'].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/forwardplayers_dec1.csv')

forwards_dec1_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/forwardplayers_dec1.csv',index_col='web_name', na_filter=False)

forwards_dec1_df[forwards_dec1_df.minutes> 0].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards_dec1.csv')
regular_forwards_dec1_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards_dec1.csv',index_col='web_name', na_filter=False)


regular_forwards_dec1_df['interaction_term1'] = regular_forwards_dec1_df.value_form * regular_forwards_dec1_df.form
regular_forwards_dec1_df['interaction_term2'] = regular_forwards_dec1_df.value_season * regular_forwards_dec1_df.form


cols = ['points_per_game','now_cost','selected_by', 'interaction_term1','interaction_term2', 'ea_index']


X2 = regular_forwards_dec1_df[cols]

y2 = regular_forwards_dec1_df.event_total

lm = LinearRegression()
scores = cross_val_score(lm, X2, y2, cv=5, scoring='mean_squared_error')
np.sqrt(-scores)
np.mean(np.sqrt(-scores)) #RMSE for Dec1 Model 1.6575


# Creating the Same Model for Dec 4 Data Set

players_dec4_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/dec4_players.csv',index_col='web_name', na_filter=False)

players_dec4_df[players_df.type_name=='Forward'].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/forwardplayers_dec4.csv')

forwards_dec4_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/forwardplayers_dec4.csv',index_col='web_name', na_filter=False)

forwards_dec4_df[forwards_dec4_df.minutes> 0].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards_dec4.csv')
regular_forwards_dec4_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards_dec4.csv',index_col='web_name', na_filter=False)


regular_forwards_dec4_df['interaction_term1'] = regular_forwards_dec4_df.value_form * regular_forwards_dec4_df.form
regular_forwards_dec4_df['interaction_term2'] = regular_forwards_dec4_df.value_season * regular_forwards_dec4_df.form


cols = ['points_per_game','now_cost','selected_by', 'interaction_term1','interaction_term2', 'ea_index']


X3 = regular_forwards_dec4_df[cols]

y3 = regular_forwards_dec4_df.event_total

lm = LinearRegression()
scores = cross_val_score(lm, X3, y3, cv=5, scoring='mean_squared_error')
np.sqrt(-scores)
np.mean(np.sqrt(-scores)) #RMSE for Dec4 Model 2.4378


# Creating the Same Model for Dec 9 Data Set
players_dec9_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/dec9_players.csv',index_col='web_name', na_filter=False)

players_dec9_df[players_df.type_name=='Forward'].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/forwardplayers_dec9.csv')

forwards_dec9_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/forwardplayers_dec9.csv',index_col='web_name', na_filter=False)

forwards_dec9_df[forwards_dec9_df.minutes> 0].to_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards_dec9.csv')
regular_forwards_dec9_df = pd.read_csv('/Users/Prathmesh/Documents/Data-Science-Course/Project/regular_forwards_dec9.csv',index_col='web_name', na_filter=False)


regular_forwards_dec9_df['interaction_term1'] = regular_forwards_dec9_df.value_form * regular_forwards_dec9_df.form
regular_forwards_dec9_df['interaction_term2'] = regular_forwards_dec9_df.value_season * regular_forwards_dec9_df.form


cols = ['points_per_game','now_cost','selected_by', 'interaction_term1','interaction_term2', 'ea_index']


X4 = regular_forwards_dec9_df[cols]

y4 = regular_forwards_dec9_df.event_total

lm = LinearRegression()
scores = cross_val_score(lm, X4, y4, cv=5, scoring='mean_squared_error')
np.sqrt(-scores)
np.mean(np.sqrt(-scores)) #RMSE for Dec9 Model 3.14755


# Training the model on the November Data Set & Testing for Dec 1, Dec4 & Dec 9 Data 

cols = ['points_per_game','now_cost','selected_by', 'interaction_term1','interaction_term2', 'ea_index']
X = regular_forwards_df[cols]
y = regular_forwards_df.event_total
lm = LinearRegression()
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(X,y)
lm.fit(X, y) # fitting the linear regression on Nov values of X & Y

# Testing and Predicting for Dec 1 Data set
preds = lm.predict(X2) #X for Dec 1
# calc RMSE to compare preds vs y for Dec 1
rms = np.sqrt(mean_squared_error(y2, preds))
rms # RMSE for preds for Dec1 data set 2.3746


# Testing and Predicting for Dec 4 Data set
preds = lm.predict(X3) #X for Dec 4
# calc RMSE to compare preds vs y for Dec 4
rms = np.sqrt(mean_squared_error(y3, preds))
rms # RMSE for preds for Dec4 data set 2.8351

# Testing and Predicting for Dec 9 Data set
preds = lm.predict(X4) #X for Dec 9
# calc RMSE to compare preds vs y for Dec 9
rms = np.sqrt(mean_squared_error(y4, preds))
rms # RMSE for preds for Dec9 data set 3.0285



