import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

olympics = pd.read_csv('data/athlete_events.csv', header = 0)
olympics.head()

# See which country has accumulated the most gold medals
medal_ranks = olympics[olympics['Medal'] == 'Gold'].groupby(['NOC', 'Year', 'Sport', 'Event', 'Season'])
medal_ranks = medal_ranks.first()
medal_ranks = medal_ranks.reset_index()
medal_ranks['NOC'].value_counts()

# See which sport the USA has earned the most medals in
usa = olympics[olympics['NOC']=='USA']
medal_sports_usa = usa[usa['Medal'] == 'Gold'].groupby(['Year', 'Sport', 'Event'])
medal_sports_usa = medal_sports_usa.first()
medal_sports_usa = medal_sports_usa.reset_index()
medal_sports_usa['Sport'].value_counts()

# Can devide sports by which season they occurred in and see which winter sport the US amassed the most medals in
usa_winter = usa[usa['Season'] == 'Winter']
medal_sports_winter = usa_winter[usa_winter['Medal'] == 'Gold'].groupby(['Year', 'Sport', 'Event'])
medal_sports_winter = medal_sports_winter.first()
medal_sports_winter = medal_sports_winter.reset_index()
medal_sports_winter['Sport'].value_counts()

# Find which country accrued the most gold in the winter games
winter = olympics[olympics['Season'] == 'Winter']
winter_ranks = winter[winter['Medal'] == 'Gold'].groupby(['NOC', 'Year', 'Sport', 'Event'])
winter_ranks = winter_ranks.first()
winter_ranks = winter_ranks.reset_index()
winter_ranks['NOC'].value_counts()

# Identify which winter sport Norway has performed the best in
nor = olympics[olympics['NOC']=='NOR']
nor_winter = nor[nor['Season'] == 'Winter']
medal_sports_winter = nor_winter[nor_winter['Medal'] == 'Gold'].groupby(['Year', 
                                                                         'Sport', 
                                                                         'Event'])
medal_sports_winter = medal_sports_winter.first()
medal_sports_winter = medal_sports_winter.reset_index()
medal_sports_winter['Sport'].value_counts()

# Visualize medal distributions using the crosstab function from pandas
nor = olympics[olympics['NOC']=='NOR']
nor_winter = nor[nor['Season'] == 'Winter']
medal_sports_winter = nor_winter.groupby(['Year', 'Sport', 'Event'])
medal_sports_winter = medal_sports_winter.first()
medal_sports_winter = medal_sports_winter.reset_index()
medal_sports_winter['Sport'].value_counts()

medal_colors = ['darkgoldenrod', 'gold', 'silver']
pd.crosstab(medal_sports_winter.Sport, medal_sports_winter.Medal).plot(kind='bar', 
                                                                       color = medal_colors, 
                                                                       stacked=True, figsize=(18,5))
plt.title('Medal frequency for Norway')
plt.xlabel('Sport')
plt.ylabel('Frequency of Medal wins')

# Can also similarly see how the US has done
usa = olympics[olympics['NOC']=='USA']
usa_winter = usa[usa['Season'] == 'Winter']
medal_sports_winter = usa_winter.groupby(['Year', 'Sport', 'Event'])
medal_sports_winter = medal_sports_winter.first()
medal_sports_winter = medal_sports_winter.reset_index()
medal_sports_winter['Sport'].value_counts()

medal_colors = ['darkgoldenrod', 'gold', 'silver']
pd.crosstab(medal_sports_winter.Sport, medal_sports_winter.Medal).plot(kind='bar', 
                                                                       color = medal_colors, 
                                                                       stacked=True, figsize=(18,5))
plt.title('Medal frequency for USA')
plt.xlabel('Sport')
plt.ylabel('Frequency of Medal wins')

# Build GLM for predicting medal outcomes in Swimming
swimming = olympics[olympics['Sport'] == 'Swimming']

# Use medal type as response variable, with 0 for NA, and 1 for Gold, Silver, or Bronze
swimming['Medal'] = swimming['Medal'].fillna(0)
swimming['Medal'] = swimming['Medal'].replace('Bronze', 1)
swimming['Medal'] = swimming['Medal'].replace('Silver', 1)
swimming['Medal'] = swimming['Medal'].replace('Gold', 1)

# Codify sex as 0 for Male and 1 for Female
swimming['Sex'] = swimming['Sex'].replace('F', 1)
swimming['Sex'] = swimming['Sex'].replace('M', 0)

# Load additional packages for modeling
import statsmodels.api as sm
from patsy import dmatrices

# Create linear predictor
y, X = dmatrices('Medal ~ Age', data=swimming, return_type='dataframe')

# Create a GLM with said predictor, modeling the response as a Binomial
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())

# Various examples of model fits, with the first one including just age
# Create a function returning the logistic link function g(eta) for our Age model
def logisticReg(beta0, Age):
    return np.exp(beta0 + Age*X.Age)/(1 + np.exp(beta0 + Age*X.Age))

# Calculate the fitted values
y_fit = logisticReg(-2.5755, 0.0334)

# Plot the fitted model (red) against the observed values from the data (blue)
plt.plot(X.Age, y.Medal,'bo', label = 'Observed')
plt.plot(X.Age, y_fit,'ro', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Age')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Swimmers')
plt.show()

# Second model includes age, height, weight, and sex, which gives a better fit
y, X = dmatrices('Medal ~ Age + Height + Weight + Sex', data=swimming, return_type='dataframe')
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())

def logisticReg2(beta0, Age, Height, Weight, Sex):
    eta = beta0 + Age*X.Age + Height*X.Height + Weight*X.Weight + Sex*X.Sex
    return np.exp(eta)/(1 + eta)

y_fit = logisticReg2(-10.4518, 0.006, 0.0297, 0.0364, 1.1196)

plt.plot(X.Height, y.Medal,'bo', label = 'Observed')
plt.plot(X.Height, y_fit,'ro', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Height')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Swimmers')
plt.show()

# Next we add the USA as an explanatory variable 
# by creating dummy variables that binarize each categorical variable of NOC
# which gives the country code for each participating country
swimming = pd.get_dummies(swimming, columns = ['NOC'], drop_first = True)

y, X = dmatrices('Medal ~ Height + Weight + Sex + NOC_USA',
                 data=swimming, return_type='dataframe')
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())

def logisticReg3(beta0, Height, Weight, Sex, USA):
  eta = beta0 + Height*X.Height + Weight*X.Weight + Sex*X.Sex + USA*X.NOC_USA
  return np.exp(eta)/(1 + np.exp(eta))

y_fit = logisticReg3(-9.9389, 0.0302, 0.0251, 0.8998, 3.0369)

plt.plot(X.Height, y.Medal + np.random.uniform(0,0.1,len(y.Medal)),'bo', 
         label = 'Observed')
plt.plot(X.Height, y_fit,'ro', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Height')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Swimmers')
plt.show()



