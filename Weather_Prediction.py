#Importing Libraries
import warnings
warnings.filterwarnings('ignore')
#for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotnine import *
import seaborn
#sklearn
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#other
from datetime import datetime
import pandas as pd
import numpy as np

#Importing data
humidities = pd.read_csv('Weather Data/humidities.csv')
pressures = pd.read_csv('Weather Data/pressures.csv')
temperatures = pd.read_csv('Weather Data/temperatures.csv')
descriptions = pd.read_csv('Weather Data/WeatherDescription.csv')
wind_directions = pd.read_csv('Weather Data/winddirections.csv')
wind_speeds = pd.read_csv('Weather Data/windspeeds.csv')
July4Test = pd.read_csv('Weather Data/July4test.csv')

#--- Exploring Data ---

class WeatherForecast:
    def __init__(self):
        pass

    def plot(self):
        #Plotting Temperature Trends
        plt.figure(figsize = (18,8))
        plt.plot(temperatures['Denver']) #Use Atlanta for a sample
        plt.plot(temperatures['Miami'], color = 'red') #Use Miami for a sample
        plt.title('Temperature over Time: Denver (blue) vs. Miami (red)')
        plt.ylabel('Temperature (k)')
        plt.xlabel('Time')
        plt.show()

        #Plotting weather disposition
        plt.figure(figsize = (8,6))
        plt.hist(descriptions['Miami'], color = 'red', edgecolor = 'black') #Use Miami for a sample
        plt.title('Temperature over Time (Miami)')
        plt.ylabel('Temperature (k)')
        plt.xlabel('Time')
        plt.xticks(rotation = 90)
        plt.show()
        # Temperature trend shows clear changes by season. Miami has a higher average temperature than Denver.
        # Seasons will be good variables to create in order to help predict temperature.

        plt.figure(figsize = (8,6))
        plt.hist(descriptions['Denver'], edgecolor = 'gold') #Use Denver for a sample
        plt.title('Temperature over Time (Denver)')
        plt.ylabel('Temperature (k)')
        plt.xlabel('Time')
        plt.xticks(rotation = 90)
        plt.show()

    def df_population(self):
        # --- Processing Data ---

        #combining datasets into a single dataframe
        self.df = pd.DataFrame()
        self.df2 = pd.DataFrame() #temporary dataframe for pd.concat
        self.cities = ['Los Angeles', 'Denver', 'Atlanta', 'Jacksonville', 'Miami']
        #combines all datasets and adds a city label
        for i in self.cities:
            self.df2['datetime'] = humidities['datetime']
            self.df2['humidities'] = humidities[i]
            self.df2['pressures'] = pressures[i]
            self.df2['temperatures'] = temperatures[i]
            self.df2['descriptions'] = descriptions[i]
            self.df2['wind_directions'] = wind_directions[i]
            self.df2['wind_speeds'] = wind_speeds[i]
            self.df2['city'] = i
            #concatenates temporary df2 into df
            self.df = pd.concat([self.df, self.df2])
        #organizing data
        self.df = shuffle(self.df)
        self.df = self.df.reset_index()
        self.df.drop('index', axis = 1, inplace = True)

        #Creating new variables
        #seasonal variables
        self.df['summer'] = 0
        self.df['winter'] = 0
        self.df['fall'] = 0
        self.df['spring'] = 0
        #many other variables in the dataset such as temperature are affected by season
        #looping through dataframe to add new variables
        for i in range(len(self.df)):
            #adding seasonal variables
            month = self.df['datetime'].iloc[i][:2] #indexs the month (first 2 characters in datetime), matches it with season
            if month == '3/' or month == '4/' or month == '5/':
                self.df['spring'].iloc[i] = 1
            elif month == '6/' or month == '7/' or month == '8/':
                self.df['summer'].iloc[i] = 1
            elif month == '9/' or month == '10' or month == '11':
                self.df['fall'].iloc[i] = 1
            elif month == '12' or month == '1/' or month == '2/':
                self.df['winter'].iloc[i] = 1
        #creating dummy variables for city
        self.df = pd.get_dummies(self.df, columns = ['city'])
        print(self.df.head())

        #dropping null values
        #print(np.sum(df.isnull())) #sum of null values in each variable
        self.df.dropna(inplace = True)

        #--- Exploring Data ---#

        #Correlation Matrix
        plt.figure(figsize = (16, 10))
        seaborn.heatmap(self.df.corr(), annot = True, cmap = 'coolwarm')

    def linear_regression(self):
        #--- Modeling Data ---

        #Predictors
        predictors = ['summer', 'winter', 'fall', 'spring', 'humidities', 'pressures', 'wind_directions', 'wind_speeds',
                    'city_Los Angeles', 'city_Denver','city_Atlanta', 'city_Jacksonville', 'city_Miami']
        #X and y variables
        self.X = self.df[predictors]
        self.y1 = self.df['descriptions'] #classification
        self.y2 = self.df['temperatures'] #regression

        #-- Linear Regression --

        #train test split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y2, test_size = 0.2)
        #zScoring
        zScore = StandardScaler()
        zScore.fit(X_train)
        zScore.transform(X_train)
        #fit model
        self.LR_Model = LinearRegression()
        self.LR_Model.fit(X_train, y_train)
        y_pred = self.LR_Model.predict(X_test)

        #Evaluating Model
        #Output r2 and MSE values
        print("LR Training data r2:", self.LR_Model.score(X_train, y_train))
        print("LR Testing data r2:", self.LR_Model.score(X_test, y_test))
        print("LR Mean Squared Error:", mean_squared_error(y_test, y_pred))
        #plotting coefficients
        coefficients = pd.DataFrame({"Coef": self.LR_Model.coef_, "Name": predictors})
        coefficients = coefficients.append({"Coef": self.LR_Model.coef_, "Name": "Intercept"}, ignore_index = True)
        print(coefficients)
        #creating a dataframe with preicted and true values
        true_vs_pred = pd.DataFrame({"predictced values": y_pred, "true values": y_test })
        #graphing true values vs predicted values
        print('\nLR Coefficients:')
        print(ggplot(true_vs_pred, aes(x = "true values", y = "predictced values")) +
            geom_point(color = "black", fill = "red") +
            geom_smooth(method = "lm", color = 'grey') +
            ggtitle("Predicted vs True Values for Temperature"))

    def k_nearest_neighbor(self):

        #-- K-Nearest Neighbor --
        #Setting number of neighbors (n)
        self.knn = KNeighborsClassifier(n_neighbors = 25)
        #train test split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y1, test_size = 0.2)
        #zScore variables
        z = StandardScaler()
        Xz_train = z.fit_transform(X_train)
        Xz_test = z.transform(X_test)
        #fit model
        self.knn.fit(Xz_train, y_train)
        #accuracy score
        print('KNN Training data accuracy', self.knn.score(Xz_train, y_train))
        print('KNN Testing data accuracy:',self.knn.score(Xz_test, y_test))

    def pred_weather(self, July4Test):
        #--- Predicting New Data ---

        #dropping datetime column
        July4Test.drop('datetime', axis = 1, inplace = True)
        #transposing dataframe
        July4Test = July4Test.T.reset_index().reindex(columns=['index',0,1,2,3])
        July4Test.drop([5], axis = 0, inplace = True)
        July4Test.rename(columns={'index': 'city', 0: 'humidities', 1: 'pressures', 2: 'wind_directions', 3: 'wind_speeds'}, inplace=True)
        #Adding new data
        #adding dummy variables
        July4Test = pd.get_dummies(July4Test, columns = ['city'])
        #adding seasonal data
        July4Test['summer'] = 1
        July4Test['winter'] = 0
        July4Test['fall'] = 0
        July4Test['spring'] = 0
        #re-ordering columnns
        new_cols = ['summer', 'winter', 'fall', 'spring', 'humidities', 'pressures', 'wind_directions', 'wind_speeds',
                'city_Los Angeles', 'city_Denver', 'city_Atlanta', 'city_Jacksonville', 'city_Miami']
        July4Test = July4Test[new_cols]
        #Outputting new formatted data
        print('\nFormatted Testing Data:')
        print(July4Test)

        #Predicting data
        self.descript_pred = self.knn.predict(July4Test)
        #convert data from object to float for predicting continuous variables
        for i in July4Test:
            July4Test[i] = July4Test[i].astype(float)
        #Predicting
        self.temp_pred = self.LR_Model.predict(July4Test)

    def file_output(self):
        #--- Outputting to file ---

        #opening file
        file = open('Weather_Predictions.txt', 'a')
        #loops through list of cities and outputs the city name with corresponding predicted temperature/description
        for i in range(len(self.cities)):
            output = (self.cities[i] + ':\n' + str(round(self.temp_pred[i], 1)) + ' °K, ' + self.descript_pred[i] + '\n')
            file.write(str(output + '\n'))
        file.close() #closing file

w = WeatherForecast()
w.plot()
w.df_population()
w.linear_regression()
w.k_nearest_neighbor()
w.pred_weather(July4Test)
w.file_output()
