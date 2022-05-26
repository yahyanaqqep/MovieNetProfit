import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import joblib





#Reading the csv file
df = pd.read_csv('D:\First\First\Testsheet.csv')




#Dropping duplicate rows
df.drop_duplicates(inplace = True)


#Changing some datatypes 
#print(df.dtypes)
#df['release_date'] = pd.to_datetime(df['release_date'])
#print(df['release_date'].head())


#Dropping unnecessary columns
df.drop(['id','overview','imdb_id','homepage','tagline','revenue_adj','original_title','release_date','budget_adj','keywords','director'],axis =1,inplace = True)



#Deleting rows which containing null values
df.dropna(inplace = True)
#print(df.isnull().sum())
print(df.info())
#Deleting columns with 0 values

df=df.loc[~(df==0).any(axis=1)]

print(df.info())
print(df.corr())
#splitting genres,cast and production_companies columns
df['genres'] = df.genres.str.split('|')
df = df.explode('genres')
df['cast'] = df.cast.str.split('|')
df = df.explode('cast')
#df['keywords'] = df.keywords.str.split('|')
#df = df.explode('keywords')
df['production_companies'] = df.production_companies.str.split('|')
df = df.explode('production_companies')
#print(df.head())



## Applying One hot encoding on genres column 
#genre = df['genres'].str.split('|').tolist()
#flat_genre = [item for sublist in genre for item in sublist]
#set_genre = set(flat_genre)
#unique_genre = list(set_genre)
#df = df.reindex(df.columns.tolist() + unique_genre, axis=1, fill_value=0)

#for index, row in df.iterrows():
#    for val in row.genres.split('|'):
#        if val != 'NA':
#            df.loc[index, val] = 1

#df.drop('genres', axis = 1, inplace = True)


#df = pd.concat([df.drop('keywords', 1), df['keywords'].str.get_dummies(sep="|")], 1)
#print(df.info())



## Applying label encoding on genres, cast and production_companies columns
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

#df ['original_title']= df['original_title'].astype('category')
#df.original_title = le.fit_transform(df.original_title)
df ['genres']=df['genres'].astype('category')      #
df.genres=le1.fit_transform(df.genres)          #
df['cast']=df['cast'].astype('category')   #
df.cast=le2.fit_transform(df.cast)     #
#df['keywords']=df['keywords'].astype('category')
#df.keywords=le.fit_transform(df.keywords)
df['production_companies']=df['production_companies'].astype('category')     #
df.production_companies=le3.fit_transform(df.production_companies)       #
#df['director']=df['director'].astype('category')
#df.director=le.fit_transform(df.director)

#df ['release_date']= df['release_date'].astype('category')
#df.release_date = le.fit_transform(df.release_date)


# Applying feature Scaling 
scaler = MinMaxScaler()
scaled= scaler.fit_transform(df)
print (scaled)
#print(df.corr())

y=df.net_profit
x=df.drop('net_profit',axis=1)
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

x_train,x_test,y_train,y_test=model_selection.train_test_split(x_poly,y,test_size=0.3)


#applying linear regression 
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(x_test,y_test))
#finalModel='finalModel.sav'
#joblib.dump(reg,finalModel)
##print(df['release_year'].corr(df['net_profit']))
##print(df.info())

#ourmodel=joblib.load(finalModel)
#print(ourmodel.score(x_test,y_test))