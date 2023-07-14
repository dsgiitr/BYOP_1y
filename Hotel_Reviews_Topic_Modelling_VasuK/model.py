import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("Restaurant_reviews_dataset.csv")

print(df.head())

# Select independent and dependent variable
x = df[' Review'].values
y = df['Liked']

# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(stop_words='english')
x_train_vect=vect.fit_transform(x_train)
x_test_vect=vect.transform(x_test)

from sklearn.svm import SVC
model=SVC()

from sklearn.pipeline import make_pipeline
text_model=make_pipeline(CountVectorizer(),SVC())

text_model.fit(x_train,y_train)

y_pred=text_model.predict(x_test)

print(y_pred)

print(text_model.predict(["Super delicious food"]))


#from sklearn.metrics import accuracy_score
#accuracy_score(y_pred,y_test)*100


# Make pickle file of our model
pickle.dump(text_model, open("model.pkl", "wb"))