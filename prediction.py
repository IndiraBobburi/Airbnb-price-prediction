import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb
# %config InlineBackend.figure_format = 'png'
import warnings
import os
warnings.filterwarnings('ignore')

class prediction(object):

    #drop data missing and data not found rows
    def preprocess(self, input):
        #initialize with original data every time
        self.train = self.original_data.copy()
        input = pd.DataFrame(input, columns = self.columns_to_keep)

        self.train = self.train.append(input, ignore_index=True)

        self.train["bedrooms"] = self.train["bedrooms"].fillna(0.5) #these are studios
        self.train["summary"] = self.train["summary"].fillna("")
        #train["bedrooms"] = train["bedrooms"].astype("str")

        #replace unpopular types with other 
        popular_types = self.train["property_type"].value_counts().head(6).index.values
        self.train.loc[~self.train.property_type.isin(popular_types), "property_type"] = "Other"

        #make price numeric:
        #if self.first:
        self.train["price"] = self.train["price"].str.replace("[$,]", "").astype("float")
        #eliminate crazy prices:
        self.train = self.train[self.train["price"] < 600]



        self.y = self.train["price"]
        train_num_cat = self.train[["neighbourhood_cleansed", "bedrooms",
                           "property_type", "room_type", "latitude", "longitude",
                           "number_of_reviews", "require_guest_phone_verification",
                            "minimum_nights"]]

        train_text = self.train[["name", "summary", "amenities"]]

        self.X_num = pd.get_dummies(train_num_cat)

        self.train.amenities = self.train.amenities.str.replace("[{}]", "")
        amenity_ohe = self.train.amenities.str.get_dummies(sep = ",")

        self.train["text"] = self.train["name"].str.cat(self.train["summary"], sep = " ")
        vect = CountVectorizer(stop_words = "english", min_df = 10)
        X_text = vect.fit_transform(self.train["text"])

        #this is numeric + amenities:
        self.X = np.hstack((self.X_num, amenity_ohe))

        #this is all of them:
        self.X_full = np.hstack((self.X_num, amenity_ohe, X_text.toarray()))
        
        return

    #metric:
    def rmse(self, y_true, y_pred):
        return(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

    #evaluates rmse on a validation set:
    def eval_model(self, model, X, y, state = 3):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state = state)
        preds = model.fit(X_tr, y_tr).predict(X_val)
        return rmse(y_val, preds)

    def pred_val(self, model, X, y):
        X_train = X[:-1]
        Y_train = y[:-1]
        
        model.fit(X_train, Y_train)
        print(X[-1])
        return model.predict([X[-1]])[0]


    def predictResult(self, data):
        inputData = []
        # for col in self.cols:
        #     inputData.append(data[col])
        print("preprocessing start with")
        #print(data);
        inp = data
        print(inp)
        self.preprocess(inp)
        print("preprocessing done")

        new_prediction = self.pred_val(xgb.XGBRegressor(), self.X, self.y)
        print("new_prediction")
        print(new_prediction)
            
        return new_prediction

    def __init__(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'input/listings.csv')
        self.original_data = pd.read_csv("input/listings.csv")
        print("file reading done")

        self.columns_to_keep = ["price", "neighbourhood_cleansed", "bedrooms",
                   "property_type", "room_type", "name", "summary",
                   "amenities", "latitude", "longitude", "number_of_reviews",
                   "require_guest_phone_verification", "minimum_nights"]

        self.original_data = self.original_data[self.columns_to_keep]

# inp = [['0', 'Roslindale', 2.0, 'House', 'Entire home/apt', 'Sunny Bungalow in the City', 'Cozy, sunny, family home.  Master bedroom high ceilings. Deck, garden with hens, beehives & play structure.   Short walk to charming village with  attractive stores, groceries & local restaurants. Friendly neighborhood. Access public transportation.','{TV,"Wireless Internet",Kitchen,"Free Parking on Premises","Pets live on this property",Dog(s),Heating,"Family/Kid Friendly",Washer,Dryer,"Smoke Detector","Fire Extinguisher",Essentials,Shampoo,"Laptop Friendly Workspace"}', 42.28261879577949, -71.13306792912681, 0, 'f', 2]]
# obj = prediction()
# obj.predictResult(inp)

# inp = [['0', 'Roslindale', 1.0, 'House', 'Entire home/apt', 'Sunny Bungalow in the City', 'Cozy, sunny, family home.  Master bedroom high ceilings. Deck, garden with hens, beehives & play structure.   Short walk to charming village with  attractive stores, groceries & local restaurants. Friendly neighborhood. Access public transportation.','{TV,"Wireless Internet",Kitchen,"Free Parking on Premises","Pets live on this property",Dog(s),Heating,"Family/Kid Friendly",Washer,Dryer,"Smoke Detector","Fire Extinguisher",Essentials,Shampoo,"Laptop Friendly Workspace"}', 40.28261879577949, -70.13306792912681, 0, 'f', 2]]
# print("calling second time")
# obj.predictResult(inp)
