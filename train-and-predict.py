import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder , StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #f1_score, precision_score, recall_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import numpy as np
import time
import sys


trainingData = sys.argv[1]
trainLabels = sys.argv[2]
testData = sys.argv[3]
numPreprocessing = sys.argv[4]
preprocessing = sys.argv[5]
modelType = sys.argv[6] #Regression, MLP Classifier, Gradient etc
testPredictionFile = sys.argv[7]

featuresDF = pd.read_csv(trainingData)
labelsDF = pd.read_csv(trainLabels)
testDF = pd.read_csv(testData)

def calculatePercentage(df):

    missingPercentage = (df.isnull().sum() / len(df)) * 100
    return missingPercentage.sort_values(ascending = False)


def imputeMissingValues(df):
    missingPercentage = calculatePercentage(df)
    for column in df.columns:
        if 0 < missingPercentage[column] <= 7:
            df[column] = df[column].fillna(df[column].mode()[0])
    return df


df = pd.read_csv(trainingData)
missingPercentage = calculatePercentage(df)
print(missingPercentage)



print(featuresDF.head())
print(labelsDF.head())
print(testDF.head())

columnsToDrop = [
    "longitude", "latitude","date_recorded" , "recorded_by", "funder", "installer" , "management" , "subvillage", "scheme_name" , "wpt_name"
]
featuresDF = featuresDF.drop(columns=columnsToDrop, errors="ignore")
testDF = testDF.drop(columns = columnsToDrop, errors = "ignore")


featuresDF = imputeMissingValues(featuresDF)
testDF = imputeMissingValues(testDF)

if "id" in labelsDF.columns:
   labelsDF = labelsDF.drop(columns =["id"])

if "id" in testDF.columns:
    testID = testDF["id"].copy()
else:
    print("Warning: 'id' column not found in test data. Generating sequential IDs.")
    testID = pd.Series(range(len(testDF)))

#labelsDF["status_group"] = labelsDF["status_group"].map(
    #{"functional":0, "functional needs repair":1 , "non functional": 2}
#)

xTrain,xVal, yTrain, yVal = train_test_split(featuresDF, labelsDF, train_size= 0.7, random_state= 42)


models = {
    "LogisticRegression": LogisticRegression(C=0.1, max_iter=5000),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),  
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),  
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=100, random_state=42),  
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(), max_iter=1000, random_state=42)  
}


encoders = {
    "OneHotEncoder": OneHotEncoder(handle_unknown= "ignore", sparse_output= False),
    "OrdinalEncoder": OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value =-1 ),
    "TargetEncoder": TargetEncoder(smoothing = 1.0)

}





class TargetEncoderWithTarget(BaseEstimator, TransformerMixin):

    def __init__(self, encoder):
        self.encoder = encoder

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)
   


if modelType not in models:
    print(f"Please choose Model from {list(models.keys())}.")
    sys.exit(1)


model = models[modelType]

categoricalFeatures = selector(dtype_include="object")(featuresDF)
numericFeatures = selector(dtype_exclude="object")(featuresDF)


if numPreprocessing == "StandardScaler":
    numEncoder = StandardScaler()
else:
    numEncoder = "passthrough"
    print("No numeric preprocessing specified. Values will not be scaled")


if preprocessing == "TargetEncoder":
   encoder = TargetEncoderWithTarget(TargetEncoder(smoothing=1.0))
   print("Using TargetEncoder with target variable.")
elif preprocessing in encoders:
    encoder = encoders[preprocessing]
else:
    print(f"Unknown preprocessing. Please choose Encoder from {list(encoders.keys())}.")
    sys.exit(1)

preprocessingWithScaling = ColumnTransformer([
     ("num", numEncoder,numericFeatures),
     ("cat", encoder,categoricalFeatures)
])

if preprocessing == "TargetEncoder":
    xTrainTransformed = preprocessingWithScaling.fit_transform(xTrain, yTrain.values.ravel())
else:
    xTrainTransformed = preprocessingWithScaling.fit_transform(xTrain)

     

kf = KFold(n_splits=5, shuffle =True,random_state=42)
cvScores = cross_val_score(model, xTrainTransformed, yTrain.values.ravel(), cv=kf, scoring='accuracy')
meanCv = np.mean(cvScores)
stdCv = np.std(cvScores)
   
startTime = time.time()

model.fit(xTrainTransformed, yTrain.values.ravel())

endTime = time.time()

trainingTime = endTime - startTime

print(f"Model used: {modelType} with {preprocessing} encoding and {numPreprocessing}.")
print(f"Training completed in {trainingTime:.2f} seconds")

xValTransformed = preprocessingWithScaling.transform(xVal)
yPred = model.predict(xValTransformed)
classificationRate = accuracy_score(yVal, yPred)


testTransformed = preprocessingWithScaling.transform(testDF.drop(columns=columnsToDrop, errors="ignore"))
testPredictions = model.predict(testTransformed)

testPredictionsDF = pd.DataFrame({"id": testID, "status_group": testPredictions})


print("First 20 Predictions:\n", testPredictionsDF.head(20))


testPredictionsDF.to_csv(testPredictionFile, index=False)
print(f"Predictions saved to {testPredictionFile}")