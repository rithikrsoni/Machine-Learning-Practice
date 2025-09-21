import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import sys

modelType = sys.argv[1] # user specifies the model used.


featuresDF = pd.read_csv(r"TrainingValues.csv")
labelsDF = pd.read_csv(r"TrainingLabels.csv")

# Drop  columns
columnsToDrop = ["longitude", "latitude", "date_recorded", "recorded_by", "funder", "installer",
                 "management", "subvillage", "scheme_name", "wpt_name"]
featuresDF = featuresDF.drop(columns=columnsToDrop, errors="ignore")

if "id" in labelsDF.columns:
    labelsDF = labelsDF.drop(columns=["id"])

xTrain, xVal, yTrain, yVal = train_test_split(featuresDF, labelsDF, train_size=0.7, random_state=42)

models = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "MLPClassifier": MLPClassifier
}
if modelType not in models:
    print(f"Please choose Model from {list(models.keys())}.")
    sys.exit(1)

encoders = {
    "OneHotEncoder": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "OrdinalEncoder": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
   # "TargetEncoder": TargetEncoder(smoothing=1.0)
}

class TargetEncoderWithTarget(BaseEstimator, TransformerMixin):
    def __init__(self, encoder):
        self.encoder = encoder

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        if isinstance(y, np.ndarray):
            y = pd.Series(y)  

        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        return self.encoder.transform(X)



def objective(trial): 
    categoricalFeatures = selector(dtype_include="object")(xTrain)
    numericFeatures = selector(dtype_exclude="object")(xTrain)

    encoderType = trial.suggest_categorical("encoder", list(encoders.keys()))
    numScaling = trial.suggest_categorical("scaling", ["StandardScaler", "None"])

    if encoderType == "TargetEncoder":
        encoder = TargetEncoderWithTarget(encoders[encoderType])
    else:
        encoder = encoders[encoderType]

    num_encoder = StandardScaler() if numScaling == "StandardScaler" else "passthrough"

    preprocessingWithScaling = ColumnTransformer([
        ("num", num_encoder, numericFeatures),
        ("cat", encoder, categoricalFeatures)
    ])

    imputer = SimpleImputer(strategy="most_frequent")
    xTrainImputed = pd.DataFrame(imputer.fit_transform(xTrain), columns=xTrain.columns)

    xTrainTransformed = preprocessingWithScaling.fit_transform(xTrainImputed, yTrain.values.ravel())

    yTrainRavel = yTrain.values.ravel()  
    
    modeltype = modelType

    if modeltype == "LogisticRegression":
        C = trial.suggest_loguniform("log_reg_C", 1e-3, 10)
        model = LogisticRegression(C=C, max_iter=1000)

    elif modeltype == "RandomForestClassifier":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 100)  
        max_depth = trial.suggest_int("rf_max_depth", 5, 20)  
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    elif modeltype == "GradientBoostingClassifier":
        n_estimators = trial.suggest_int("gb_n_estimators", 50, 100)  
        learning_rate = trial.suggest_loguniform("gb_learning_rate", 0.01, 0.2)  
        max_depth = trial.suggest_int("gb_max_depth", 3, 20)  
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    elif modeltype == "HistGradientBoostingClassifier":
        maxIter = trial.suggest_int("hgb_max_iter", 50, 100)  
        learningRate = trial.suggest_loguniform("hgb_learning_rate", 0.01, 0.2)  
        model = HistGradientBoostingClassifier(max_iter=maxIter, learning_rate=learningRate, random_state=42)

    elif modeltype == "MLPClassifier":
        hiddenLayerSizes = trial.suggest_categorical("mlp_hidden_layers", [(50,), (100,), (50, 50)])
        alpha = trial.suggest_loguniform("mlp_alpha", 1e-4, 1e-1)
        model = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, alpha=alpha, max_iter=100, random_state=42)  # Reduced iterations

    kf = KFold(n_splits=5, shuffle=True, random_state=42)  
    cvScores = cross_val_score(model, xTrainTransformed, yTrainRavel, cv=kf, scoring='accuracy')

    return np.mean(cvScores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  


print("Best Hyperparameters:", study.best_trial.params)


trialsDF = study.trials_dataframe()

plt.figure(figsize=(10, 5))
plt.bar(range(len(trialsDF)), trialsDF["value"], color="skyblue")
plt.xlabel("Trial Number")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Optuna Hyperparameter Optimization Results")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
