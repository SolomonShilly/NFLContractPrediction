import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

trainDF = pd.read_csv("MergedDF.csv")

# TCV, APY, & GTD exhibit multicollinearity (they are attributes of TCV), so APY & GTD will be excluded
# Many player stats exhibit multicollinearity with each other, so correlations were used to choose features
X = trainDF[['PASSTD', 'RECYPG', 'RECTD', 'RUSHTD', 'DINT', 'PDEF', 'PASSINT']]

# Total Contract Value is divided by 10 million to scale MSE for readability
y = trainDF[["TCV"]] / 10000000

# Using sklearn train_test_split to seperate the data
# test_size is 0.20 (0.25 & 0.30 were also tested
# random_state=0 had the best performance
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=0)

# Using RobustScaler to deal with possible outliers in contract
# The data will be fit and transformed to the training features
# The test data will than be transformed using the transform function
scaler = RobustScaler()
xTrainScaled = scaler.fit_transform(XTrain)
xTestScaled = scaler.transform(XTest)

# Creating polynomial and interaction terms for the XGB and Random Forest models only
poly = PolynomialFeatures()
polyTrain = poly.fit_transform(xTrainScaled)
polyTest = poly.transform(xTestScaled)

# Random Forest Regressor from sklearn.ensemble
# 1,000 branches for Random Forest trees
# Poisson criterion had best results
forest = RandomForestRegressor(n_estimators=1000, criterion="poisson")
forest.fit(polyTrain, yTrain)
forestYPred = forest.predict(polyTest)

# XGBoost model with 150 n_estimators
# The model favors a lower learning rate
xgb = xgb.XGBRegressor(n_estimators=150, learning_rate=0.005)
xgb.fit(polyTrain, yTrain)
xgbYPred = xgb.predict(polyTest)

# Ridge Regression model from sklearn
# Alpha of 100
ridgeReg = Ridge(alpha=100)
ridgeReg.fit(xTrainScaled, yTrain)
ridgeYPred = ridgeReg.predict(xTestScaled)

# Linear Regression model from sklearn
linearReg = LinearRegression()
linearReg.fit(xTrainScaled, yTrain)
linearYPred = linearReg.predict(xTestScaled)
print("Multiple Linear Regression Intercept:", linearReg.intercept_)

# Printing the coefficients for each model using for loops
# The feature names and coefficients can be iterated over together using the zip function
# For OLS coefficients, enumerate has to be used because coef are returned as a vector
print("\nMultiple Linear Regression Coefficients:")
for i, target_coef in enumerate(linearReg.coef_):
    for feature, coef in zip(X.columns, target_coef):
        print(f"{feature}: {coef}")

print("\nRandom Forest Feature Importances:")
for feature, importance in zip(X.columns, forest.feature_importances_):
    print(f"{feature}: {importance}")

print("\nXGBoost Feature Importances:")
for feature, importance in zip(X.columns, xgb.feature_importances_):
    print(f"{feature}: {importance}")

print("\nRidge Regression Coefficients:")
for feature, coef in zip(X.columns, ridgeReg.coef_):
    print(f"{feature}: {coef}")

# Function to evaluate each model
def evaluate_model(yTrue, yPred, model_name):
    mse = mean_squared_error(yTrue, yPred)
    r_squared = r2_score(yTrue, yPred)
    print(f"{model_name}:")
    print("  Mean Squared Error: ", mse)
    print("  R-Squared: ", r_squared)
    print("\n")

# Main function to run all code
def main():
    evaluate_model(yTest, ridgeYPred, "\nRidge Regression")
    evaluate_model(yTest, linearYPred, "Multiple Linear Regression")
    evaluate_model(yTest, xgbYPred, "XGBoost")
    evaluate_model(yTest, forestYPred, "Random Forest Regressor")

main()

quit()

# Extra graphing work
plt.scatter(trainDF["PASSYDS"], trainDF["TCV"])
plt.xlabel("PASSYDS")
plt.ylabel("TCV")
plt.title("Pass Yards vs Total Contract Value")
plt.show()

plt.scatter(trainDF["PASSATT"], trainDF["TCV"])
plt.xlabel("PASSATT")
plt.ylabel("TCV")
plt.title("Pass Attempts vs Total Contract Value")
plt.show()

plt.scatter(trainDF["CMP"], trainDF["TCV"])
plt.xlabel("CMP")
plt.ylabel("TCV")
plt.title("Completions vs Total Contract Value")
plt.show()