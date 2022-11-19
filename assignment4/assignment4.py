import pandas as pd
import pathlib
from sklearn.linear_model import LinearRegression
from cerealsDataPreprocessing import *
from cerealsPerformanceReport import regressionPerformanceReport

# * Tasks
# * (1) Program performance report & data visualization module.
# * (2) Improve program to approch lowest Mean absolute error as low as possible. 
# * (3) Clean codes.

# Define cereals csv file path.
cereals_path = pathlib.Path(__file__).parent / "cereal.csv"
 
if __name__ == "__main__":
    # Read cereals csv file.
    cereals_df = pd.read_csv(cereals_path)

    # # Print cereals dataframe first 5 records and information.
    # print(cereals_df.head())
    # print(cereals_df.info())

    # Do data preprocessing
    # 1. Drop attributes `name`, `mfr` and `type`.  
    # 2. Fill missing value (-1) with average value of attribute.
    cereals_df = dataPreprocessing(cereals_df)

    # Split dataframe to train set & test set.
    # Split feature labels & class label(s).
    cereals_feature_train, cereals_feature_test, cereals_rating_train, cereals_rating_test = splitValidation(cereals_df)
    
    # Initialize linear regression model.
    cereals_rating_prediction_model = LinearRegression()

    # Train model.
    cereals_rating_prediction_model.fit(cereals_feature_train, cereals_rating_train)

    # Predict
    cereals_predicted_rating = cereals_rating_prediction_model.predict(cereals_feature_test)

    
    #* SOURCE : https://pandas.pydata.org/docs/reference/api/pandas.concat.html
    Compare_Ad_dataframe = pd.concat( 
        [ cereals_rating_test.reset_index() ,
          pd.Series(
            cereals_predicted_rating , 
            name="Predicted Rating")
        ] ,
        axis = "columns"  
    )

    print(Compare_Ad_dataframe.head(5))
    regressionPerformanceReport(cereals_feature_test, cereals_rating_prediction_model, cereals_rating_test , cereals_predicted_rating)
    # Coeff, Slope
    print("\nCoef : ",cereals_rating_prediction_model.coef_)
    
    # Intercept
    print("Intercept : ",cereals_rating_prediction_model.intercept_)
    
    

    
    