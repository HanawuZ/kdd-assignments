from modules.cerealsDataPreprocessing import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class CerealRatingPrediction:
    """
    ## Class for cereals rating prediction.
    ### Attributes : 
    - / prepared cereals dataframe.
    - / cereals train dataframe.
    - / cereals test dataframe.
    
    - / cereals feature train/test dataframe.
    - / cereals rating train/test dataframe.

    
    ### Method :
    - / Get cereals dataframe.
    - / Set cereals dataframe.
    - / SplitValidation : split feature & labels and train & test.
    - / Train model method
    - / Prediction method
    - / Rating prediction performance report object.
    - / Printing regression equation.
    - (WIP) rating prediction data visualization object. 

    """
    # Constructor.
    def __init__(self, cereals_df=None):
        self.cereals_df = cereals_df

    # Get cereals dataframe.
    def getCerealDataframe(self):
        return self.cereals_df

    # Get cereals dataframe informantion.
    def getCerealDataframeInfo(self):
        print(self.cereals_df.info())

    # Set cereals dataframe.
    def setCerealDataframe(self, cereals_df):
        self.cereals_df = cereals_df

    # Split validation method.
    def cerealsDataframeSplit(self):
        (self.cereals_feature_train, 
        self.cereals_feature_test, 
        self.cereals_rating_train, 
        self.cereals_rating_test) = splitValidation(self.cereals_df)
        
    # Regression model initialize method.
    def modelInit(self):
        self.cereal_rating_prediction_model = LinearRegression()

    # Cereal rating prediction training method
    def trainModel(self):
        self.cereal_rating_prediction_model.fit(
            self.cereals_feature_train,
            self.cereals_rating_train
        )

    # Cereal rating prediction method
    def predictRating(self):
        self.predicted_rating = self.cereal_rating_prediction_model.predict(self.cereals_feature_test)
        # print(self.predicted_rating)

    # * Methos for show regression statistics
    def showStats(self):
        """ ### Cereal rating prediction performance report method. """
        # Add constant 
        const_cereals_feature_train = sm.add_constant(self.cereals_feature_train)
        const_cereals_feature_test = sm.add_constant(self.cereals_feature_test)

        # Create regression model with OLS class from statmodel.api
        model = sm.OLS(self.cereals_rating_train, const_cereals_feature_train).fit()
        
        # Predicted model
        model.predict(const_cereals_feature_test)

        # Display model Summary
        print(model.summary())

    #* Method for showing regression equation of cereals rating prediction model.
    def showRegressionEquation(self):
        """
        ### Method for showing regression equation of cereals rating prediction model.
        """
        # Get intercept value of prediction model.
        self.regression_intecept = self.cereal_rating_prediction_model.intercept_

        # Get array of coefficent value of prediction model.
        self.regression_coef = self.cereal_rating_prediction_model.coef_

        # Get list of feature names. 
        # Zip model coefficient with cereals feature names.
        # Iteration for creating regression equation as string.
        print("\nRegression Equation :")
        for coef, feature_name in zip(self.regression_coef, self.cereals_feature_train.columns):
            # Round coefficient value to get value with 4 decimals.
            
            # format ==> - X.XXXX * feature_name 
            if (coef < 0):
                print("- {coef} * {feature_name}".format(coef=np.around(np.absolute(coef), decimals=4), feature_name=feature_name))

            # format ==> + X.XXXX * feature_name 
            else :
                print("+ {coef} * {feature_name}".format(coef=np.around(coef, decimals=4), feature_name=feature_name))

        # Print intercept value.
        if (self.regression_intecept < 0):
            print("- {intercept}".format(intercept=np.around(np.absolute(self.regression_intecept), decimals=4), feature_name=feature_name))

        else :
            print("+ {intercept}".format(intercept=np.around(self.regression_intecept, decimals=4), feature_name=feature_name))

    
    #* Method for data visualization
    def visualizePrediction(self):
        """
        ### Plan(s):
        - Using matplotlib or seaborn
        - Visualize rating and predicted rating line graph
        """
        # Visualize line graph of actual cereal rating.
        # Y Axis is feature labels
        # X Axis is class labels (`rating`).
        
        # Get protein and actual cereals rating then reset index.

        # protein = self.cereals_feature_test["protein"].reset_index().drop(columns = ["index"])
        # Transfrom numpy array into pandas series and named column 'rating'
        predicted_rating = pd.Series(self.predicted_rating)
        predicted_rating.name = "rating"
        
        # Visualize line graph of predicted cereal rating.
        # Plot all fetaures-cereal rating relations with seaborn.
        for attr in self.cereals_feature_test.columns:
            current_attr_df = self.cereals_feature_test[attr].reset_index().drop(columns = ["index"])
            sns.lineplot(
                x="rating" , y=attr ,
                data = pd.concat([current_attr_df,predicted_rating], axis=1),
                label="{attr} - rating".format(attr=attr)
            )

        plt.xlabel("Predicted Cereals Rating")
        plt.ylabel("Features")
        plt.legend()
        plt.show()

    def findLowestMeanAbsoluteError(self):
        # +++++ Pseudo Code +++++
        # Get array of feature names
        # Initialize temporary linear regression model.
        # Assign ignore_index = 0
        # Iteration from i = 0 to i = 10
        #     Choose features[i],features[i+1]
        #     If i < 10 then
        #         If i >= 2 then
        #             Increment ignore_index by 1
        #             Iteration from k = 0 to k = ignore_index
        #                 Choose features[k]
        #                 Create new feature_train_df with features[i],features[i+1] and features[j]
        #                 Train regression model with new feature_train_df and label_train_df
        #                 Get mean absolute error
                        
        #         Assign j = i+2
        #         Iteration from j to j = 11
        #             Choose features[j]
        #             Create new feature_train_df with features[i],features[i+1] and features[j]
        #             Train regression model with new feature_train_df and label_train_df
        #             Get mean absolute error
                
        #     Else then
        #         Increment ignore_index by 1
        #         Iteration from k = 0 to k = ignore_index
        #             Choose features[k]
        #             Create new feature_train_df with features[i],features[i+1] and features[j]
        #             Train regression model with new feature_train_df and label_train_df
        #             Get mean absolute error
        pass


        
        
        
