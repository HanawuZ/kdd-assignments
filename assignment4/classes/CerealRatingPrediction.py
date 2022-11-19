class CerealRatingPrediction:
    """
    ## Class for cereals rating prediction.
    ### Attributes : 
    - prepared cereals dataframe.
    - cereals train dataframe.
    - cereals test dataframe.
    
    - cereals feature train/test dataframe.
    - cereals rating train/test dataframe.

    - rating prediction performance report object.
    - rating prediction data visualization object. 
    
    ### Method :
    - Get cereals dataframe.
    - Set cereals dataframe.
    - SplitValidation : split feature & labels and train & test.
    - Train model method
    - Rating Prediction method

    """
    # Constructor.
    def __init__(self, cereals_df=None):
        self.cereals_df = cereals_df

    # Get cereals dataframe.
    def getCerealDataframe(self):
        return self.cereals_df

    def getCerealDataframeInfo(self):
        print(self.cereals_df.info())

    # Set cereals dataframe.
    def setCerealDataframe(self, cereals_df):
        self.cereals_df = cereals_df

    # * Statements
    # Split validation method.

    # Regression model initialize method.

    # Cereal rating prediction training method

    # Cereal rating prediction method
    
    # Cereal rating prediction performance report method.