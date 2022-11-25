import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class airlineSatisfactionClassifier:
    def __init__(self, airline_satisfaction_train_df, airline_satisfaction_test_df):
        self.airline_satisfaction_df = pd.concat([airline_satisfaction_train_df, airline_satisfaction_test_df])

    def getAirlineSatisfactionInfo(self):
        print(self.airline_satisfaction_df.info())

    def dataPreparation(self):
        # Exclude unique attributes e.g. `Id`, `Age` and `Gender`.
        self.airline_satisfaction_df = self.airline_satisfaction_df.drop(columns=["id","Gender","Age"])
        
        # Reset index.
        self.airline_satisfaction_df = self.airline_satisfaction_df.reset_index().drop(columns=["index"])

    def sampleDatas(self,sample=5):
        print(self.airline_satisfaction_df.head(sample))
        
    #* Data preprocessing method.
    def dataPreprocessing(self):
        # Remove records that have missing value of `Arrival Delay in Minute`.
        self.airline_satisfaction_df = self.airline_satisfaction_df.dropna()

        # Encode string attributes to polynomial values.
        # Iteration all attributes in dataframes.
        for column in self.airline_satisfaction_df.columns:

            # Initialize LabelEncoder object.
            label_encoder = LabelEncoder()

            # If attribute's datatype is object, then encode all value at current attribute with fit_transform method.
            if (self.airline_satisfaction_df[column].dtype == type(object)):
                self.airline_satisfaction_df[column] = label_encoder.fit_transform(self.airline_satisfaction_df[column])
 
    #* Method for splitting train & test sets and features & class.
    def splitValidation(self):
        # Split dataframe into train & test sets.
        airline_satisfaction_train_df , airline_satisfaction_test_df = train_test_split(self.airline_satisfaction_df, test_size=0.3,random_state=0)

        # # Split feature and class labels.
        self.airline_satisfaction_feature_train = airline_satisfaction_train_df.drop(columns=["satisfaction"])
        self.airline_satisfaction_feature_test = airline_satisfaction_test_df.drop(columns=["satisfaction"])

        self.airline_satisfaction_class_train = airline_satisfaction_train_df["satisfaction"]
        self.airline_satisfaction_class_test = airline_satisfaction_test_df["satisfaction"]

        # print(self.airline_satification_feature_train.shape)
        # print(self.airline_satification_feature_test.shape)
        
    #* Method for models initialization.
    def modelInit(self):
        self.naive_bayes_model = GaussianNB()
        self.decision_tree_model = DecisionTreeClassifier()

    #* Method for train model.
    def trainModel(self):
        # Train naive bayes & decision tree model.
        self.naive_bayes_model.fit(self.airline_satisfaction_feature_train, self.airline_satisfaction_class_train)
        self.decision_tree_model.fit(self.airline_satisfaction_feature_train, self.airline_satisfaction_class_train)

    #* Method for predict satisfacton results.
    def predictedTestSet(self):
        # Predict test dataframe.
        self.naive_satisfaction_predicted_class = self.naive_bayes_model.predict(self.airline_satisfaction_feature_test)  
        self.decision_tree_satisfaction_predicted_class = self.decision_tree_model.predict(self.airline_satisfaction_feature_test)  

    #* Method for visualize datas.
    def dataVisualization(self):
        pass
    