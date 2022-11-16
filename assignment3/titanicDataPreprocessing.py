from sklearn.preprocessing import LabelEncoder
import numpy as np

# * Function for encoding string value to numerical value.
def attributeEncoder(titanic_df):
    # loop for วนรอบคอลัมน์ในชุดข้อมูล
    for column in titanic_df.columns:

        # ถ้าชุดข้อมูลที่คอลัมน์ใดมีประเภทข้อมูลเป็น object
        if (titanic_df[column].dtype == type(object)):
            # ประกาศ object LabelEncoder ถูกอ้างอิงโดยตัวแปร le
            label_encoder = LabelEncoder()

            titanic_df[column] = label_encoder.fit_transform(titanic_df[column])

    return titanic_df 

# //----------------------------------------------------------------------------------------------
# * Function for split feature attributes and class attributes
def splitFeatureClass(titanic_df, titanic_survived_class):
    # นำชุดข้อมูลมาลบ attribute ที่เป็นผลลัพธ์ออกไป
    dataframe = titanic_df.drop(titanic_survived_class, axis=1)
    # นำชุดข้อมูล attribute ที่เป็นผลลัพธ์ ถูกอ้างอิงโดยตัวแปร result
    result = titanic_df[titanic_survived_class].copy()
    return dataframe, result    # return ชุดข้อมูล data และ result กลับไป

# //----------------------------------------------------------------------------------------------
# * function for doing data preprocessing.
def dataPreprocessing(titanic_df):
    # Excludes attributes `PassengerId`, `Name` and `Ticket` because these attributes are an unique index.
    titanic_df = titanic_df.loc[:, ~titanic_df.columns.isin(["PassengerId", "Name", "Ticket"])]

    # Excludes attributes `Cabin` because of too many missing values.
    titanic_df = titanic_df.loc[:, titanic_df.columns != "Cabin"]

    # Filled attribute `age` with mean of age.
    titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].mean())
    
    # Filled attribute `age` with median of age.
    # titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
    
    # Filled attribute `age` with mode of age.
    # titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].mode())

    # Flooring Age values.
    titanic_df["Age"] = titanic_df["Age"].apply(np.floor)

    # Drps records with missing value of `Embarked`.
    titanic_df = titanic_df.dropna()

    # Encoding values of `Sex` and `Embarked` into numerical value.
    titanic_df = attributeEncoder(titanic_df)

    # Resets instances index.
    titanic_df.reset_index(drop=True,inplace=True)

    # Return current dataframe.
    return titanic_df
