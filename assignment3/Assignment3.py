import pandas as pd
import pathlib

# * Tasks
# * -> ? ปรับปรุงข้อมูลให้ Decision Tree มีประสิทธิภาพมากขึ้น
# * -> ? ปรับปรุงโค๊ดให้อ่านง่าย
# * -> / ปรับปรุงการทำ Data Preparation
# * -> / ทำ Visualization 

# * URL: https://www.geeksforgeeks.org/how-to-merge-multiple-csv-files-into-a-single-pandas-dataframe/
# * URL: https://www.statology.org/pandas-exclude-column/

# ########################## + Datas Import + ##########################
# Titanic train dataframe path.
titanic_train_path = pathlib.Path(__file__).parent /"titanic"/"train.csv"
# Titanic test dataframe path.
titanic_test_path = pathlib.Path(__file__).parent /"titanic"/"test.csv"
# Titanic test class label dataframe path.
survived_class_path = pathlib.Path(__file__).parent /"titanic"/"gender_submission.csv"

# Imports Titanic train csv dataframe.
titanic_train_df = pd.read_csv(titanic_train_path)

# Imports Titanic test csv dataframe.
titanic_test_df = pd.read_csv(titanic_test_path)

# Read Titanic test survived class dataframe.
titanic_test_survived_class = pd.read_csv(survived_class_path)
titanic_test_df = pd.merge(titanic_test_df,titanic_test_survived_class)

# * Dataframes check
# Show dataframe's infos.
# print(train_titanic_df.info())
# print(train_titanic_df)
# print(test_titanic_df.info())
# print(test_titanic_df)

# ########################## + Data Preprocessing + ##########################
from titanicDataPreprocessing import dataPreprocessing, splitFeatureClass
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Preprocesses titanic train & test dataframes.
titanic_train_df = dataPreprocessing(titanic_train_df)
titanic_test_df = dataPreprocessing(titanic_test_df)

# Show prepared dataframe infos.
# print(train_titanic_df.info())
# print(test_titanic_df.info())

# Merge prepared dataframes for re-splitting train & test.
prepared_titanic_df = pd.concat([titanic_train_df, titanic_test_df]).reset_index(drop=True)

# Train and test split.
titanic_train_df, titanic_test_df = train_test_split(prepared_titanic_df, test_size=0.3, random_state=100)

# Split class attribute `Survived`.
prepared_titanic_train_df , prepared_titanic_train_survived_class = splitFeatureClass(titanic_train_df, "Survived")
prepared_titanic_test_df , actual_titanic_test_survived_class = splitFeatureClass(titanic_test_df, "Survived")

# print(prepared_train_titanic_df.head())
# print(prepared_test_titanic_df.head())

# //----------------------------------------------------------------------------------------------
# ########################## + Data Evaluation + ##########################
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Initialize Naive Bayes model.
naive_bayes_model = GaussianNB()

# Initialize Decision Tree model.
decision_tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=None)

# Train models.
naive_bayes_model.fit(prepared_titanic_train_df,prepared_titanic_train_survived_class)
decision_tree_model.fit(prepared_titanic_train_df,prepared_titanic_train_survived_class)


# Prediction
bayes_pred_survived_class = naive_bayes_model.predict(prepared_titanic_test_df)
decision_tree_pred_survived_class = decision_tree_model.predict(prepared_titanic_test_df)

# //--------------------------------------------------------------------------------------
# ########################## + Performance + ##########################
from sklearn.metrics import classification_report
from graphviz import Source
from titanicDataVisualization import *

# Show Naive Bayes classifaction report.
print("++++++++ Naive Bayes Classfication Report +++++++")
bayes_clf_report = classification_report(
    bayes_pred_survived_class, 
    actual_titanic_test_survived_class,
    target_names=["Not Survived","Survived"],
    output_dict=True
)
# print(bayes_clf_report)

# Show Decision Tree classification report.
print("\n\n++++++++ Decision Tree Classfication Report +++++++")
decision_tree_clf_report = classification_report(
    decision_tree_pred_survived_class, 
    actual_titanic_test_survived_class, 
    target_names=["Not Survived","Survived"],
    output_dict=True
)
# print(decision_tree_clf_report)

clf_list= [naive_bayes_model,decision_tree_model]

# ########################## + Confusion Matrix Visualization + ##########################
confusionMatrixComparison(clf_list, prepared_titanic_test_df, actual_titanic_test_survived_class)
# classificationReportComparison(bayes_clf_report, decision_tree_clf_report)

# # ########################## + Decision Tree mdoel Export + ##########################
# # Gets feature names of dataframe.
# titanic_class_labels = list(prepared_train_titanic_df.columns)
# # print(titanic_class_labels)

# # Creates and exports Decision Tree graph with graphviz.
# graph = export_graphviz(
#     decision_tree_model,                        
#     out_file=None,  
#     filled=True, 
#     rounded=True,
#     special_characters=True,
#     feature_names = titanic_class_labels,
#     class_names=['0','1'],
# )

# # Set export image format as "png"
# graph = Source(graph, format="png")

# # Export decision tree model as png image.
# graph.render("Titanic_Survivor_Decision_Tree")

