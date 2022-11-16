from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# * Function For classification report bar plot.
def classificationBarPlot(clf_report_axe , clf_report_labels, clf_report, survived_class) :
    """
    Call function `classificationBarPlot` for plotting not survived and survived bayes classification report bars.
    has arguments.
       - bayes classification report axes.
       - classification report labels = ("Precision" , "Recall" ,"F1-Score")
       - bayes classification reports = (not survived classification report, survived classification report)
    """
    bar_width = 0.15
    opacity = 0.8
    index = np.arange(len(clf_report_labels))

    clf_report_axe.bar(
        clf_report_labels,
        height = clf_report,
        width = bar_width,
        alpha = opacity,
    )
    # clf_report_axe.bar_label(rect, padding=3)
    clf_report_axe.set_xticks(index , clf_report_labels)
    clf_report_axe.title.set_text(survived_class)

def confusionMatrixComparison(cls_list : list, test_titanic_df, titanic_survived_class):
    fig, axes = plt.subplots(1, 2, figsize = (18, 8))
    fig.suptitle("Titanic Survivor Classifier Model comparison")
    for cls, ax in zip(cls_list, axes):
        plot_confusion_matrix(
            cls,
            test_titanic_df,
            titanic_survived_class,
            ax = ax,
            cmap = 'Blues',
            display_labels=["Not Survived","Survived"]
        )
        ax.xaxis.set_label_text("Predicted Survived Class")
        ax.yaxis.set_label_text("Actual Survived Class")
        ax.title.set_text(type(cls).__name__)
    
    plt.show()

def classificationReportComparison(bayes_clf_report : dict, decision_tree_clf_report : dict):
    bayes_clf_report_values = [bayes_clf_report["Not Survived"],bayes_clf_report["Survived"]]
    """ Returned List will be something like :
    [   {'precision': 0.8766519823788547, 
        'recall': 0.8504273504273504,
        'f1-score': 0.8633405639913233, 
        'support': 234}, 
        
        {'precision': 0.7878787878787878, 
        'recall': 0.8227848101265823, 
        'f1-score': 0.804953560371517, 
        'support': 158}                     ]
    """
    # Not Survived classification report values.
    notSurvived_bayes_clfReport_values = list(bayes_clf_report_values[0].values())
    notSurvived_bayes_clfReport_values.pop()
    # ? Out : [0.8766519823788547, 0.8504273504273504, 0.8633405639913233]

    Survived_bayes_clfReport_values = list(bayes_clf_report_values[1].values())
    Survived_bayes_clfReport_values.pop()
    
    decision_tree_clf_report_values = [decision_tree_clf_report["Not Survived"], decision_tree_clf_report["Survived"]]
    notSurvived_decision_tree_clfReport_values = list(decision_tree_clf_report_values[0].values())
    notSurvived_decision_tree_clfReport_values.pop()

    Survived_decision_tree_clfReport_values = list(decision_tree_clf_report_values[1].values())
    Survived_decision_tree_clfReport_values.pop()

    notSurvived_clf_report = [notSurvived_bayes_clfReport_values,  notSurvived_decision_tree_clfReport_values]
    survived_clf_report = [Survived_bayes_clfReport_values, Survived_decision_tree_clfReport_values]

    clf_report_list = [notSurvived_clf_report, survived_clf_report]

    clfReport_labels = ("Precision","Recall","F1-Score")
    fig, axes = plt.subplots(2,1)
    index = np.arange(len(clfReport_labels))
    bar_width = 0.15
    opacity = 0.8

    # ? Iterates not survived axe and survived axe. 
    # จับคู่ ax[0] <--> [notSurvived_bayes, notSurvived_decision_tree] <--> ["Naive Bayes", "Decision Tree"]
    # จับคู่ ax[1] <--> [Survived_bayes, Survived_decision_tree] <--> ["Naive Bayes", "Decision Tree"]
    
    for i in range(2):
        rects1 = axes[i].bar(index-bar_width/2, clf_report_list[i][0], bar_width,
            alpha=opacity,
            label="Naive Bayes")
        rects2 = axes[i].bar(index+bar_width/2, clf_report_list[i][1], bar_width,
            alpha=opacity,
            label="Decision Tree")
        axes[i].bar_label(rects1, padding=3)
        axes[i].bar_label(rects2, padding=3)
        axes[i].set_xticks(index , clfReport_labels)
        axes[i].legend()
    axes[0].title.set_text('Not Survived Classification score')
    axes[1].title.set_text('Survived Classification score')
    plt.tight_layout()
    plt.show()

#//------------------------------------------------------------------------------------------------------------
def classificationComparison(clf_dict : dict):
    """ Parameter clf_dict
    clf_dict = {
        'Naive Bayes': [naive_bayes_model, bayes_clf_report],
        'Decision Tree': [decision_tree_model, decision_tree_clf_report],
        'Survived Class' : [prepared_titanic_test_df, actual_titanic_test_survived_class]
    }
    """
    # Define Classification Labels.
    clfReport_labels = ("Precision" , "Recall" ,"F1-Score")
    survived_class_labels = ["Not Survived", "Survived"]


    # Define Naive Bayes model & classification report.
    bayes_clf = clf_dict['Naive Bayes'][0]
    bayes_clf_report = clf_dict['Naive Bayes'][1]
    bayes_clf_report = [bayes_clf_report["Not Survived"], bayes_clf_report["Survived"]]
    

    decision_tree_clf = clf_dict['Decision Tree'][0]
    decision_tree_clf_report = clf_dict['Decision Tree'][1]

    clf_fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    clf_fig.suptitle('Titanic Classification Comparison')
    bayes_fig, decision_tree_fig = clf_fig.subfigures(1, 2, wspace=0.07)

    bayes_fig.suptitle('Naive Bayes Classification')

    # Plots Naive Bayes confusion matrix
    bayes_confusion_matrix_figs, bayes_clf_report_figs =  bayes_fig.subfigures(2, 1, height_ratios=[1, 1.4])
    bayes_confusion_matrix_figs.suptitle('Confusion Matrix')
    plot_confusion_matrix(
        bayes_clf,
        clf_dict['Survived Class'][0],
        clf_dict['Survived Class'][1],
        ax = bayes_confusion_matrix_figs.subplots(1, 1),
        cmap = 'Blues',
        display_labels=survived_class_labels
    )

    # Plots Naive Bayes Classifaction report. 
    bayes_clf_report_figs.suptitle('Bayes Classification Report')

    # Create Naive Bayes classification report list.
    notSurvied_bayes_clfReport = list(bayes_clf_report[0].values())
    notSurvied_bayes_clfReport.pop()
    survied_bayes_clfReport = list(bayes_clf_report[1].values())
    survied_bayes_clfReport.pop()
    bayes_clfReport = [notSurvied_bayes_clfReport, survied_bayes_clfReport]
    
    # Plot subplot with 2 rows & 1 columns.
    bayes_clf_report_axes = bayes_clf_report_figs.subplots(nrows=2, ncols=1, sharey=True)

    # Call function `classificationBarPlot` for plotting not survived and survived bayes classification report bars.
    # has arguments.
    #   -> bayes classification report axes.
    #   -> classification report labels = ("Precision" , "Recall" ,"F1-Score")
    #   -> bayes classification reports = (not survived classification report, survived classification report)
    
    # ? Test not survived bayes clf report.
    for axe, clf_report,survived_class in zip(bayes_clf_report_axes, bayes_clfReport, survived_class_labels):
        classificationBarPlot(axe, clfReport_labels, clf_report, survived_class)

    ########################################################################
    decision_tree_fig.suptitle('Decision Tree Classification')
    decision_tree_confusion_matrix_figs, decision_tree_clf_report_figs =  decision_tree_fig.subfigures(2, 1, height_ratios=[1, 1.4])
    decision_tree_confusion_matrix_figs.suptitle('Confusion Matrix')
    
    #//-------------------------------------------------------------------------------------
    plot_confusion_matrix(
        decision_tree_clf,
        clf_dict['Survived Class'][0],
        clf_dict['Survived Class'][1],
        ax = decision_tree_confusion_matrix_figs.subplots(1, 1),
        cmap = 'hot_r',
        display_labels=survived_class_labels
    )

    notSurvied_decision_tree_clfReport = list(bayes_clf_report[0].values())
    notSurvied_decision_tree_clfReport.pop()
    survied_decision_tree_clfReport = list(bayes_clf_report[1].values())
    survied_decision_tree_clfReport.pop()
    decision_tree_clfReport = [notSurvied_decision_tree_clfReport, survied_decision_tree_clfReport]

    decision_tree_clf_report_figs.suptitle('Decision Tree Classification Report')
    decision_tree_clf_report_axes = decision_tree_clf_report_figs.subplots(2, 1, sharex=True)
    for axe, clf_report,survived_class in zip(decision_tree_clf_report_axes, decision_tree_clfReport, survived_class_labels):
        classificationBarPlot(axe, clfReport_labels, clf_report, survived_class)

    plt.show()
    
    