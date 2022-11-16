from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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