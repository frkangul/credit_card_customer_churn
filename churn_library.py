"""
Functions to identify credit card customers that are most likely to churn

Update numpy and seaborn before running this script:
pip install -U numpy
pip install -U seaborn

Author: Furkan Gul
Date: 13.12.2021
"""

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    Args:
            pth: (str) a path to the csv
    Returns:
            df: (DataFrame) pandas dataframe for the file in the input path
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder

    Args:
            df: (DataFrame) pandas dataframe for churn dataset
    Returns:
            None
    """
    # Add Churn label to the data set
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot and save histogram of "Churn" column
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title("Churn Distribution")
    plt.savefig("./images/eda/churn_distribution.png")
    plt.close()

    # Plot and save histogram of "Customer_age" column
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title("Customer Age Distribution")
    plt.savefig("./images/eda/customer_age_distribution.png")
    plt.close()

    # Plot and save normalized value counts of "Marital_Status" column
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Distribution")
    plt.savefig("./images/eda/marital_status_distribution.png")
    plt.close()

    # Plot and save distribution of "Total_Trans_Ct" column
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.title("Total Transaction Distribution")
    plt.savefig("./images/eda/total_transaction_distribution.png")
    plt.close()

    # Plot and save heatmap correlation of all columns
    plt.figure(figsize=(40, 20))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Heatmap")
    plt.savefig("./images/eda/heatmap.png")
    plt.close()


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Args:
            df: (DataFrame) pandas dataframe for churn dataset
            category_lst: (list of str) contains column names which are categorical features
            response: (str) response name [optional argument that could be used
            for naming variables or index y column]
    Returns:
            df: pandas dataframe with new columns for
    """
    # col stands for encoded columns
    for col in category_lst:
        col_lst = []
        col_groups = df.groupby(col).mean()['Churn']

        # Loop over all values in corresponding columns
        for val in df[col]:
            col_lst.append(col_groups.loc[val])

        # Give columns name accordingly
        col_name = col + '_' + response
        df[col_name] = col_lst
    return df


def perform_feature_engineering(df, response):
    """
    Apply encoder_helper function to dataframe and then preparing training and testing datasets

    Args:
              df: (DataFrame) pandas dataframe for
              response: (str) response name [optional argument that could be used for
              naming variables or index y column]
    Returns:
              X_train: (DataFrame) X training data
              X_test: (DataFrame) X testing data
              y_train: (DataFrame) y training data
              y_test: (DataFrame) y testing data
    """
    # Define categorical columns
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    df = encoder_helper(df, cat_columns, response)

    # Prepare dataframe for label and features
    y = df['Churn']
    X = pd.DataFrame()

    # Only include columns in keep_cols as features
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder

    Args:
            y_train: (DataFrame) training response values
            y_test: (DataFrame) test response values
            y_train_preds_lr: (DataFrame) training predictions from logistic regression
            y_train_preds_rf: (DataFrame) training predictions from random forest
            y_test_preds_lr: (DataFrame) test predictions from logistic regression
            y_test_preds_rf: (DataFrame) test predictions from random forest
    Returns:
             None
    """
    # Generate and save training and testing result reports for Random Forest
    # model
    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.8, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.2, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    # Generate and save training and testing result reports for Logistic
    # Regression model
    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.8, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.2, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores feature importances in pth

    Args:
            model: (model object) containing feature_importances_
            X_data: (DataFrame) pandas dataframe of X values
            output_pth: (str) path to store the figure
    Returns:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(60, 30))  # to see it better

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def shap_plots(model, X_data, output_pth):
    """
    plot and save shap values graph

    Args:
        model: (model object) ML model
        X_data: (DataFrame) pandas dataframe of X test values
        output_pth: (str) path to store the figure
    Returns:
        None
    """
    # Generate and save plot
    plt.figure(figsize=(20, 5))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.title("Shap Values")
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models

    Args:
              X_train: (DataFrame) X training data
              X_test: (DataFrame) X testing data
              y_train: (DataFrame) y training data
              y_test: (DataFrame) y testing data
    Returns:
              None
    """
    # define random forest and lg classifiers
    rfc = RandomForestClassifier(random_state=42)
    # Due to errors with 'lbfgs', it is changed.
    lrc = LogisticRegression(solver='sag', max_iter=300)

    # grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # models fitting
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # model predictions with best random forest classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # model predictions with logistic regression classifier
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # plot and save roc curve for rfc and lgc
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.title("ROC Curve")
    plt.savefig("./images/results/roc_curve_results.png")
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # save classification reports
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # save feature importance plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importances.png')

    # save shap plot
    # shap_plots(cv_rfc.best_estimator_, X_test, './images/results/shap_values.png')


def main():
    """
    Run all required functions in a correct order

    Args:
        -
    Returns:
        None
    """
    # Import the data
    data_df = import_data("./data/bank_data.csv")

    # Apply exploratory data analysis
    perform_eda(data_df)

    # Perform feature engineering and split the dataset
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        data_df, response='Churn')

    # Model training & storing all necessary outputs
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)


if __name__ == '__main__':
    main()
