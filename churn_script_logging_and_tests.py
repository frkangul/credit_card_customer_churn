"""
Testing and logging functions in the churn_library file

Update numpy and seaborn before running this script:
pip install -U numpy
pip install -U seaborn

Author: Furkan Gul
Date: 13.12.2021
"""
import os
import logging
import churn_library as cl

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    """
    test perform eda function
    """
    try:
        perform_eda(df)
        logging.info("SUCCESS: Testing perform_eda")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: The function couldn't save eda images in the file path")
        raise err

    try:
        imdir = 'images/eda'
        files = os.listdir(imdir)
        assert len(files) > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: There aren't five images in ./images/eda path")
        raise err
    return df


def test_encoder_helper(encoder_helper, df, cat_cols, response='Churn'):
    """
    test encoder helper
    """
    df = encoder_helper(df, cat_cols, response)
    try:
        for cat_col in cat_cols:
            assert cat_col in df.columns
        logging.info("SUCCESS: Testing encoder_helper")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: There aren't categorical columns in the updated dataframe")
        raise err
    return df


def test_perform_feature_engineering(
        perform_feature_engineering,
        df,
        response='Churn'):
    """
    test perform_feature_engineering
    """
    dataset_tuple = perform_feature_engineering(df, response)
    X_train, X_test, y_train, y_test = dataset_tuple
    try:
        for dataset in dataset_tuple:
            assert dataset.shape[0] > 0
            if len(dataset.shape) == 2:
                assert dataset.shape[1] > 0
        logging.info("SUCCESS: Testing perform_feature_engineering")
    except AssertionError as err:
        logging.error("Resulting datasets don't have rows and columns")
        raise err
    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    """
    test train_models
    """
    train_models(X_train, X_test, y_train, y_test)
    try:
        result_dir = 'images/results'
        result_files = os.listdir(result_dir)
        assert len(result_files) > 0

        model_dir = 'models'
        model_files = os.listdir(model_dir)
        assert len(model_files) > 0
        logging.info("SUCCESS: Testing train_models")
    except AssertionError as err:
        logging.error("There aren't resulting images and model files")
        raise err


def main():
    """
    Test functions and logging

    Args:
        -
    Returns:
        None
    """
    # test import_data
    data_df = test_import(cl.import_data)

    # test perform_eda
    test_eda(cl.perform_eda, data_df)

    # test encoder_helper
    category_cols = ['Gender', 'Education_Level', 'Marital_Status',
                     'Income_Category', 'Card_Category']
    data_df = test_encoder_helper(cl.encoder_helper, data_df, category_cols)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, data_df)

    # test train_models
    test_train_models(cl.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)


if __name__ == "__main__":
    main()
