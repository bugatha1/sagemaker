
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path = os.path.join('/opt/ml/processing/input', 'raw_churn.csv')
    
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df)
    
    # Encode target
    lb = LabelBinarizer()
    label = lb.fit_transform(df['Churn?'])
    df['Churn?'] = label.flatten()
    
    negative_examples, positive_examples = np.bincount(df['Churn?'])
    print('Data after cleaning: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))
    
    split_ratio = args.train_test_split_ratio
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn?', axis=1), df['Churn?'], test_size=split_ratio, random_state=0)

    
    numerical_cols = ['Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Eve Mins',
                      'Eve Calls', 'Night Mins', 'Night Calls', 'Intl Mins', 'Intl Calls',
                      'CustServ Calls']
    categorical_cols = ["State", "Int'l Plan", "VMail Plan"]

    num_proc = make_pipeline(SimpleImputer(strategy='median'))
    cat_proc = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore', sparse=False))    
    preprocessor = make_column_transformer((numerical_cols, num_proc),
                                           (categorical_cols, cat_proc))
    print('Running preprocessing and feature engineering transformations')
    train_features = preprocessor.fit_transform(X_train)
    test_features = preprocessor.transform(X_test)
    
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    
    one_hot_encoder = preprocessor.named_transformers_['pipeline-2'].named_steps['onehotencoder']
    encoded_cat_cols = one_hot_encoder.get_feature_names(input_features=categorical_cols).tolist()
    processed_cols = numerical_cols + encoded_cat_cols

    train_df = pd.DataFrame(train_features, columns=processed_cols)
    train_df.insert(0, 'churn', y_train)

    test_df = pd.DataFrame(test_features, columns=processed_cols)
    test_df.insert(0, 'churn', y_test)
    
    train_output_path = os.path.join('/opt/ml/processing/train', 'train.csv')
    test_output_path = os.path.join('/opt/ml/processing/test', 'test.csv')

    print('Saving training features to {}'.format(train_output_path))
    train_df.to_csv(train_output_path, header=True, index=False)

    print('Saving test features to {}'.format(test_output_path))
    test_df.to_csv(test_output_path, header=True, index=False)
