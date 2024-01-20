import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def train_and_save_models(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)

    # Map 'Male' to 0 and 'Female' to 1
    dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})

    # Map 'Yes' to 1 and 'No' to 0 for relevant columns
    binary_columns = ['Depressed', 'Anxiety', 'Fear of Failure in Exam', 'Alcohol or Drugs', 'Alone']
    for column in binary_columns:
        dataset[column] = dataset[column].map({'No': 0, 'Yes': 1})

    categorical_columns = ['Year', 'Work Load', 'Edu Level']
    numeric_columns = ['Social Media Usage', 'Study Hours']

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
            ('imputer', SimpleImputer(strategy='most_frequent'), ['Fear of Failure in Exam', 'Alcohol or Drugs', 'Alone']),
            ('numeric_imputer', SimpleImputer(strategy='mean'), numeric_columns)
        ],
        remainder='passthrough'
    )
    random_seed = 42

    for target_variable in ['Depressed', 'Anxiety']:
        features = dataset.drop(['Timestamp', 'Name', 'Depressed', 'Anxiety'], axis=1)
        labels = dataset[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=random_seed)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=random_seed))
        ])

        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [3, 6, 9],
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        model_filename = f'{target_variable.lower()}_model.pkl'
        joblib.dump(best_model, model_filename)

        print(f'{target_variable} Model Saved!')

if __name__ == '__main__':
    dataset_path = 'Student.csv'
    train_and_save_models(dataset_path)
