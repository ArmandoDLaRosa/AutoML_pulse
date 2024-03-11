from main import *

df = pd.read_csv('/home/armando/repos/personal/AutoML_pulse/train.csv')

# Assumption 1: Columns types are correct
for column in df.columns:
    if df[column].nunique() > 20 or df[column].dtype == 'object':
        print(f"{column}: Object. Example: {df[column].unique()[:10]}\n")
        df[column] = df[column].astype('object')
    elif df[column].nunique() <= 20 or df[column].dtype == 'object':
        print(f"{column}: Categorical with {df[column].nunique()} levels. Example: {df[column].unique()}\n")
        df[column] = df[column].astype('category')
    elif pd.to_numeric(df[column], errors='coerce').notnull().all():
        print(f"{column}: Numeric. Example: {df[column].unique()[:10]}\n")
        # Check if the column can be cast to an integer
        df[column] = pd.to_numeric(df[column], errors='coerce')

# Now df contains columns with appropriate data types as per your assumptions


train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

preprocess_and_transform_data(train_data, "Survived")