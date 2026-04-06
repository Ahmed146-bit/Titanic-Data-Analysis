import pandas as pd
dataset=pd.read_csv('train.csv')

print("------Displaying dataset------")
print(dataset)

print("------Displaying the first 10 rows of the dataset------")
print(dataset.head(10))

print("------Displaying dataset information------")
print(dataset.info())

print("------Displaying the description of the dataset------")
print(dataset.describe())

print("-----------------------------------------------------------")
survival_counts = dataset['Survived'].value_counts()

print(f"Number of passengers who survived: {survival_counts[1]}")
print(f"Number of passengers who did not survive: {survival_counts[0]}")

print("-----------------------------------------------------------")
gender_counts = dataset['Sex'].value_counts()
male_num = gender_counts.get('male', 0)
female_num = gender_counts.get('female', 0)
print(f"Number of male passengers: {male_num}")
print(f"Number of female passengers: {female_num}")

print("-----------------------------------------------------------")
unique_var=dataset['Embarked'].unique()
print(f"Unique values in 'Embarked' column: {unique_var}")
 
print("-----------------null values in each column-----------------")
print(dataset.isnull().sum())\

print("------Female passengers in First Class------")
result = dataset[(dataset['Sex'] == 'female') & (dataset['Pclass'] == 1)]
print(result)

print("------finding the most expensive ticket------")
sorted_result=result.sort_values(by='Fare', ascending=False)
print(sorted_result)

female_count=result['Survived'].value_counts()
print(f"Number of female passengers who survived: {female_count.get(1, 0)}")
print(f"Number of female passengers who did not survive: {female_count.get(0, 0)}")

print("-----filling missing values-----")
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

status_map = {1: 'Alive', 0: 'Dead'}

dataset['Status'] = dataset['Survived'].map(status_map)

print("------Updated dataset with Status------")
print(dataset[['Name', 'Survived', 'Status']].head())

print("-----the average survival rate for each  class-----")
average_survival_rate = dataset.groupby('Pclass')['Survived'].mean()
print(average_survival_rate)

print("-----the average ticket price by gender-----")
average_fare=dataset.groupby('Sex')['Fare'].mean()
print(average_fare)

dataset.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
print(f"Dataset after dropping 'Cabin' and 'Ticket' columns:\n{dataset.head()}")

dataset.to_csv('titanic_cleaned_results.csv', index=False)
print("------File Saved Successfully!------")