# preprocess data
# Load library
import collections
import numpy as np
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show first 5 rows
print(dataframe.head(5))

# Load library

# Create DataFrame
dataframe = pd.DataFrame()

# Add columns
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]

print()
# Show DataFrame
print(dataframe)

# Create row
new_person = pd.Series(['Molly Mooney', 40, True],
                       index=['Name', 'Age', 'Driver'])

# Append row
dataframe.concat(new_person, ignore_index=True)

print(dataframe)

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Show two rows
print(dataframe.head(2))

print('=======================================================')
# Show dimensions
print(dataframe.shape)

print('=======================================================')
# Show statistics
print(dataframe.describe)

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Select first row
print(dataframe.iloc[0])

print('=======================================================')
# Select four rows
print(dataframe.iloc[1:4])

print('=======================================================')
# Select three rows
print(dataframe.iloc[:4])

print('=======================================================')
# Set index
dataframe = dataframe.set_index(dataframe['Name'])

# Show row
print(dataframe.loc['Allen, Miss Elisabeth Walton'])

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Show top two rows where column 'sex' is 'female'
print(dataframe[dataframe['Sex'] == 'female'].head(2))

# Filter rows
print(dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)])

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Replace values, show two rows
print(dataframe['Sex'].replace("female", "Woman").head(2))

print('=======================================================')
# Replace "female" and "male with "Woman" and "Man"
print(dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))

print('=======================================================')
# Replace values, show two rows
print(dataframe.replace(1, "One").head(2))

print('=======================================================')
# Replace values, show two rows (supported by RegEx)
print(dataframe.replace(r"1st", "First", regex=True).head(2))


dataframe = pd.read_csv(url)

# Rename column, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)

# Rename columns, show two rows can accept dictionary as param
dataframe.rename(
    columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)

# Load library

# Create dictionary
column_names = collections.defaultdict(str)

# Create keys
for name in dataframe.columns:
    column_names[name]

# Show dictionary
column_names

collections.defaultdict(str,
                        {'Age': '',
                         'Name': '',
                         'PClass': '',
                         'Sex': '',
                         'SexCode': '',
                         'Survived': ''})

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())

# (var), standard deviation (std), kurtosis (kurt), skewness (skew), standard error of the mean (sem), mode (mode), median (median), and a number of others also offered.
print('=======================================================')
# Show counts
print(dataframe.count())

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Select unique values
print(dataframe['Sex'].unique())

# Show counts
print(dataframe['Sex'].value_counts())

# Show counts
print(dataframe['PClass'].value_counts())

# Show number of unique values
print(dataframe['PClass'].nunique())

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Select missing values, show two rows
print(dataframe[dataframe['Age'].isnull()].head(2))

# Replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

# Load data, set missing values
dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])

print(dataframe)

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print('=======================================================')
# Delete column
print(dataframe.drop('Age', axis=1).head(2))

# Drop columns
print(dataframe.drop(['Age', 'Sex'], axis=1).head(2))

# Drop column
print(dataframe.drop(dataframe.columns[1], axis=1).head(2))

# Create a new DataFrame
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)
print(dataframe_name_dropped)

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Delete rows, show first two rows of output
print(dataframe[dataframe['Sex'] != 'male'].head(2))

# Delete row, show first two rows of output
print(dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2))

# Delete row, show first two rows of output
print(dataframe[dataframe.index != 0].head(2))

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

print(dataframe)
# Drop duplicates, show first two rows of output
dataframe.drop_duplicates().head(2)

# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))

# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'])

# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'], keep='last')

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean
# of each group
dataframe.groupby('Sex').mean()

# Group rows
dataframe.groupby('Sex')

# Group rows, count rows
dataframe.groupby('Survived')['Name'].count()

# Group rows, calculate mean
dataframe.groupby(['Sex', 'Survived'])['Age'].mean()

# Load libraries

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
dataframe.resample('W').sum()

# Show three rows
dataframe.head(3)

# Group by two weeks, calculate mean
dataframe.resample('2W').mean()

# Group by month, count rows
dataframe.resample('M').count()

# Group by month, count rows
dataframe.resample('M', label='left').count

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Print first two names uppercased
for name in dataframe['Name'][0:2]:
    print(name.upper())

# Show first two names uppercased
[name.upper() for name in dataframe['Name'][0:2]]

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Create function


def uppercase(x):
    return x.upper()


# Apply function, show two rows
dataframe['Name'].apply(uppercase)[0:2]

# Load library

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows, apply function to groups
dataframe.groupby('Sex').apply(lambda x: x.count())

# Load library

# Create DataFrame
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns=['id', 'first', 'last'])

# Create DataFrame
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns=['id', 'first', 'last'])

# Concatenate DataFrames by rows
pd.concat([dataframe_a, dataframe_b], axis=0)

# Concatenate DataFrames by columns
pd.concat([dataframe_a, dataframe_b], axis=1)

# Create row
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

# Append row
dataframe_a.append(row, ignore_index=True)

# Load library

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                          'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns=['employee_id',
                                                           'name'])

# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns=['employee_id',
                                                    'total_sales'])

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')

# Merge DataFrames
pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id')

"""

Oftentimes, the data we need to use is complex; it doesn’t always come
in one piece. Instead in the real world, we’re usually faced with
disparate datasets, from multiple database queries or files. To get all
that data into one place, we can load each data query or data file into
pandas as individual DataFrames and then merge them together into a
single DataFrame.

This process might be familiar to anyone who has used SQL, a popular
language for doing merging operations (called joins). While the exact
parameters used by pandas will be different, they follow the same
general patterns used by other software languages and tools.

There are three aspects to specify with any merge operation. First, we have to specify the two DataFrames we want to merge together. In the
solution we named them dataframe_employees and dataframe_sales. Second, we have to specify the name(s) of the columns to merge on—that is, the columns whose values are shared between the two DataFrames. For example, in our solution both DataFrames have a column named employee_id. To merge the two DataFrames we will match up the values in each DataFrame’s employee_id column with each other. If these two columns use the same name, we can use the on parameter. However, if they have different names we can use left_on and right_on.

What is the left and right DataFrame? The simple answer is that the left
DataFrame is the first one we specified in merge and the right
DataFrame is the second one. This language comes up again in the next
sets of parameters we will need.

The last aspect, and most difficult for some people to grasp, is the
type of merge operation we want to conduct. This is specified by the
how parameter. merge supports the four main types of joins:

Inner

  Return only the rows that match in both DataFrames (e.g.,
return any row with an employee_id value appearing in both
dataframe_employees and dataframe_sales).

Outer

  Return all rows in both DataFrames. If a row exists in one
DataFrame but not in the other DataFrame, fill NaN values for the
missing values (e.g., return all rows in both dataframe_employee and
dataframe_sales).

Left

  Return all rows from the left DataFrame but only rows from the
right DataFrame that matched with the left DataFrame. Fill NaN values
for the missing values (e.g., return all rows from dataframe_employees but only rows from dataframe_sales that have a value for employee_id that appears in dataframe_employees).

Right

  Return all rows from the right DataFrame but only rows from
the left DataFrame that matched with the right DataFrame. Fill NaN
values for the missing values (e.g., return all rows from dataframe_sales but only rows from dataframe_employees that have a
value for employee_id that appears in dataframe_sales).""

"""