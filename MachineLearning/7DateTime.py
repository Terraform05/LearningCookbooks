# Load libraries
from pytz import all_timezones
import numpy as np
import pandas as pd

# Create strings
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

# Convert to datetimes
datetime = [pd.to_datetime(date, format='%d-%m-%Y %I:%M %p')
            for date in date_strings]
print(datetime)

coerced = [pd.to_datetime(date, format="%d-%m-%Y %I:%M %p",
                          errors="coerce") for date in date_strings]
print(coerced)

"""
%Y Full year 2001
%m Month w/ zero padding 04   
%d Day of the month w/ zero padding 09  
%I Hour (12hr clock) w/ zero padding 02  
%p AM or PM AM  
%M Minute w/ zero padding 05  
%S Second w/ zero padding 09
"""
print('=======================|timezones|=======================')
# Load library

# Create datetime
pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
# Create datetime
date = pd.Timestamp('2017-05-01 06:00:00')

# Set time zone
date_in_london = date.tz_localize('Europe/London')

# Show datetime
print(date_in_london)

# Change time zone
date_in_london.tz_convert('Africa/Abidjan')

# Create three dates
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))

# Set time zone
dates.dt.tz_localize('Africa/Abidjan')

# Load library

# Show two time zones
all_timezones[0:2]

print('=======================|select date and time|=======================')

# Load library

# Create data frame
dataframe = pd.DataFrame()

# Create datetimes
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# Select observations between two datetimes
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
          (dataframe['date'] <= '2002-1-1 04:00:00')]

# Set index
dataframe = dataframe.set_index(dataframe['date'])

# Select observations between two datetimes
dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']

print('=======================|date data to features|=======================')

# Load library

# Create data frame
dataframe = pd.DataFrame()

# Create five dates
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')

# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

# Show three rows
print(dataframe.head(3))

print('=======================|date difference|=======================')

# Load library

# Create data frame
dataframe = pd.DataFrame()

# Create two datetime features
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# Calculate duration between features
print(dataframe['Left'] - dataframe['Arrived'])

# Calculate duration between features
print(pd.Series(delta.days for delta in (
    dataframe['Left'] - dataframe['Arrived'])))

print('=======================|encode days of wk|=======================')

# Load library

# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

print(dates)
# Show days of the week
print(dates.dt.day_name())

# Show days of the week
print('day', dates.dt.day)
print('day', dates.dt.weekday)

print('=======================|lagged n time period feature|=======================')

# Load library

# Create data frame
dataframe = pd.DataFrame()

# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1, 2.2, 3.3, 4.4, 5.5]

print(dataframe)

# Lagged values by one row
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

# Show data frame
print(dataframe)

print('=======================|rolling time windows|=======================')

# Load library

# Create datetimes
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Create data frame, set index
dataframe = pd.DataFrame(index=time_index)

# Create feature
dataframe["Stock_Price"] = [1, 2, 3, 4, 5]

print(dataframe)
# Calculate rolling mean
print('\nrolling mean\n', dataframe.rolling(window=2).mean())

print('=======================|missing time series data|=======================')

# Load libraries

# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Create data frame, set index
dataframe = pd.DataFrame(index=time_index)

# Create feature with a gap of missing values
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]

print(dataframe)

print('=======================|fill gaps w/ interpolate|=======================')

# Interpolate missing values (Fill gaps)
dataframe.interpolate()
print(dataframe)

print('=======================|fill w/ last known|=======================')

# Forward-fill
dataframe.ffill()
print(dataframe)

print('=======================|fill w/ last known val|=======================')

# Back-fill
dataframe.bfill()
print(dataframe)

print('=======================|interpolate|=======================')

# Interpolate missing values
dataframe.interpolate(method="quadratic")

# Interpolate missing values
dataframe.interpolate(limit=1, limit_direction="forward")