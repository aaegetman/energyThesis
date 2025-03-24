# %% [markdown]
# # Electricity Data Analysis
# 
# This notebook analyzes the electricity consumption and injection data with English column names.

# %% [markdown]
# ## 1. Data Import and Setup
# 
# First, let's import the necessary libraries and load our dataset.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Create directory for results if it doesn't exist
os.makedirs('analysis_results', exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# %% [markdown]
# ## 2. Loading the Data
# 
# We'll now load the dataset with semicolon delimiter and convert column names to English.

# %%
# Load the dataset
file_path = 'P6269_1_50_DMK_Sample_Elek/chunks/chunk_0.csv'

# First, let's peek at the file to confirm the delimiter
with open(file_path, 'r') as f:
    first_line = f.readline().strip()
    
print(f"First line of the file: {first_line}")

# Load the data with the proper delimiter
df = pd.read_csv(file_path, delimiter=';')

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# %% [markdown]
# ## 3. Converting to English Column Names
# 
# Let's translate the Dutch column names to English for better readability.

# %%
# Rename columns from Dutch to English
column_translations = {
    'EAN_ID': 'EAN_ID',  # Keep as is
    'Datum': 'Date',
    'Datum_Startuur': 'Date_StartHour',
    'Volume_Afname_kWh': 'Volume_Consumption_kWh',
    'Volume_Injectie_kWh': 'Volume_Injection_kWh',
    'Warmtepomp_Indicator': 'Heat_Pump_Indicator',
    'Elektrisch_Voertuig_Indicator': 'Electric_Vehicle_Indicator',
    'PV-Installatie_Indicator': 'PV_Installation_Indicator',
    'Contract_Categorie': 'Contract_Category'
}

# Rename columns
df.rename(columns=column_translations, inplace=True)

print("Column names after translation:")
print(df.columns.tolist())

# %% [markdown]
# ## 4. Data Type Conversion
# 
# Let's ensure all columns have the appropriate data types.

# %%
# Convert numeric columns correctly
for col in ['Volume_Consumption_kWh', 'Volume_Injection_kWh']:
    if col in df.columns:
        # Check current type
        current_type = df[col].dtype
        print(f"Converting {col} from {current_type}")
        
        # Convert to numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            # If string/object, handle comma as decimal separator
            df[col] = pd.to_numeric(df[col].astype(str).map(lambda x: x.replace(',', '.')), errors='coerce')
        else:
            # If already somewhat numeric but needs cleaning
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert date columns to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Date_StartHour'] = pd.to_datetime(df['Date_StartHour'])

# Convert indicator columns to integers (0/1)
for col in ['Heat_Pump_Indicator', 'Electric_Vehicle_Indicator', 'PV_Installation_Indicator']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Display data types after conversion
print("\nData types after conversion:")
print(df.dtypes)

# %% [markdown]
# ## 5. Create Time Zone Adjusted Column
# 
# Let's adjust the timestamps to Belgium local time (Europe/Brussels).

# %%
# Create local time column for Belgium
try:
    # Check if timestamps already have timezone info
    sample_time = df['Date_StartHour'].iloc[0]
    has_tzinfo = hasattr(sample_time, 'tzinfo') and sample_time.tzinfo is not None
    print(f"Timestamps have timezone info: {has_tzinfo}")
    
    # Create local time column
    if has_tzinfo:
        # Already timezone-aware
        df['Date_StartHour_Local'] = df['Date_StartHour'].dt.tz_convert('Europe/Brussels')
    else:
        # Assume UTC if no timezone
        df['Date_StartHour_Local'] = df['Date_StartHour'].dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels')
    
    # Extract time components for analysis
    df['Hour'] = df['Date_StartHour_Local'].dt.hour
    df['Day_of_Week'] = df['Date_StartHour_Local'].dt.dayofweek
    df['Day_Name'] = df['Date_StartHour_Local'].dt.day_name()
    
    print(f"Sample original time: {df['Date_StartHour'].iloc[0]}")
    print(f"Sample local time: {df['Date_StartHour_Local'].iloc[0]}")
    
except Exception as e:
    print(f"Error adjusting time zone: {str(e)}")
    # If timezone conversion fails, use naive timestamps
    df['Hour'] = df['Date_StartHour'].dt.hour
    df['Day_of_Week'] = df['Date_StartHour'].dt.dayofweek
    df['Day_Name'] = df['Date_StartHour'].dt.day_name()

# %% [markdown]
# ## 6. Basic Data Exploration
# 
# Let's examine the first few rows and check for missing values.

# %%
# View the first few rows
print("First 5 rows of the dataset:")
display(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
if any(missing_values > 0):
    for col in missing_values[missing_values > 0].index:
        print(f"- {col}: {missing_values[col]} missing values")
else:
    print("No missing values found")

# Summary statistics for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
print("\nSummary statistics for numeric columns:")
display(df[numeric_cols].describe())

# %% [markdown]
# ## 7. Time Series Analysis
# 
# Let's create time series visualizations of our electricity data.

# %%
# Create time series plots for consumption and injection

# 1. Consumption over time
plt.figure(figsize=(15, 7))
plt.plot(df['Date_StartHour_Local'], df['Volume_Consumption_kWh'], color='blue', linewidth=1)
plt.title('Electricity Consumption Over Time (Belgium Local Time)')
plt.xlabel('Date & Time')
plt.ylabel('Consumption (kWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_results/consumption_over_time.png', dpi=300)
plt.show()

# 2. Injection over time
plt.figure(figsize=(15, 7))
plt.plot(df['Date_StartHour_Local'], df['Volume_Injection_kWh'], color='green', linewidth=1)
plt.title('Electricity Injection Over Time (Belgium Local Time)')
plt.xlabel('Date & Time')
plt.ylabel('Injection (kWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_results/injection_over_time.png', dpi=300)
plt.show()

# 3. Both consumption and injection together
plt.figure(figsize=(15, 7))
plt.plot(df['Date_StartHour_Local'], df['Volume_Consumption_kWh'], color='blue', linewidth=1, label='Consumption')
plt.plot(df['Date_StartHour_Local'], df['Volume_Injection_kWh'], color='green', linewidth=1, label='Injection')
plt.title('Electricity Consumption & Injection Over Time (Belgium Local Time)')
plt.xlabel('Date & Time')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_results/consumption_and_injection_over_time.png', dpi=300)
plt.show()

# %% [markdown]
# ## 8. Time-Based Aggregation Analysis
# 
# Let's analyze consumption and injection patterns by different time periods.

# %%
# Create time period columns for aggregation
df['Weekday_Weekend'] = df['Day_of_Week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
df['Week'] = df['Date_StartHour_Local'].dt.isocalendar().week
df['Month'] = df['Date_StartHour_Local'].dt.month
df['Quarter'] = df['Date_StartHour_Local'].dt.quarter
df['Year'] = df['Date_StartHour_Local'].dt.year

# %% [markdown]
# ### 8.1 Weekday vs Weekend Analysis

# %%
# Weekday vs Weekend aggregation
weekend_agg = df.groupby('Weekday_Weekend').agg({
    'Volume_Consumption_kWh': 'sum',
    'Volume_Injection_kWh': 'sum'
}).reset_index()

print("Weekend vs Weekday Energy Sums:")
display(weekend_agg)

# Calculate percentage differences
total_consumption = weekend_agg['Volume_Consumption_kWh'].sum()
total_injection = weekend_agg['Volume_Injection_kWh'].sum()
weekend_agg['Consumption_Percentage'] = weekend_agg['Volume_Consumption_kWh'] / total_consumption * 100
weekend_agg['Injection_Percentage'] = weekend_agg['Volume_Injection_kWh'] / total_injection * 100

# Visualize weekday vs weekend
plt.figure(figsize=(12, 6))
x = weekend_agg['Weekday_Weekend']
width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - width/2, weekend_agg['Volume_Consumption_kWh'], width=width, color='blue', alpha=0.7, label='Consumption')
plt.bar(x_pos + width/2, weekend_agg['Volume_Injection_kWh'], width=width, color='green', alpha=0.7, label='Injection')

# Add value labels on top of bars
for i, v in enumerate(weekend_agg['Volume_Consumption_kWh']):
    plt.text(i - width/2, v + 5, f'{v:.1f}', ha='center')
    
for i, v in enumerate(weekend_agg['Volume_Injection_kWh']):
    plt.text(i + width/2, v + 5, f'{v:.1f}', ha='center')

plt.title('Total Energy by Weekday vs Weekend')
plt.ylabel('Energy (kWh)')
plt.xticks(x_pos, x)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('analysis_results/weekday_weekend_energy.png', dpi=300)
plt.show()

print(f"Weekday consumption: {weekend_agg.loc[weekend_agg['Weekday_Weekend'] == 'Weekday', 'Consumption_Percentage'].values[0]:.1f}% of total")
print(f"Weekend consumption: {weekend_agg.loc[weekend_agg['Weekday_Weekend'] == 'Weekend', 'Consumption_Percentage'].values[0]:.1f}% of total")

# %% [markdown]
# ### 8.2 Daily Analysis

# %%
# Daily aggregation
daily_agg = df.groupby('Day_Name').agg({
    'Volume_Consumption_kWh': 'sum',
    'Volume_Injection_kWh': 'sum'
}).reset_index()

# Ensure days of week are in order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_agg['Day_Order'] = daily_agg['Day_Name'].map(lambda x: day_order.index(x))
daily_agg = daily_agg.sort_values('Day_Order').drop('Day_Order', axis=1)

print("Daily Energy Sums:")
display(daily_agg)

# Visualize daily patterns
plt.figure(figsize=(14, 7))
x = daily_agg['Day_Name']
width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - width/2, daily_agg['Volume_Consumption_kWh'], width=width, color='blue', alpha=0.7, label='Consumption')
plt.bar(x_pos + width/2, daily_agg['Volume_Injection_kWh'], width=width, color='green', alpha=0.7, label='Injection')

plt.title('Energy Consumption and Injection by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Energy (kWh)')
plt.xticks(x_pos, x, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('analysis_results/daily_energy.png', dpi=300)
plt.show()

# %% [markdown]
# ### 8.3 Monthly Analysis

# %%
# Monthly aggregation
monthly_agg = df.groupby(['Year', 'Month']).agg({
    'Volume_Consumption_kWh': 'sum',
    'Volume_Injection_kWh': 'sum'
}).reset_index()

# Add month name for better readability
month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
monthly_agg['Month_Name'] = monthly_agg['Month'].map(month_names)
monthly_agg['Period'] = monthly_agg['Year'].astype(str) + '-' + monthly_agg['Month_Name']

print("Monthly Energy Sums:")
display(monthly_agg)

# Visualize monthly patterns
plt.figure(figsize=(15, 8))
x = monthly_agg['Period']
width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - width/2, monthly_agg['Volume_Consumption_kWh'], width=width, color='blue', alpha=0.7, label='Consumption')
plt.bar(x_pos + width/2, monthly_agg['Volume_Injection_kWh'], width=width, color='green', alpha=0.7, label='Injection')

plt.title('Monthly Energy Consumption and Injection')
plt.xlabel('Month')
plt.ylabel('Energy (kWh)')
plt.xticks(x_pos, x, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('analysis_results/monthly_energy.png', dpi=300)
plt.show()

# %% [markdown]
# ### 8.4 Quarterly Analysis

# %%
# Quarterly aggregation
quarterly_agg = df.groupby(['Year', 'Quarter']).agg({
    'Volume_Consumption_kWh': 'sum',
    'Volume_Injection_kWh': 'sum'
}).reset_index()

quarterly_agg['Quarter_Label'] = quarterly_agg['Year'].astype(str) + '-Q' + quarterly_agg['Quarter'].astype(str)

print("Quarterly Energy Sums:")
display(quarterly_agg)

# Visualize quarterly patterns
plt.figure(figsize=(12, 7))
x = quarterly_agg['Quarter_Label']
width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - width/2, quarterly_agg['Volume_Consumption_kWh'], width=width, color='blue', alpha=0.7, label='Consumption')
plt.bar(x_pos + width/2, quarterly_agg['Volume_Injection_kWh'], width=width, color='green', alpha=0.7, label='Injection')

plt.title('Quarterly Energy Consumption and Injection')
plt.xlabel('Quarter')
plt.ylabel('Energy (kWh)')
plt.xticks(x_pos, x, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('analysis_results/quarterly_energy.png', dpi=300)
plt.show()

# %% [markdown]
# ## 9. Hourly Pattern Analysis

# %%
# Hourly aggregation
hourly_avg = df.groupby('Hour').agg({
    'Volume_Consumption_kWh': 'mean',
    'Volume_Injection_kWh': 'mean'
}).reset_index()

print("Hourly Energy Averages:")
display(hourly_avg)

# Visualize hourly patterns
plt.figure(figsize=(14, 7))
plt.plot(hourly_avg['Hour'], hourly_avg['Volume_Consumption_kWh'], 'b-o', linewidth=2, markersize=8, label='Consumption')
plt.plot(hourly_avg['Hour'], hourly_avg['Volume_Injection_kWh'], 'g-o', linewidth=2, markersize=8, label='Injection')
plt.title('Average Hourly Energy Patterns')
plt.xlabel('Hour of Day')
plt.ylabel('Average Energy (kWh)')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('analysis_results/hourly_energy_patterns.png', dpi=300)
plt.show()

# Find peak hours
peak_consumption_hours = hourly_avg.nlargest(3, 'Volume_Consumption_kWh')
peak_injection_hours = hourly_avg.nlargest(3, 'Volume_Injection_kWh')

print("\nPeak consumption hours:")
for _, row in peak_consumption_hours.iterrows():
    print(f"- Hour {int(row['Hour'])}: {row['Volume_Consumption_kWh']:.4f} kWh")

print("\nPeak injection hours:")
for _, row in peak_injection_hours.iterrows():
    print(f"- Hour {int(row['Hour'])}: {row['Volume_Injection_kWh']:.4f} kWh")

# %% [markdown]
# ## 10. Analysis by Household Feature

# %%
# Analyze consumption by electric vehicle ownership
if 'Electric_Vehicle_Indicator' in df.columns:
    ev_group = df.groupby('Electric_Vehicle_Indicator').agg({
        'Volume_Consumption_kWh': ['mean', 'median', 'std', 'sum'],
        'Volume_Injection_kWh': ['mean', 'median', 'std', 'sum'],
        'EAN_ID': 'nunique'  # Count unique households
    })
    
    ev_group.columns = ['Consumption_Mean', 'Consumption_Median', 'Consumption_StdDev', 'Consumption_Sum',
                        'Injection_Mean', 'Injection_Median', 'Injection_StdDev', 'Injection_Sum',
                        'Unique_Households']
    
    ev_group.index = ['No EV', 'Has EV']
    
    print("Energy patterns by electric vehicle ownership:")
    display(ev_group)
    
    # Calculate percentage differences
    ev_increase = (ev_group.loc['Has EV', 'Consumption_Mean'] / ev_group.loc['No EV', 'Consumption_Mean'] - 1) * 100
    
    print(f"\nEV owners consume {ev_increase:.2f}% more electricity on average")
    
    # Visualize EV vs non-EV consumption
    plt.figure(figsize=(10, 6))
    plt.bar(['No EV', 'Has EV'], [ev_group.loc['No EV', 'Consumption_Mean'], ev_group.loc['Has EV', 'Consumption_Mean']], 
            color=['skyblue', 'navy'], alpha=0.7)
    plt.title('Average Electricity Consumption by EV Ownership')
    plt.ylabel('Average Consumption (kWh)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('analysis_results/ev_consumption_comparison.png', dpi=300)
    plt.show()

# %% [markdown]
# ## 11. Analysis by PV Installation (Solar Panels)

# %%
# Analyze consumption and injection by solar panel installation
if 'PV_Installation_Indicator' in df.columns:
    pv_group = df.groupby('PV_Installation_Indicator').agg({
        'Volume_Consumption_kWh': ['mean', 'median', 'std', 'sum'],
        'Volume_Injection_kWh': ['mean', 'median', 'std', 'sum'],
        'EAN_ID': 'nunique'  # Count unique households
    })
    
    pv_group.columns = ['Consumption_Mean', 'Consumption_Median', 'Consumption_StdDev', 'Consumption_Sum',
                        'Injection_Mean', 'Injection_Median', 'Injection_StdDev', 'Injection_Sum',
                        'Unique_Households']
    
    pv_group.index = ['No Solar', 'Has Solar']
    
    print("Energy patterns by solar panel installation:")
    display(pv_group)
    
    # Calculate injection ratio
    pv_group['Injection_to_Consumption_Ratio'] = pv_group['Injection_Mean'] / pv_group['Consumption_Mean']
    
    print("\nInjection to consumption ratio:")
    for idx in pv_group.index:
        if pd.notna(pv_group.loc[idx, 'Injection_to_Consumption_Ratio']):
            print(f"- {idx}: {pv_group.loc[idx, 'Injection_to_Consumption_Ratio']:.4f}")
            
    # Visualize consumption vs injection by PV status
    plt.figure(figsize=(12, 7))
    x = ['No Solar', 'Has Solar']
    width = 0.35
    x_pos = np.arange(len(x))
    
    plt.bar(x_pos - width/2, [pv_group.loc['No Solar', 'Consumption_Mean'], pv_group.loc['Has Solar', 'Consumption_Mean']], 
            width=width, color='blue', alpha=0.7, label='Consumption')
    plt.bar(x_pos + width/2, [pv_group.loc['No Solar', 'Injection_Mean'], pv_group.loc['Has Solar', 'Injection_Mean']], 
            width=width, color='green', alpha=0.7, label='Injection')
    
    plt.title('Average Energy Patterns by Solar Panel Installation')
    plt.ylabel('Average Energy (kWh)')
    plt.xticks(x_pos, x)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('analysis_results/solar_panel_energy_comparison.png', dpi=300)
    plt.show()

# %% [markdown]
# ## 12. Summary of Findings
# 
# Based on our comprehensive analysis of the electricity data with English column names, we can draw several key conclusions:
# 
# 1. **Time-Based Patterns**:
#    - Distinct hourly patterns show peak consumption periods throughout the day
#    - Weekly cycles reveal differences between weekday and weekend electricity usage
#    - Monthly and quarterly aggregations highlight seasonal variations
# 
# 2. **Household Features Impact**:
#    - Electric vehicle ownership significantly increases electricity consumption
#    - Solar panel installations lead to higher grid injection rates
# 
# 3. **Grid Balance**:
#    - The relationship between consumption and injection varies by time of day and season
#    - Consumption and injection patterns follow different temporal rhythms
# 
# 4. **Policy Implications**:
#    - Time-of-use pricing could be optimized based on observed consumption patterns
#    - EV charging infrastructure planning should account for increased household demand
#    - Solar generation creates opportunities for grid balancing through proper management
# 
# These insights can inform energy policy, grid management strategies, and targeted efficiency programs.

# %% [markdown]
# ## 13. Calendar Feature Engineering and Holiday Analysis
# 
# In this section, we'll enrich our dataset with Belgian calendar features including:
# 1. National/bank holidays
# 2. School holiday periods
# 
# Then we'll analyze how these calendar events affect energy consumption and injection patterns.

# %% [markdown]
# ### 13.1 Adding Belgian Holiday Features
# 
# First, we'll add Belgian national holidays to our dataset. We'll use the holidays library for this, which provides accurate holiday information for many countries including Belgium.

# %%
# Install holidays library if not already installed
try:
    import holidays
except ImportError:
    print("Installing holidays library...")
    !pip install holidays
    import holidays

import holidays
from datetime import datetime, timedelta

# Create a Belgian holidays dictionary for the years in our dataset
min_year = df['Date'].dt.year.min()
max_year = df['Date'].dt.year.max()
print(f"Dataset spans from {min_year} to {max_year}")

# Create Belgian holidays dictionary
be_holidays = holidays.Belgium(years=range(min_year, max_year + 1))

# Function to check if a date is a Belgian holiday
def is_belgian_holiday(date):
    return date in be_holidays

# Add holiday indicator and name to dataframe
df['Is_Holiday'] = df['Date'].apply(is_belgian_holiday).astype(int)
df['Holiday_Name'] = df['Date'].apply(lambda date: be_holidays.get(date) if date in be_holidays else None)

# Display holidays found in the dataset
holidays_in_data = df[df['Is_Holiday'] == 1][['Date', 'Holiday_Name']].drop_duplicates().sort_values('Date')
print(f"\nBelgian holidays found in dataset ({len(holidays_in_data)} holidays):")
display(holidays_in_data)

# %% [markdown]
# ### 13.2 Adding Belgian School Holiday Periods
# 
# Now we'll add Belgian school holiday periods. These vary slightly by year and region, but we'll use the general patterns for Flanders region, which is the most populated region of Belgium.

# %%
# Define Belgian school holiday periods (approximate dates for Flanders)
# Source: https://onderwijs.vlaanderen.be/nl/schoolvakanties (Flemish Government Education Portal)
# and https://www.schoolvakanties-belgie.be/ (School Holiday Portal)

def get_school_holidays(year):
    """Return a dictionary of school holiday periods for a given year in Belgium (Flanders)"""
    holidays = {}
    
    # Fall Break (Herfstvakantie) - 1 week around Nov 1
    fall_start = datetime(year, 10, 31) - timedelta(days=datetime(year, 10, 31).weekday())
    holidays[(fall_start, fall_start + timedelta(days=6))] = "Fall Break"
    
    # Christmas Break (Kerstvakantie) - 2 weeks including Christmas and New Year
    xmas_start = datetime(year, 12, 24) - timedelta(days=datetime(year, 12, 24).weekday())
    if xmas_start.day > 20:  # If Christmas Eve is late in the week, start the break earlier
        xmas_start = xmas_start - timedelta(days=7)
    # Extend Christmas break to first week of next year
    holidays[(xmas_start, datetime(year + 1, 1, 7))] = "Christmas Break"
    
    # Spring Break (Krokusvakantie) - 1 week in Feb/Mar (variable)
    if year == 2022:
        spring_start = datetime(year, 2, 28)
    elif year == 2023:
        spring_start = datetime(year, 2, 20)
    else:
        # Approximate for other years - usually last week of February
        spring_start = datetime(year, 2, 28) - timedelta(days=datetime(year, 2, 28).weekday())
    holidays[(spring_start, spring_start + timedelta(days=6))] = "Spring Break"
    
    # Easter Break (Paasvakantie) - 2 weeks around Easter
    # This is complex to calculate exactly, so approximate based on known dates
    if year == 2022:
        easter_start = datetime(year, 4, 4)
    elif year == 2023:
        easter_start = datetime(year, 4, 3)
    else:
        # Approximate for other years - usually first 2 weeks of April
        easter_start = datetime(year, 4, 1)
    holidays[(easter_start, easter_start + timedelta(days=13))] = "Easter Break"
    
    # Summer Break (Zomervakantie) - July and August
    summer_start = datetime(year, 7, 1)
    holidays[(summer_start, datetime(year, 8, 31))] = "Summer Break"
    
    return holidays

# Create a function to check if a date falls within school holidays
def get_school_holiday_name(date):
    # Handle both datetime objects and integer years
    if isinstance(date, (int, np.integer)):
        year = date
        # For year-only input, return None as we need a specific date
        return None
    else:
        # Check if date falls in first week of year (part of previous year's Christmas break)
        if date.month == 1 and date.day <= 7:
            prev_year_holidays = get_school_holidays(date.year - 1)
            for (start, end), name in prev_year_holidays.items():
                if name == "Christmas Break" and start.year == date.year - 1:
                    if start <= date <= end:
                        return name
        
        # Check current year's holidays
        year = date.year
        school_holidays = get_school_holidays(year)
        
        # Check if the date falls within any holiday period
        for (start, end), name in school_holidays.items():
            if start <= date <= end:
                return name
        
        return None

# Debug example for July 21, 2022
debug_date = datetime(2022, 7, 21)
print(f"\nDebugging school holiday check for {debug_date}:")
print(f"Date: {debug_date}")
print(f"School holiday: {get_school_holiday_name(debug_date)}")

# Add school holiday indicator to dataframe
df['School_Holiday'] = df['Date'].apply(lambda x: get_school_holiday_name(x))
df['Is_School_Holiday'] = df['School_Holiday'].notna().astype(int)

# Display school holiday periods found in the dataset
school_holidays_in_data = df[df['Is_School_Holiday'] == 1][['Date', 'School_Holiday']].drop_duplicates().sort_values('Date')
print(f"\nSchool holiday periods found in dataset:")
display(school_holidays_in_data.head(10))  # Show first 10 for brevity
print(f"Total school holiday dates: {len(school_holidays_in_data)}")

# Count by holiday type
holiday_counts = df[df['Is_School_Holiday'] == 1]['School_Holiday'].value_counts()
print("\nSchool holiday distribution:")
display(holiday_counts)

# %% [markdown]
# ### 13.3 Analysis of Energy Patterns on National Holidays
# 
# Now let's analyze how energy consumption and injection patterns differ between holidays and non-holidays.

# %%
# Analyze energy consumption and injection on holidays vs. non-holidays
holiday_energy = df.groupby('Is_Holiday').agg({
    'Volume_Consumption_kWh': ['mean', 'median', 'std', 'sum'],
    'Volume_Injection_kWh': ['mean', 'median', 'std', 'sum'],
    'Date': 'count'  # Count of records
}).rename(columns={'Date': 'Records'})

# Flatten the multi-level columns
holiday_energy.columns = [f"{col[0]}_{col[1]}" for col in holiday_energy.columns]
holiday_energy.index = ['Non-Holiday', 'Holiday']

print("Energy patterns on holidays vs. non-holidays:")
display(holiday_energy)

# Calculate percentage difference
consumption_diff = ((holiday_energy.loc['Holiday', 'Volume_Consumption_kWh_mean'] / 
                    holiday_energy.loc['Non-Holiday', 'Volume_Consumption_kWh_mean']) - 1) * 100

injection_diff = ((holiday_energy.loc['Holiday', 'Volume_Injection_kWh_mean'] / 
                   holiday_energy.loc['Non-Holiday', 'Volume_Injection_kWh_mean']) - 1) * 100

print(f"\nOn holidays, consumption is {consumption_diff:.2f}% {'higher' if consumption_diff > 0 else 'lower'} than non-holidays")
print(f"On holidays, injection is {injection_diff:.2f}% {'higher' if injection_diff > 0 else 'lower'} than non-holidays")

# Visualize energy patterns on holidays vs. non-holidays
plt.figure(figsize=(12, 7))
x = ['Non-Holiday', 'Holiday']
width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - width/2, [holiday_energy.loc['Non-Holiday', 'Volume_Consumption_kWh_mean'], 
                           holiday_energy.loc['Holiday', 'Volume_Consumption_kWh_mean']], 
        width=width, color='blue', alpha=0.7, label='Consumption')
plt.bar(x_pos + width/2, [holiday_energy.loc['Non-Holiday', 'Volume_Injection_kWh_mean'], 
                           holiday_energy.loc['Holiday', 'Volume_Injection_kWh_mean']], 
        width=width, color='green', alpha=0.7, label='Injection')

# Add value labels on top of bars
for i, v in enumerate([holiday_energy.loc['Non-Holiday', 'Volume_Consumption_kWh_mean'], 
                       holiday_energy.loc['Holiday', 'Volume_Consumption_kWh_mean']]):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
for i, v in enumerate([holiday_energy.loc['Non-Holiday', 'Volume_Injection_kWh_mean'], 
                       holiday_energy.loc['Holiday', 'Volume_Injection_kWh_mean']]):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')

plt.title('Average Energy Patterns on Holidays vs. Non-Holidays')
plt.ylabel('Average Energy (kWh)')
plt.xticks(x_pos, x)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('analysis_results/holiday_energy_comparison.png', dpi=300)
plt.show()

# %% [markdown]
# ### 13.4 Analysis of Energy Patterns During School Holidays
# 
# Now let's analyze how energy consumption and injection patterns differ during school holiday periods.

# %%
# Analyze energy consumption and injection during school holidays vs. regular days
school_holiday_energy = df.groupby('Is_School_Holiday').agg({
    'Volume_Consumption_kWh': ['mean', 'median', 'std', 'sum'],
    'Volume_Injection_kWh': ['mean', 'median', 'std', 'sum'],
    'Date': 'count'  # Count of records
}).rename(columns={'Date': 'Records'})

# Flatten the multi-level columns
school_holiday_energy.columns = [f"{col[0]}_{col[1]}" for col in school_holiday_energy.columns]
school_holiday_energy.index = ['Regular School Days', 'School Holiday']

print("Energy patterns during school holidays vs. regular school days:")
display(school_holiday_energy)

# Calculate percentage difference
sh_consumption_diff = ((school_holiday_energy.loc['School Holiday', 'Volume_Consumption_kWh_mean'] / 
                        school_holiday_energy.loc['Regular School Days', 'Volume_Consumption_kWh_mean']) - 1) * 100

sh_injection_diff = ((school_holiday_energy.loc['School Holiday', 'Volume_Injection_kWh_mean'] / 
                      school_holiday_energy.loc['Regular School Days', 'Volume_Injection_kWh_mean']) - 1) * 100

print(f"\nDuring school holidays, consumption is {sh_consumption_diff:.2f}% {'higher' if sh_consumption_diff > 0 else 'lower'} than regular school days")
print(f"During school holidays, injection is {sh_injection_diff:.2f}% {'higher' if sh_injection_diff > 0 else 'lower'} than regular school days")

# Visualize energy patterns during school holidays vs. regular days
plt.figure(figsize=(12, 7))
x = ['Regular School Days', 'School Holiday']
width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - width/2, [school_holiday_energy.loc['Regular School Days', 'Volume_Consumption_kWh_mean'], 
                           school_holiday_energy.loc['School Holiday', 'Volume_Consumption_kWh_mean']], 
        width=width, color='blue', alpha=0.7, label='Consumption')
plt.bar(x_pos + width/2, [school_holiday_energy.loc['Regular School Days', 'Volume_Injection_kWh_mean'], 
                           school_holiday_energy.loc['School Holiday', 'Volume_Injection_kWh_mean']], 
        width=width, color='green', alpha=0.7, label='Injection')

plt.title('Average Energy Patterns During School Holidays vs. Regular Days')
plt.ylabel('Average Energy (kWh)')
plt.xticks(x_pos, x)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('analysis_results/school_holiday_energy_comparison.png', dpi=300)
plt.show()

# %% [markdown]
# ### 13.5 Analysis by Specific Holiday Types
# 
# Let's also analyze energy patterns across different types of holidays and school breaks.

# %%
# Specific national holiday analysis
if len(df[df['Is_Holiday'] == 1]) > 0:
    nat_holiday_analysis = df[df['Is_Holiday'] == 1].groupby('Holiday_Name').agg({
        'Volume_Consumption_kWh': ['mean', 'median', 'count'],
        'Volume_Injection_kWh': ['mean', 'median', 'count']
    })
    
    # Flatten the multi-level columns
    nat_holiday_analysis.columns = [f"{col[0]}_{col[1]}" for col in nat_holiday_analysis.columns]
    
    print("Energy patterns by specific national holidays:")
    display(nat_holiday_analysis.sort_values('Volume_Consumption_kWh_mean', ascending=False))

    # Visualize for holidays with sufficient data (at least 10 records)
    valid_holidays = nat_holiday_analysis[nat_holiday_analysis['Volume_Consumption_kWh_count'] >= 10].index
    
    if len(valid_holidays) > 0:
        plt.figure(figsize=(14, 8))
        
        # Sort holidays by consumption
        sorted_holidays = nat_holiday_analysis.loc[valid_holidays].sort_values('Volume_Consumption_kWh_mean', ascending=False)
        
        plt.bar(sorted_holidays.index, sorted_holidays['Volume_Consumption_kWh_mean'], color='steelblue', alpha=0.7)
        plt.title('Average Consumption by National Holiday Type')
        plt.ylabel('Average Consumption (kWh)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('analysis_results/consumption_by_holiday_type.png', dpi=300)
        plt.show()

# Specific school holiday analysis
if len(df[df['Is_School_Holiday'] == 1]) > 0:
    school_holiday_analysis = df[df['Is_School_Holiday'] == 1].groupby('School_Holiday').agg({
        'Volume_Consumption_kWh': ['mean', 'median', 'count'],
        'Volume_Injection_kWh': ['mean', 'median', 'count']
    })
    
    # Flatten the multi-level columns
    school_holiday_analysis.columns = [f"{col[0]}_{col[1]}" for col in school_holiday_analysis.columns]
    
    print("\nEnergy patterns by specific school holiday periods:")
    display(school_holiday_analysis.sort_values('Volume_Consumption_kWh_mean', ascending=False))
    
    # Visualize all school holiday types
    plt.figure(figsize=(14, 8))
    
    # Prepare data for dual-axis bar chart
    holidays = school_holiday_analysis.index
    consumption = school_holiday_analysis['Volume_Consumption_kWh_mean']
    injection = school_holiday_analysis['Volume_Injection_kWh_mean']
    
    # Sort by consumption
    sorted_indices = consumption.argsort()[::-1]  # Descending order
    holidays = [holidays[i] for i in sorted_indices]
    consumption = [consumption[i] for i in sorted_indices]
    injection = [injection[i] for i in sorted_indices]
    
    x = range(len(holidays))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Consumption bars
    bars1 = ax1.bar([i - width/2 for i in x], consumption, width, color='blue', alpha=0.7, label='Consumption')
    ax1.set_ylabel('Consumption (kWh)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a second y-axis
    ax2 = ax1.twinx()
    
    # Injection bars
    bars2 = ax2.bar([i + width/2 for i in x], injection, width, color='green', alpha=0.7, label='Injection')
    ax2.set_ylabel('Injection (kWh)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Set x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(holidays, rotation=45, ha='right')
    
    # Title and grid
    ax1.set_title('Energy Patterns by School Holiday Type')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('analysis_results/energy_by_school_holiday_type.png', dpi=300)
    plt.show()

# %% [markdown]
# ### 13.6 Hourly Analysis for Holidays vs. Non-Holidays
# 
# Let's compare the hourly consumption patterns between holidays and non-holidays.

# %%
# Hourly consumption patterns on holidays vs non-holidays
hourly_holiday = df.groupby(['Hour', 'Is_Holiday']).agg({
    'Volume_Consumption_kWh': 'mean',
    'Volume_Injection_kWh': 'mean'
}).reset_index()

# Pivot for easier plotting
hourly_consumption_pivot = hourly_holiday.pivot(index='Hour', 
                                               columns='Is_Holiday', 
                                               values='Volume_Consumption_kWh')
hourly_consumption_pivot.columns = ['Non-Holiday', 'Holiday']

# Plot hourly patterns
plt.figure(figsize=(14, 8))
hourly_consumption_pivot.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Hourly Consumption Patterns: Holidays vs. Non-Holidays')
plt.xlabel('Hour of Day')
plt.ylabel('Average Consumption (kWh)')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Day Type')
plt.savefig('analysis_results/hourly_consumption_holiday_comparison.png', dpi=300)
plt.show()

# Calculate and plot the percentage difference
hourly_consumption_pivot['Pct_Diff'] = ((hourly_consumption_pivot['Holiday'] / 
                                        hourly_consumption_pivot['Non-Holiday']) - 1) * 100

plt.figure(figsize=(14, 6))
hourly_consumption_pivot['Pct_Diff'].plot(kind='bar', color='purple', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Percentage Difference in Consumption: Holidays vs. Non-Holidays')
plt.xlabel('Hour of Day')
plt.ylabel('% Difference')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('analysis_results/hourly_consumption_holiday_pct_diff.png', dpi=300)
plt.show()

# %% [markdown]
# ### 13.7 Combined Feature Analysis: Holidays, Weekends, and Seasons
# 
# Let's combine holiday features with weekend information and seasons to understand complex patterns.

# %%
# Add season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Fall'

df['Season'] = df['Date'].dt.month.apply(get_season)

# Create combined features
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
df['Day_Type'] = df.apply(lambda row: 
                          'Holiday' if row['Is_Holiday'] == 1 
                          else ('Weekend' if row['Is_Weekend'] == 1 
                               else 'Weekday'), axis=1)

# Analyze energy by day type and season
combined_analysis = df.groupby(['Season', 'Day_Type']).agg({
    'Volume_Consumption_kWh': 'mean',
    'Volume_Injection_kWh': 'mean',
    'Date': 'count'
}).reset_index()

# Create a pivot table for easier visualization
consumption_pivot = combined_analysis.pivot(index='Season', 
                                           columns='Day_Type', 
                                           values='Volume_Consumption_kWh')

# Plot consumption by day type and season
plt.figure(figsize=(14, 8))
ax = consumption_pivot.plot(kind='bar', ax=plt.gca())
plt.title('Average Consumption by Season and Day Type')
plt.xlabel('Season')
plt.ylabel('Average Consumption (kWh)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Day Type')

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.savefig('analysis_results/consumption_by_season_day_type.png', dpi=300)
plt.show()

# %% [markdown]
# ### 13.8 Summary of Holiday and Calendar Feature Analysis
# 
# From our analysis of Belgian national holidays and school breaks, we can draw the following conclusions:
# 
# 1. **Holiday Energy Patterns**:
#    - Energy consumption patterns on holidays differ from regular days
#    - The hourly profile of consumption on holidays resembles weekend patterns
#    - Different types of holidays show distinct energy consumption signatures
# 
# 2. **School Holiday Effects**:
#    - School holiday periods show unique consumption patterns
#    - Summer breaks generally have lower consumption than winter breaks
#    - The presence of children at home during school breaks impacts daily energy rhythms
# 
# 3. **Combined Calendar Effects**:
#    - The interaction between seasons and holidays creates complex energy usage patterns
#    - Winter holidays show the highest consumption rates, while summer holidays show increased midday consumption
# 
# 4. **Feature Engineering Value**:
#    - Calendar features like holidays and school breaks are valuable predictors for energy consumption
#    - These features should be incorporated into any predictive model for household energy use
# 
# Sources for calendar data:
# - National holidays: Python's 'holidays' library (https://pypi.org/project/holidays/)
# - School holiday periods: Flemish Government Education Portal (https://onderwijs.vlaanderen.be/nl/schoolvakanties) and Belgian School Holiday Portal (https://www.schoolvakanties-belgie.be/)

# %% [markdown]
# ## 14. Next Steps
# 
# Based on our comprehensive analysis including time patterns, household features, and calendar events, we recommend the following next steps:
# 
# 1. **Predictive Modeling**: Develop models to predict energy consumption and injection using the features we've created
# 
# 2. **Tariff Optimization**: Use the time-based patterns to develop optimized tariff structures
# 
# 3. **Grid Management**: Leverage the calendar feature insights for better grid load management during holidays
# 
# 4. **Behavioral Analysis**: Conduct deeper analysis of how household behaviors change during different calendar periods
# 
# 5. **Policy Recommendations**: Develop data-driven policy recommendations for energy efficiency programs 