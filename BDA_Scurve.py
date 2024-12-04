#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#--------------------------------------------------------------------------------------------------

# GSI - Big Data Analysis System
# LMS and Artificial Intelligence Code Application

# Project S-Curve Generation
# Itamar Nogueira - itamar.nogueira.ext@andritz.com
# Rev.25 - 12.November.2024

        # Code Structure
            # Block 01: Library Imports
            # Block 02: Database Connection
            # Block 03: Linear Distribution
            # Block 04: Normal Distribution
            # Block 05: Baseline Progress
            # Block 06: Customer Forecast Progress
            # Block 07: Actual Progress Calculation and Curve Generation
            # Block 08: ANDRITZ Forecast Progress
            # Block 09: Block to Calculate Days Value Based on PlannedHoursBaseline
            # Block 10: Used Man-days Curve
            # Block 11: Manpower Plan using LMS Data
            # Block 12: Plot Baseline, ANDRITZ Forecast, Customer Forecast, Actual Progress, and Used Mandays S-Curves
            # Block 13: Streamlit App
            # Final Block: Main Execution

#--------------------------------------------------------------------------------------------------

# Block 01: Library Imports

import os
import urllib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.dates as mdates
import streamlit as st

#--------------------------------------------------------------------------------------------------

# Block 02: Database Connection

def get_connection():
    import urllib
    from sqlalchemy import create_engine
    import streamlit as st

    # Get secrets from Streamlit
    DB_USER = st.secrets["DB_USER"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    DB_SERVER = st.secrets["DB_SERVER"]

    # Build connection string
    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}"
    try:
        engine = create_engine(connection_uri)
        with engine.connect() as connection:
            st.success("Connection to the database was successful!")
        return engine
    except Exception as ex:
        st.error(f"Failed to connect to the database. Error: {str(ex)}")
        return None

#-------------------------------------------------------------------------------------------------- 

# Block 03: Linear Distribution

def linear_distribution(start, end, weight, days):
    start = int(start)
    end = int(end)
    interval = end - start
    if interval <= 0:
        return np.zeros(days)
    
    daily_progress = weight / interval
    progress = np.zeros(days)
    progress[start:end] = daily_progress
    return progress

#-------------------------------------------------------------------------------------------------- 

# Block 04: Normal Distribution

def normal_distribution(start, end, weight, days, mean_percent=0.75, variance_percent=0.22):
    start = int(start)
    end = int(end)
    interval = end - start
    if interval <= 0:
        return np.zeros(days)
    
    mean = start + interval * mean_percent
    std_dev = max(interval * np.sqrt(variance_percent), 1e-6)
    
    daily_progress = norm.pdf(np.arange(days), loc=mean, scale=std_dev)
    daily_progress /= daily_progress.sum()
    progress = daily_progress * weight
    return progress

#--------------------------------------------------------------------------------------------------

# Block 05: Baseline Progress

def generate_baseline_curve(df_baseline, distribution_type, project_start, total_days):
    """
    Generate the cumulative baseline S-curve for a given project using the specified distribution type and date range.
    """
    if df_baseline.empty:
        return None, None

    # Calculate total weight (Points)
    total_weight = df_baseline['Points'].sum()

    # Initialize baseline progress array
    baseline_s_curve = np.zeros(total_days)
    for _, row in df_baseline.iterrows():
        baseline_start = max(int((row['BaselineStart'] - project_start).days), 0)
        baseline_end = min(int((row['BaselineEnd'] - project_start).days), total_days)
        weight = row['Points']
        
        if distribution_type == 'linear':
            progress = linear_distribution(baseline_start, baseline_end, weight, total_days)
        else:
            progress = normal_distribution(baseline_start, baseline_end, weight, total_days)
        baseline_s_curve += progress

    # Normalize cumulative baseline curve
    baseline_s_curve_cumulative = np.cumsum(baseline_s_curve)
    today = pd.Timestamp.today()
    days_up_to_today = min((today - project_start).days + 1, total_days)
    baseline_progress_today = (baseline_s_curve_cumulative[days_up_to_today - 1] / total_weight) * 100

    # Normalize to 100%
    baseline_s_curve_cumulative = (baseline_s_curve_cumulative / total_weight) * 100

    return baseline_s_curve_cumulative, baseline_progress_today

#--------------------------------------------------------------------------------------------------

# Block 06: Customer Forecast Progress

def generate_customer_forecast_curve(customer_df, distribution_type, project_start, total_days):
    """
    Generate the cumulative customer forecast S-curve for a given project.
    """
    if customer_df.empty:
        return None, None

    # Calculate start and end days relative to project_start
    customer_df['start_days'] = (customer_df['Cust_PlannedStart'] - project_start).dt.days
    customer_df['end_days'] = (customer_df['Cust_PlannedEnd'] - project_start).dt.days

    # Initialize the customer forecast progress array
    customer_progress = np.zeros(total_days)

    # Distribute points over the period defined by start_days and end_days
    for _, row in customer_df.iterrows():
        start = int(max(int(row['start_days']), 0))
        end = int(min(int(row['end_days']), total_days))
        weight = row['Points']
        
        if distribution_type == 'linear':
            customer_prog = linear_distribution(start, end, weight, total_days)
        else:
            customer_prog = normal_distribution(start, end, weight, total_days)
        
        customer_progress += customer_prog

    # Normalize the cumulative customer forecast S-curve to reach 100% if there's progress data
    total_customer_progress = np.sum(customer_progress)
    if total_customer_progress > 0:
        customer_forecast_curve = np.cumsum(customer_progress) / total_customer_progress * 100
    else:
        customer_forecast_curve = np.zeros(total_days)

    # Calculate the index for the current progress of the customer forecast
    today_index = min((pd.Timestamp.today() - project_start).days, len(customer_forecast_curve) - 1)
    customer_forecast_today = customer_forecast_curve[today_index]

    return customer_forecast_curve, customer_forecast_today

#--------------------------------------------------------------------------------------------------

# Block 07: Actual Progress Calculation and Curve Generation

def calculate_and_generate_actual_progress_curve(activity_df, project_start, total_days, distribution_type):
    """
    Calculate the actual progress percentage and generate the actual progress curve over time based on activity data.
    """
    if activity_df.empty:
        return None, None, None

    # Convert date columns to datetime
    activity_df['ActivityActualStart'] = pd.to_datetime(activity_df['ActivityActualStart'], errors='coerce')
    activity_df['ActivityActualEnd'] = pd.to_datetime(activity_df['ActivityActualEnd'], errors='coerce')
    activity_df['ActivityActualEnd'] = activity_df['ActivityActualEnd'].fillna(pd.Timestamp.today())

    # Calculate the weighted progress for each activity
    activity_df['weighted_progress_value'] = (
        activity_df['Act_Points'] * 
        (activity_df['ActivityPercent'] / 100.0) * 
        (activity_df['ActivityProgress'] / 100.0)
    )

    # Calculate total points considering unique SiteProgressItemNo
    unique_points_df = activity_df.drop_duplicates(subset=['SiteProgressItemNo'])
    total_points = unique_points_df['Act_Points'].sum()

    # Sum the weighted progress values to get the total actual progress
    total_actual_progress = activity_df['weighted_progress_value'].sum()

    # Calculate the actual progress percentage
    actual_progress_percentage = (total_actual_progress / total_points) * 100 if total_points > 0 else 0

    # Calculate start and end days relative to project_start
    activity_df['start_days'] = (activity_df['ActivityActualStart'] - project_start).dt.days
    activity_df['end_days'] = (activity_df['ActivityActualEnd'] - project_start).dt.days

    # Drop rows with NaN in 'start_days' or 'end_days'
    activity_df = activity_df.dropna(subset=['start_days', 'end_days']).copy()

    # Convert 'start_days' and 'end_days' to integers
    activity_df.loc[:, 'start_days'] = activity_df['start_days'].astype(int)
    activity_df.loc[:, 'end_days'] = activity_df['end_days'].astype(int)

    # Initialize progress array
    actual_progress = np.zeros(total_days)

    # Loop through the activity data to accumulate the weighted progress over the date range
    for _, row in activity_df.iterrows():
        start = int(max(row['start_days'], 0))
        end = int(min(row['end_days'], total_days - 1))
        weight = row['weighted_progress_value']

        if start >= end or start >= total_days or end < 0:
            continue  # Ignore invalid or out-of-bounds dates

        # Choose distribution type
        if distribution_type == 'linear':
            progress_distribution = linear_distribution(start, end, weight, total_days)
        else:
            progress_distribution = normal_distribution(start, end, weight, total_days)

        # Accumulate the progress
        actual_progress += progress_distribution

    # Now, scale the actual_progress array to match total_actual_progress
    total_progress_from_distribution = np.sum(actual_progress)

    if total_progress_from_distribution > 0:
        scaling_factor = total_actual_progress / total_progress_from_distribution
        actual_progress = actual_progress * scaling_factor

    # Create cumulative curve for actual progress
    actual_progress_curve = (np.cumsum(actual_progress) / total_points) * 100

    # Generate dates
    dates = pd.date_range(project_start, periods=total_days)

    # Limit the actual progress curve and dates to today's date
    today_index = (pd.Timestamp.today() - project_start).days
    today_index = min(today_index, total_days - 1)

    actual_progress_curve = actual_progress_curve[:today_index + 1]
    dates = dates[:today_index + 1]

    return actual_progress_curve, actual_progress_percentage, dates

#--------------------------------------------------------------------------------------------------

# Block 08: ANDRITZ Forecast Progress

def generate_ANDRITZ_forecast_curve(detailed_df, total_days, project_start, actual_progress_today, distribution_type):
    """
    Generate the detailed forecast S-curve based on replanned data, ensuring it starts at actual_progress_today.
    """
    # The dataframe 'detailed_df' is already provided

    # Convert date columns to datetime
    detailed_df['Replanned_PlannedStart'] = pd.to_datetime(detailed_df['Replanned_PlannedStart'], errors='coerce')
    detailed_df['Replanned_PlannedEnd'] = pd.to_datetime(detailed_df['Replanned_PlannedEnd'], errors='coerce')
    detailed_df['ActualStart'] = pd.to_datetime(detailed_df['ActualStart'], errors='coerce')
    detailed_df['ActualEnd'] = pd.to_datetime(detailed_df['ActualEnd'], errors='coerce')

    # Adjust dates according to specified rules
    today = pd.Timestamp.today()

    for index, row in detailed_df.iterrows():
        replanned_start = row['Replanned_PlannedStart']
        replanned_end = row['Replanned_PlannedEnd']
        actual_start = row['ActualStart']
        actual_end = row['ActualEnd']

        if pd.isna(replanned_start) or pd.isna(replanned_end):
            continue  # Skip rows with invalid replanned dates

        if replanned_start > today and replanned_end > today:
            continue

        elif replanned_start < today < replanned_end and pd.isna(actual_start) and pd.isna(actual_end):
            duration = (replanned_end - replanned_start).days
            detailed_df.at[index, 'Replanned_PlannedStart'] = today
            detailed_df.at[index, 'Replanned_PlannedEnd'] = today + pd.Timedelta(days=duration)
            
        elif replanned_start < today < replanned_end and not pd.isna(actual_start) and pd.isna(actual_end):
            duration = (replanned_end - replanned_start).days
            detailed_df.at[index, 'Replanned_PlannedStart'] = actual_start
            detailed_df.at[index, 'Replanned_PlannedEnd'] = today + pd.Timedelta(days=duration)

        elif replanned_start < today < replanned_end and not pd.isna(actual_start) and not pd.isna(actual_end):
            detailed_df.at[index, 'Replanned_PlannedStart'] = actual_start
            detailed_df.at[index, 'Replanned_PlannedEnd'] = actual_end

        elif replanned_start < today and replanned_end < today and pd.isna(actual_start) and pd.isna(actual_end):
            duration = (replanned_end - replanned_start).days
            detailed_df.at[index, 'Replanned_PlannedStart'] = today
            detailed_df.at[index, 'Replanned_PlannedEnd'] = today + pd.Timedelta(days=duration)
        
        elif replanned_start < today and replanned_end < today and not pd.isna(actual_start) and pd.isna(actual_end):
            detailed_df.at[index, 'Replanned_PlannedStart'] = actual_start
            detailed_df.at[index, 'Replanned_PlannedEnd'] = today
        
        elif replanned_start < today and replanned_end < today and not pd.isna(actual_start) and not pd.isna(actual_end):
            detailed_df.at[index, 'Replanned_PlannedStart'] = actual_start
            detailed_df.at[index, 'Replanned_PlannedEnd'] = actual_end

    # Calculate start and end days relative to the project start
    detailed_df['start_days'] = (detailed_df['Replanned_PlannedStart'] - project_start).dt.days
    detailed_df['end_days'] = (detailed_df['Replanned_PlannedEnd'] - project_start).dt.days

    # Drop rows with NaN in 'start_days' or 'end_days'
    detailed_df = detailed_df.dropna(subset=['start_days', 'end_days']).copy()

    # Convert 'start_days' and 'end_days' to integers
    detailed_df.loc[:, 'start_days'] = detailed_df['start_days'].astype(int)
    detailed_df.loc[:, 'end_days'] = detailed_df['end_days'].astype(int)

    # Initialize the progress array for the detailed forecast
    detailed_progress = np.zeros(total_days)

    # Calculate the distribution for the detailed forecast
    for _, row in detailed_df.iterrows():
        start = int(max(row['start_days'], 0))
        end = int(min(row['end_days'], total_days - 1))

        if start >= end or start >= total_days or end < 0:
            continue  # Ignore invalid or out-of-bounds dates

        weight = 1  # Assuming equal weight for simplicity; adjust if weights are available

        # Choose distribution type
        if distribution_type == 'linear':
            progress = linear_distribution(start, end, weight, total_days)
        else:
            progress = normal_distribution(start, end, weight, total_days)

        detailed_progress += progress

    # Create the cumulative curve for the detailed forecast
    if np.sum(detailed_progress) > 0:
        detailed_s_curve = np.cumsum(detailed_progress)
    else:
        detailed_s_curve = np.zeros(total_days)

    # Filter the curve to start from today
    today_index = (today - project_start).days
    if today_index >= len(detailed_s_curve):
        today_index = len(detailed_s_curve) - 1
    detailed_s_curve_from_today = detailed_s_curve[today_index:]

    # Normalize the forecast curve from today to end from 0 to 1
    if detailed_s_curve_from_today[-1] - detailed_s_curve_from_today[0] != 0:
        detailed_s_curve_normalized = (detailed_s_curve_from_today - detailed_s_curve_from_today[0]) / (detailed_s_curve_from_today[-1] - detailed_s_curve_from_today[0])
    else:
        detailed_s_curve_normalized = np.zeros_like(detailed_s_curve_from_today)

    # Calculate remaining progress to reach 100%
    remaining_progress = 100 - actual_progress_today

    # Scale the normalized curve by remaining progress and shift to start at actual_progress_today
    detailed_s_curve_scaled = detailed_s_curve_normalized * remaining_progress + actual_progress_today

    # Adjust the dates to match the filtered curve length
    dates_filtered = pd.date_range(today, periods=len(detailed_s_curve_scaled))

    return detailed_s_curve_scaled, dates_filtered

#--------------------------------------------------------------------------------------------------

# Block 09: Block to Calculate Days Value Based on PlannedHoursBaseline

def calculate_days_value(engine, project_name):
    """
    Calculate days_value for a given project based on PlannedHoursBaseline in the database.
    """
    query = f"""
    SELECT SUM(PlannedHoursBaseline) / 8 AS days_value
    FROM BDA_SiteProgress
    WHERE ProjectName = '{project_name}' AND Pyfilter = 'Manpower Plan'
    """
    result = pd.read_sql(query, engine)
    days_value = result['days_value'].iloc[0] if not result.empty else 0
    return days_value

#--------------------------------------------------------------------------------------------------

# Block 10: Used Man-days Curve

def generate_used_mandays_curve(workday_df, total_days, project_start, days_value):
    """
    Calculate the used man-days curve for a given project, truncated at today's date.
    """
    # Convert Date to datetime
    workday_df['Date'] = pd.to_datetime(workday_df['Date'], errors='coerce')

    # Initialize a distribution array with zeros for the total number of days
    workday_progress = np.zeros(total_days)
    
    # Distribute Has_hours along the time period
    for _, row in workday_df.iterrows():
        day_index = (row['Date'] - project_start).days
        if 0 <= day_index < total_days:  # Ensure we stay within bounds
            workday_progress[day_index] += row['Has_hours']
    
    # Divide the accumulated progress by the value of days to normalize the distribution
    workday_progress /= days_value if days_value > 0 else 1  # Avoid division by zero
    
    # Calculate the cumulative percentage curve and truncate at today's date
    workday_s_curve = np.cumsum(workday_progress) * 100
    today_index = min((pd.Timestamp.today() - project_start).days, total_days - 1)
    workday_s_curve = workday_s_curve[:today_index + 1]

    return workday_s_curve

#--------------------------------------------------------------------------------------------------

# Block 11: Manpower Plan using LMS Data

def generate_manpower_plan_curve(mp_df, project_start, final_workday_value, distribution_type):
    """
    Generate the manpower plan S-curve based on LMS data, ensuring it starts at final_workday_value.
    """
    # Convert date columns to datetime and drop NaNs
    mp_df['MP_PlannedStart'] = pd.to_datetime(mp_df['MP_PlannedStart'], errors='coerce')
    mp_df['MP_PlannedEnd'] = pd.to_datetime(mp_df['MP_PlannedEnd'], errors='coerce')
    mp_df = mp_df.dropna(subset=['MP_PlannedStart', 'MP_PlannedEnd'])

    # Calculate the total planned hours in man-days
    planned_hours_baseline_sum = mp_df['PlannedHoursBaseline'].sum() / 8  # Assuming an 8-hour workday

    # Get today's date
    today = pd.Timestamp.today().normalize()  # Normalize to remove time component

    # Initialize a list to store the results
    remaining_hours_list = []

    # Initialize total for MP_Remaining
    total_mp_remaining = 0

    # Iterate over each row and apply the logic
    for index, row in mp_df.iterrows():
        mp_planned_start = row['MP_PlannedStart']
        mp_planned_end = row['MP_PlannedEnd']

        # Skip tasks that have already ended
        if mp_planned_end < today:
            continue

        # Adjust the planned start date if it is in the past
        if mp_planned_start < today:
            mp_planned_start = today

        # Calculate remaining man-days for the task
        remaining_days = (mp_planned_end - mp_planned_start).days
        mp_remaining = (remaining_days / 7) * 5  # Convert calendar days to workdays

        # Append the result to the list
        remaining_hours_list.append({
            'SiteProgressItemNo': row['SiteProgressItemNo'],
            'MP_PlannedStart': mp_planned_start,
            'MP_PlannedEnd': mp_planned_end,
            'MP_Remaining': mp_remaining
        })

        # Add to total MP_Remaining
        total_mp_remaining += mp_remaining

    # Convert the results into a DataFrame
    remaining_hours_df = pd.DataFrame(remaining_hours_list)

    # If there is no remaining work, return zeros
    if remaining_hours_df.empty or planned_hours_baseline_sum == 0:
        manpower_plan_s_curve = np.array([final_workday_value])
        mp_dates_filtered = pd.date_range(today, periods=1)
        max_hours_percentage = final_workday_value
        delta_MP = max_hours_percentage - 100
        return remaining_hours_df, mp_dates_filtered, manpower_plan_s_curve, max_hours_percentage, delta_MP

    # Calculate the remaining percentage
    remaining_percentage = (total_mp_remaining / planned_hours_baseline_sum) * 100
    max_hours_percentage = remaining_percentage + final_workday_value
    delta_MP = max_hours_percentage - 100

    # Calculate the total days for the manpower plan curve
    total_days_for_mp = int((remaining_hours_df['MP_PlannedEnd'].max() - today).days) + 1  # Include today
    total_days_for_mp = max(total_days_for_mp, 1)  # Ensure at least 1 day

    # Initialize array to store the cumulative progress
    manpower_plan_progress = np.zeros(total_days_for_mp)

    for _, row in remaining_hours_df.iterrows():
        start_day = int((row['MP_PlannedStart'] - today).days)
        end_day = int((row['MP_PlannedEnd'] - today).days) + 1  # Include end day

        # Ensure start_day and end_day are within valid ranges
        start_day = max(start_day, 0)
        end_day = max(end_day, start_day + 1)

        weight = row['MP_Remaining']  # MP_Remaining to be distributed

        # Truncate indices to array bounds
        start_idx = max(start_day, 0)
        end_idx = min(end_day, total_days_for_mp)

        if start_idx >= end_idx:
            continue  # Skip invalid ranges

        # Choose distribution type
        if distribution_type == 'linear':
            mp_distribution = linear_distribution(start_idx, end_idx, weight, total_days_for_mp)
        else:
            mp_distribution = normal_distribution(start_idx, end_idx, weight, total_days_for_mp)

        # Add the distribution to the overall manpower plan progress
        manpower_plan_progress += mp_distribution

    # Create cumulative curve for the manpower plan
    if manpower_plan_progress.sum() != 0:
        manpower_plan_s_curve = np.cumsum(manpower_plan_progress)
        # Normalize the curve to ensure it reaches exactly remaining_percentage
        manpower_plan_s_curve = manpower_plan_s_curve / manpower_plan_s_curve[-1] * remaining_percentage
    else:
        manpower_plan_s_curve = np.zeros(len(manpower_plan_progress))

    # Shift the curve vertically by final_workday_value
    manpower_plan_s_curve = manpower_plan_s_curve + final_workday_value

    # Generate the corresponding dates (starting from today)
    mp_dates_filtered = pd.date_range(today, periods=len(manpower_plan_s_curve))

    # Return both the DataFrame and the calculated S-curve data for further plotting
    return remaining_hours_df, mp_dates_filtered, manpower_plan_s_curve, max_hours_percentage, delta_MP

#--------------------------------------------------------------------------------------------------

# Block 12: Plot S-Curves

def plot_s_curve(project_name, ylim, 
                 baseline_s_curve_cumulative=None, baseline_progress_today=None,
                 customer_forecast_curve=None, customer_forecast_today=None,
                 detailed_s_curve=None, detailed_dates=None,
                 actual_progress_curve=None, actual_progress_today=None, actual_dates=None,
                 workday_s_curve=None,
                 manpower_plan_s_curve=None, mp_dates_filtered=None, max_hours_percentage=None, delta_MP=None,
                 delta_end_days=None,
                 start_filter_date=None, project_start=None, total_days=None,
                 return_fig=False):  # Add return_fig parameter
    
    """
    Plot the cumulative baseline S-curve and, optionally, the ANDRITZ forecast S-curve, customer forecast S-curve, actual progress, and used man-days.
    """
    # Generate dates for the full range and apply start_filter_date
    dates = pd.date_range(project_start, periods=total_days)
    mask = dates >= start_filter_date
    dates_filtered = dates[mask]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Baseline S-Curve if available
    if baseline_s_curve_cumulative is not None:
        baseline_s_curve_filtered = baseline_s_curve_cumulative[mask]
        plt.plot(dates_filtered, baseline_s_curve_filtered,
                 label=f'Baseline ({baseline_progress_today:.2f}%)', 
                 linestyle='-', color='gray', linewidth=1)
    
    # Plot ANDRITZ Forecast S-Curve if provided
    if detailed_s_curve is not None and detailed_dates is not None:
        detailed_mask = detailed_dates >= start_filter_date
        detailed_dates_filtered = detailed_dates[detailed_mask]
        detailed_s_curve_filtered = detailed_s_curve[detailed_mask]

        # Ensure that the detailed_dates_filtered and detailed_s_curve_filtered are aligned
        plt.plot(detailed_dates_filtered, detailed_s_curve_filtered,
                 label=f'Forecast ANDRITZ - status date (from {actual_progress_today:.2f}%)', 
                 linestyle='--', color='blue', linewidth=1)

    # Plot Customer Forecast S-Curve if provided
    if customer_forecast_curve is not None and customer_forecast_today is not None:
        customer_forecast_curve_filtered = customer_forecast_curve[mask]
        plt.plot(dates_filtered, customer_forecast_curve_filtered,
                 label=f'Customer Forecast ({customer_forecast_today:.2f}%)', 
                 linestyle='-', color='black', linewidth=1)

    # Plot Actual Progress Curve if provided
    if actual_progress_curve is not None and actual_dates is not None:
        actual_mask = (actual_dates >= start_filter_date) & (actual_dates <= pd.Timestamp.today())
        actual_dates_filtered = actual_dates[actual_mask]
        actual_progress_filtered = actual_progress_curve[actual_mask]

        plt.plot(actual_dates_filtered, actual_progress_filtered,
                 label=f'Actual Progress ({actual_progress_today:.2f}%)', 
                 linestyle='-', color='orange', linewidth=1)

    # Plot Used Man-days Curve if provided
    if workday_s_curve is not None:
        # Limit the workday dates to the start_filter_date and today's date
        workday_dates = pd.date_range(project_start, periods=len(workday_s_curve))
        workday_mask = workday_dates >= start_filter_date
        workday_dates_filtered = workday_dates[workday_mask]
        workday_s_curve_filtered = workday_s_curve[workday_mask]
        
        plt.plot(workday_dates_filtered, workday_s_curve_filtered,
                 label=f'Used Mandays ({workday_s_curve_filtered[-1]:.2f}%)',
                 linestyle='-', color='green', linewidth=1)

    # Plot Manpower Plan Curve if provided
    if manpower_plan_s_curve is not None and mp_dates_filtered is not None:
        mp_mask = mp_dates_filtered >= start_filter_date
        mp_dates_filtered = mp_dates_filtered[mp_mask]
        manpower_plan_s_curve_filtered = manpower_plan_s_curve[mp_mask]

        plt.plot(mp_dates_filtered, manpower_plan_s_curve_filtered, 
                 label=f'Mandays Forecast ({max_hours_percentage:.2f}%)',
                 linestyle='--', color='purple', linewidth=1)

        # Add text for Mandays Overrun (delta_MP)
        plt.text(0.95, 0.01, f'Mandays Overrun: {delta_MP:.2f}%', 
                 verticalalignment='bottom', horizontalalignment='right', 
                 transform=plt.gca().transAxes, color='purple', fontsize=10)

        # Add text for Delta Forecast-Baseline
        if delta_end_days is not None:
            plt.text(0.95, 0.05, f'Delta Forecast-Baseline: {delta_end_days} days', 
                     verticalalignment='bottom', horizontalalignment='right', 
                     transform=plt.gca().transAxes, color='blue', fontsize=10)


    # Format the x-axis and set the labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xlabel('Months')
    plt.ylabel('Cumulative Progress (%)')
    
    # Set Y-axis limit
    plt.ylim(0, ylim)
    
    # Title and final plot adjustments
    plt.title(f'Site Progress for Project {project_name}')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if return_fig:
        fig = plt.gcf()
        plt.close(fig)  # Close the figure to prevent duplication in Streamlit
        return fig
    else:
        plt.show()

#--------------------------------------------------------------------------------------------------

# Block 13: Streamlit App

def main():
    st.title("Site Progress S-Curve")

    # Project selection
    project_names = [
        'APP OKI II, WP Wood Processing',
        'APP OKI II, FL Fiberline',
        'APP OKI II, PD Pulp Drying',
        'APP OKI II, EV Evaporation',
        'APP OKI II, RB Recovery Boiler',
        'APP OKI II, BG Bark Gasifier',
        'APP OKI II, RC Recausticizing',
        'APP OKI II, LK Lime Kiln'
    ]

    # Create a dropdown selection box
    selected_project = st.selectbox("Select Project", project_names)

    # When a project is selected, generate the S-curve
    if selected_project:
        st.write(f"Generating S-curve for project: {selected_project}")

        # Get the database connection
        engine = get_connection()
        if engine:
            distribution_type = 'linear'
            range_type = 'full'
            start_filter_date = pd.to_datetime('2023-06-01')

            project_name = selected_project

            #--------------------------------------------
            # Calculate days_value for the project
            days_value = calculate_days_value(engine, project_name)

            #--------------------------------------------
            # Block 05: Load Baseline Data
            s_curve_query = f"""
            SELECT BaselineStart, BaselineEnd, Points
            FROM BDA_SiteProgress
            WHERE ProjectName = '{project_name}' AND 
                  Pyfilter IN ('Schedule ANDRITZ')
            """
            df_baseline = pd.read_sql(s_curve_query, engine)

            #--------------------------------------------
            # Block 06: Load Customer Forecast Data
            customer_forecast_query = f"""
            SELECT PlannedStart AS Cust_PlannedStart, PlannedEnd AS Cust_PlannedEnd, Points
            FROM BDA_SiteProgress
            WHERE ProjectName = '{project_name}' AND 
                  Pyfilter = 'Schedule Customer'
            """
            customer_df = pd.read_sql(customer_forecast_query, engine)

            #--------------------------------------------
            # Block 07: Load Actual Progress Data
            actual_progress_query = f"""
            SELECT SiteProgressItemNo, ActivityActualStart, ActivityActualEnd, Points AS Act_Points, ActivityPercent, ActivityProgress
            FROM BDA_SiteProgressActivity
            WHERE ProjectName = '{project_name}' AND 
                  Pyfilter IN ('Schedule ANDRITZ')
            """
            activity_df = pd.read_sql(actual_progress_query, engine)

            #--------------------------------------------
            # Block 08: Load Detailed Replanned Data for ANDRITZ Forecast
            detailed_query = f"""
            SELECT PlannedStart AS Replanned_PlannedStart, PlannedEnd AS Replanned_PlannedEnd, ActualStart, ActualEnd
            FROM BDA_SiteProgress
            WHERE ProjectName = '{project_name}' AND 
                  Pyfilter IN ('Schedule ANDRITZ')
            """
            detailed_df = pd.read_sql(detailed_query, engine)

            #--------------------------------------------
            # Block 09: Load Used Man-days
            workday_query = f"""
            SELECT Date, Has_hours, CustomText06
            FROM BDA_WorkDay
            WHERE ProjectName = '{project_name}' AND 
                  Has_hours = 1 AND CustomText06 = 'No'
            """
            workday_df = pd.read_sql(workday_query, engine)

            #--------------------------------------------
            # Block 11: Load Manpower Plan Data
            mp_query = f"""
            SELECT SiteProgressItemNo, PlannedStart AS MP_PlannedStart, PlannedEnd AS MP_PlannedEnd, PlannedHoursBaseline, PlannedHours, Points AS MP_Points
            FROM BDA_SiteProgress
            WHERE ProjectName = '{project_name}' AND 
                  Pyfilter IN ('Manpower Plan')
            ORDER BY SiteProgressItemNo
            """
            mp_df = pd.read_sql(mp_query, engine)

            #--------------------------------------------
            # Convert dates to datetime
            df_baseline['BaselineStart'] = pd.to_datetime(df_baseline['BaselineStart'], errors='coerce')
            df_baseline['BaselineEnd'] = pd.to_datetime(df_baseline['BaselineEnd'], errors='coerce')
            df_baseline = df_baseline.dropna(subset=['BaselineStart', 'BaselineEnd', 'Points'])

            customer_df['Cust_PlannedStart'] = pd.to_datetime(customer_df['Cust_PlannedStart'], errors='coerce')
            customer_df['Cust_PlannedEnd'] = pd.to_datetime(customer_df['Cust_PlannedEnd'], errors='coerce')
            customer_df = customer_df.dropna(subset=['Cust_PlannedStart', 'Cust_PlannedEnd', 'Points'])

            #--------------------------------------------
            # Check if datasets are empty
            if df_baseline.empty and customer_df.empty and activity_df.empty and detailed_df.empty:
                st.write(f"No data available for project: {project_name}")
                return

            #--------------------------------------------
            # Calculate project_start and project_end based on available data
            dates_list = []
            if not df_baseline.empty:
                dates_list.extend([df_baseline['BaselineStart'].min(), df_baseline['BaselineEnd'].max()])
            if not customer_df.empty:
                dates_list.extend([customer_df['Cust_PlannedStart'].min(), customer_df['Cust_PlannedEnd'].max()])
            if not detailed_df.empty:
                detailed_df['Replanned_PlannedStart'] = pd.to_datetime(detailed_df['Replanned_PlannedStart'], errors='coerce')
                detailed_df['Replanned_PlannedEnd'] = pd.to_datetime(detailed_df['Replanned_PlannedEnd'], errors='coerce')
                detailed_df = detailed_df.dropna(subset=['Replanned_PlannedStart', 'Replanned_PlannedEnd'])
                dates_list.extend([detailed_df['Replanned_PlannedStart'].min(), detailed_df['Replanned_PlannedEnd'].max()])
            if not activity_df.empty:
                activity_df['ActivityActualStart'] = pd.to_datetime(activity_df['ActivityActualStart'], errors='coerce')
                activity_df['ActivityActualEnd'] = pd.to_datetime(activity_df['ActivityActualEnd'], errors='coerce')
                dates_list.extend([activity_df['ActivityActualStart'].min(), activity_df['ActivityActualEnd'].max()])

            # Ensure dates_list is not empty
            dates_list = [date for date in dates_list if pd.notnull(date)]
            if not dates_list:
                st.write(f"No date data available for project: {project_name}")
                return

            project_start = min(dates_list)
            project_end = max(dates_list)

            #--------------------------------------------
            # Calculate total_days
            total_days = (project_end - project_start).days + 1

            #--------------------------------------------
            # Block 07: Calculate Actual Progress Percentage and Curve
            actual_progress_curve, actual_progress_today, actual_dates = calculate_and_generate_actual_progress_curve(
                activity_df=activity_df,
                project_start=project_start,
                total_days=total_days,
                distribution_type=distribution_type
            )

            #--------------------------------------------
            # Block 08: Generate ANDRITZ Forecast S-curve
            detailed_s_curve, detailed_dates = generate_ANDRITZ_forecast_curve(
                detailed_df=detailed_df,
                total_days=total_days,
                project_start=project_start,
                actual_progress_today=actual_progress_today,
                distribution_type=distribution_type
            )

            #--------------------------------------------
            # Block 06: Generate Customer Forecast S-curve if customer forecast data is available
            customer_forecast_curve, customer_forecast_today = generate_customer_forecast_curve(
                customer_df=customer_df,
                distribution_type=distribution_type,
                project_start=project_start,
                total_days=total_days
            )

            #--------------------------------------------
            # Block 05: Generate Baseline S-curve if baseline data is available
            baseline_s_curve_cumulative, baseline_progress_today = generate_baseline_curve(
                df_baseline=df_baseline, 
                distribution_type=distribution_type, 
                project_start=project_start,
                total_days=total_days
            )
            
            #--------------------------------------------
            # Block 10: Generate Used Man-days Curve
            workday_s_curve = generate_used_mandays_curve(
                workday_df=workday_df,
                total_days=total_days,
                project_start=project_start,
                days_value=days_value
            )
            
            #--------------------------------------------
            # Block 11: Generate Manpower Plan Curve
            final_workday_value = workday_s_curve[-1] if workday_s_curve is not None and len(workday_s_curve) > 0 else 0

            remaining_hours_df, mp_dates_filtered, manpower_plan_s_curve, max_hours_percentage, delta_MP = generate_manpower_plan_curve(
                mp_df=mp_df,
                project_start=project_start,
                final_workday_value=final_workday_value,
                distribution_type=distribution_type
            )

            #--------------------------------------------
            # Calculate Delta Forecast-Baseline in days
            if not df_baseline.empty and not detailed_df.empty:
                baseline_end_date = df_baseline['BaselineEnd'].max()
                st.write(f"Baseline End Date: {baseline_end_date}")
                forecast_end_date = detailed_df['Replanned_PlannedEnd'].max()
                st.write(f"Forecast End Date: {forecast_end_date}")
                delta_end_days = (forecast_end_date - baseline_end_date).days
                st.write(f"Delta Forecast-Baseline in Days: {delta_end_days}")
            else:
                delta_end_days = None  # or set to 0 if preferred

            #--------------------------------------------
            # Determine ylim for plotting
            ylim_values = []
            if baseline_s_curve_cumulative is not None:
                ylim_values.append(baseline_s_curve_cumulative.max())
            if customer_forecast_curve is not None:
                ylim_values.append(customer_forecast_curve.max())
            if detailed_s_curve is not None:
                ylim_values.append(detailed_s_curve.max())
            if actual_progress_curve is not None:
                ylim_values.append(actual_progress_curve.max())
            if workday_s_curve is not None:
                ylim_values.append(workday_s_curve.max())
            if manpower_plan_s_curve is not None:
                ylim_values.append(manpower_plan_s_curve.max())
            ylim_max = max(ylim_values) if ylim_values else 100
            ylim = 100 if range_type == '100' else ylim_max + 10

            #--------------------------------------------
            # Plot all S-curves, including used man-days and manpower plan
            fig = plot_s_curve(
                project_name=project_name,
                ylim=ylim,
                baseline_s_curve_cumulative=baseline_s_curve_cumulative,
                baseline_progress_today=baseline_progress_today,
                customer_forecast_curve=customer_forecast_curve,
                customer_forecast_today=customer_forecast_today,
                detailed_s_curve=detailed_s_curve,
                detailed_dates=detailed_dates,
                actual_progress_curve=actual_progress_curve,
                actual_progress_today=actual_progress_today,
                actual_dates=actual_dates,
                workday_s_curve=workday_s_curve,
                manpower_plan_s_curve=manpower_plan_s_curve,
                mp_dates_filtered=mp_dates_filtered,
                max_hours_percentage=max_hours_percentage,
                delta_MP=delta_MP,
                delta_end_days=delta_end_days,
                start_filter_date=start_filter_date,
                project_start=project_start,
                total_days=total_days,
                return_fig=True
            )

            # Display the figure in Streamlit
            st.pyplot(fig)

#--------------------------------------------------------------------------------------------------

# Run the Streamlit app
if __name__ == "__main__":
    main()

