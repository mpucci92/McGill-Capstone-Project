# Author: Michael Pucci
# Section 1: Importing Libraries

import pandas as pd
import numpy as np
import sys, os
from haversine import haversine
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Section 2 - Importing data - Coordinates Dataset
# Load the first X lines - law of large numbers good enough approximation of the dataset.

path = "D:\Itinerum Data\MTLTrajet2018\MTLTrajet2018\coordinates.csv"
num_rows = 500000
data_path = path

raw = pd.read_csv(data_path, nrows=num_rows) 

# Section 3 - Cleaning the data - Dropping useless columns
columns_todrop = ['altitude','direction','mode_detected','point_type','h_accuracy','v_accuracy','acceleration_x','acceleration_y','acceleration_z']

raw.drop(columns_todrop,axis=1,inplace=True)

# Renaming Speed Column to Instantaneous Speed
raw.rename(columns={"speed":'instantaneous_speed'},inplace=True)

# Processing timestamp columns into datetime object type
def convert(time):
    return datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')

temp = raw.timestamp_UTC.apply(convert)

raw.drop(['timestamp_UTC'],axis=1,inplace=True)

raw['timestamp_UTC'] = temp

columns_tosort = ['uuid','timestamp_UTC']
raw = raw.sort_values(columns_tosort) 

# Resetting the index to follow sequential order
raw.reset_index(drop=True,inplace=True)

# Value Counts for the unique User IDs
uuid_sortedfrequency = (raw.uuid.value_counts()).index.tolist() # List of UUIDs sorted in descending order by frequency

frequency_threshold = 4      # Keep UUIDs that have greater than 4 data points

uuid_sortedfrequency = raw.uuid.value_counts()[raw.uuid.value_counts()>frequency_threshold].index.tolist()

# Section 4: Feature Engineering

raw = (raw[raw.instantaneous_speed < 40]) # Sorting the speeds to ensure less than 140 km/hour.

raw['time_difference'] = raw.timestamp_UTC.diff().dt.seconds + 1   #Time difference is in seconds

raw['insta_speed_difference'] = raw.instantaneous_speed.diff() # This will subtract the speed in postiion 2 - position 1

raw['instantaneous_acceleration'] = raw.insta_speed_difference / raw.time_difference  # divide the speed difference with the time difference to get the acceleration

raw['insta_acceleration_difference'] = raw.instantaneous_acceleration.diff()   # This will subtract the acceleration in position 2 - position 1

raw['instantaneous_jerk'] = raw.insta_acceleration_difference / raw.time_difference  # Jerk (4th derivative of Transportation)

coordinates =  list(zip(raw.latitude, raw.longitude))

raw['LatLon'] = coordinates

# Identifying the time of the day

def hours(x):
    if x>=6 and x<=9:
        return 'MR'             # Morning Rush
    elif x>=15 and x<=18:
        return 'NR'             #Night rush
    else:
        return 'S' #Stationary
    
raw['RushhourType'] = raw.timestamp_UTC.dt.hour.apply(hours)  

# Identifying when public transit is closed or open

def closed_transit(x):
    if x >=1 and x<=5:
        return 1        # Closed Transit
    else:
        return 0        # Open Transit

raw['ClosedTransit'] = raw.timestamp_UTC.dt.hour.apply(closed_transit)

# Defining the Bearing (Direction) Function
import math
def Bearing(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1;
    y = math.sin(dLon) * math.cos(lat2);
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    brng = np.degrees(math.atan2(y, x))
    if brng < 0:
        brng+= 360
    return brng  


# Adding the Distance between points and Direction between points Features

direction = [np.nan]
vals = [np.nan]
LL_col = pd.DataFrame()
LL_col['LL'] = raw.LatLon.values

for i in range(1, raw.shape[0]):
    
    pair1 = LL_col['LL'][i-1]
    pair2 = LL_col['LL'][i]
    val = (haversine(pair1,pair2,unit='m'))
    vals.append(val)
    direction_value = Bearing(pair1[0],pair1[1],pair2[0],pair2[1])
    direction.append(direction_value)
    
raw['Distance'] = vals
raw['Direction'] = direction

# Average Velocity Feature
raw['Average Velocity'] = raw.Distance/raw.time_difference  # Average Velocity Feature #

# Average Acceleration Feature
raw['Average Acceleration'] = raw['Average Velocity'] / raw.time_difference  # Average Accelration Feature #

# Determining if point is on a weekday or weekend

raw['Weekends'] = raw.timestamp_UTC.dt.day_name()
raw.Weekends = raw.Weekends.replace(to_replace=['Saturday','Sunday'], value=1) # Weekends denoted with value of 1. 
raw.Weekends = raw.Weekends.replace(to_replace=['Monday','Tuesday','Wednesday','Thursday','Friday'], value=0) # Weekdays denoted with a value of 0. 

# Determing what time of the day coordinate is in

def time_of_day(x):
    if x>= 5 and x<10:
        return 'morning'
    elif x>=10 and x<12:
        return 'lunch'
    elif x>=12 and x<16:
        return 'afternoon'
    elif x>=16 and x<18:
        return 'evening'
    elif ((x>=18 and x<=24) or (x>24 and x<5)):
        return 'night'
    else:
        return np.nan

raw['TimeOfDay'] = raw.timestamp_UTC.dt.hour.apply(time_of_day)

# Convert Latitude and Longitude into x,y,z coordinates #

x_coordinate = []
y_coordinate = []
z_coordinate = []

R = 6371000 # radius of earth in meters 

import math 
for i in range(len(raw)):
   
    x = R * math.cos(raw.latitude.iloc[i]) * math.cos(raw.longitude.iloc[i])

    y = R * math.cos(raw.latitude.iloc[i]) * math.sin(raw.longitude.iloc[i])

    z = R * math.sin(raw.latitude.iloc[i])

       
    x_coordinate.append(x)
    y_coordinate.append(y)
    z_coordinate.append(z)

raw['x_coordinate'] = x_coordinate
raw['y_coordinate'] = y_coordinate
raw['z_coordinate'] = z_coordinate

# Fill NaNs with 0's
fill_nans = 0
raw.fillna(fill_nans,inplace=True)

# Section 5: Determining Segment Stops to classify Trip Time Series

# Part 1: Identifying stops based on prompt_csv dataset

# sample of first 5 unique user ID - TEMP_DF is the new dataframe moving forward

# Select most frequent UUID

# Number of UUIDS
uuid_length = 5

# len(uuid_sortedfrequency)

for i in range(0,5):
    user_uuid = uuid_sortedfrequency[i]

# Manual UUID Selection: 
#user_uuid = '003502B3-82CE-46C3-96E1-68F92A14C9D5'

    temp_df = raw[raw['uuid'] == user_uuid]

    # Copy of original dataframe #
    eval_df = temp_df

    prompt_datapath = 'D:\Itinerum Data\MTLTrajet2018\MTLTrajet2018\prompt_responses.csv'

    prompt = pd.read_csv(prompt_datapath)

    prompt.rename({'user_uuid':'uuid'},axis=1,inplace=True)

    cols_todrop = ['prompt_uuid', 'mode', 'mode_id', 'purpose', 'purpose_id','displayed_at_epoch', 'recorded_at_UTC','recorded_at_epoch', 'latitude', 'longitude']

    prompt.drop(cols_todrop,axis=1,inplace=True)

    def convert(s):
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

    temp = prompt.displayed_at_UTC.apply(convert)

    lbound_timestamp = temp-timedelta(seconds=5)

    rbound_timestamp = temp+timedelta(seconds=5)

    prompt['lb_timestamp'] = lbound_timestamp
    prompt['rb_timestamp'] = rbound_timestamp
    prompt['timestamp_UTC'] = temp

    prompt.drop(['displayed_at_UTC'],axis=1,inplace=True)

    # Selecting data for 1 day from prompt dataset #

    start = temp_df.head(1).timestamp_UTC
    start_time = pd.to_datetime(start.values[0]).date()
    end = temp_df.tail(1).timestamp_UTC
    end_time = pd.to_datetime(end.values[0]).date()
    end_time = end_time +timedelta(days=1)
    end_time = pd.Timestamp(end_time)
    
    prompt = prompt[prompt.timestamp_UTC <= end_time]

    times = list(zip(prompt.lb_timestamp,prompt.rb_timestamp))

    stops = []

    for i in range(len(times)):
        val = temp_df.loc[temp_df.loc[:, 'timestamp_UTC'].between(times[i][0],times[i][1]), : ]
        stops.append(val)

    clean_time = []

    for val in stops:
        if val.empty:
            continue
        elif not val.empty:
            clean_time.append(val)

    indexes = []
    for x in clean_time:
        indexes.append(x.index)

    index_ = []
    for x in indexes:
        index_.append((x.values))

    index_values = []
    for i in range(len(index_)):
        val = index_[i][0]
        index_values.append(val)

    flag = []

    for x in temp_df.index.values:
        if x in index_values:
            flag.append(1)
        else:
            flag.append(0)

    temp_df['Application_stop'] = flag

    # Part 2: Signal Loss Stops

    signal_loss = []

    threshold_loss = np.percentile(temp_df.time_difference,97.5)

    for val in temp_df.time_difference:
        if val > threshold_loss:
            signal_loss.append(1)
        elif val <= threshold_loss:
            signal_loss.append(0)

    temp_df['signal_loss'] = signal_loss

    speed_threshold = 0.5      # Units: m/s

    temp_df['User_stopped'] = (temp_df['Average Velocity'] < speed_threshold).astype(int) # True is 1, False is 0.

    signal_loss_stops = []

    for x in temp_df.index.values:
        if x in temp_df[(temp_df.signal_loss == 1) & (temp_df.User_stopped == 1)].index.values:
            signal_loss_stops.append(1)
        else:
            signal_loss_stops.append(0)

    for i in range(len(signal_loss_stops)):
        if signal_loss_stops[i] == 1:
            signal_loss_stops[(i-1)] = 1
            signal_loss_stops[i] = 0

    temp_df['signal_loss_stops'] = signal_loss_stops

    # Can omit this cell - if would like to keep these columns #

    temp_df.drop(['signal_loss','User_stopped'],axis=1,inplace=True)

    # Part 3 - Stops at Home, Work, Study Proximity ###

    survey_path = "D:\Itinerum Data\MTLTrajet2018\MTLTrajet2018\survey_responses.csv"
    survey_df = pd.read_csv(survey_path)

    survey_df = survey_df[survey_df['uuid'] == user_uuid]

    survey_df.drop(['age_bracket_id', 'gender_id', 'member_type_id', 'document_id',
           'num_of_people', 'num_of_cars', 'num_of_minors', 'email', 'model',
           'participation_code', 'licence_id', 'referrer_id', 'travel_mode_work',
           'travel_mode_alt_work', 'travel_mode_study', 'travel_mode_alt_study',
           'created_at_UTC', 'created_at_epoch', 'use_notification_id', 'version',
           'os', 'osversion','location_home', 'location_study', 'location_work'],axis=1,inplace=True)

    if len(survey_df) < 1:
        print("UUID is not in the list")
    else:
        print("UUID is in list")

    location_home = list(zip(survey_df.location_home_lat,survey_df.location_home_lon))
    location_study =  list(zip(survey_df.location_study_lat,survey_df.location_study_lon))
    location_work =  list(zip(survey_df.location_work_lat,survey_df.location_work_lon))

    vals_location_home = [np.nan]
    LL_col = pd.DataFrame()
    LL_col['LL'] = temp_df.LatLon.values

    for i in range(0,temp_df.shape[0]):

        pair1 = LL_col['LL'][i]

        pair2 = location_home[0] # needed or else haversine will mess up

        val = (haversine(pair1,pair2,unit='m'))
        vals_location_home.append(val)

    vals_location_study = [np.nan]
    LL_col = pd.DataFrame()
    LL_col['LL'] = temp_df.LatLon.values

    for i in range(0,temp_df.shape[0]):

        pair1 = LL_col['LL'][i]

        pair2 = location_study[0] # needed or else haversine will mess up

        val = (haversine(pair1,pair2,unit='m'))

        vals_location_study.append(val)
    
    vals_location_work = [np.nan]
    LL_col = pd.DataFrame()
    LL_col['LL'] = temp_df.LatLon.values

    for i in range(0,temp_df.shape[0]):

        pair1 = LL_col['LL'][i]

        pair2 = location_work[0] # needed or else haversine will mess up

        val = (haversine(pair1,pair2,unit='m'))
        vals_location_work.append(val)

    location_home = np.delete(np.array(vals_location_home),0)
    loc_home_flag = []

    if (np.sum(np.isnan(location_home))) > 1:
        None
    else:
        for i in range(len(temp_df)):
            if (temp_df['Average Velocity'].iloc[i] < 0.5) & (location_home[i] < 10):
                loc_home_flag.append(1)
            else:
                loc_home_flag.append(0)
        
    temp_df['Home_location_distance'] = loc_home_flag  # Less than 10 meters away from home

    location_study = np.delete(np.array(vals_location_study),0)
    loc_study_flag = []

    if (np.sum(np.isnan(location_study))) > 1:
        None
    else:
        for i in range(len(temp_df)):
            if (temp_df['Average Velocity'].iloc[i] < 0.5) & (location_study[i] < 10):
                loc_study_flag.append(1)
            else:
                loc_study_flag.append(0)

        temp_df['Study_location_distance'] = loc_study_flag  # Less than 10 meters away from home

    location_work = np.delete(np.array(vals_location_work),0)
    loc_work_flag = []

    if (np.sum(np.isnan(location_work))) > 1:
        None
    else:
        for i in range(len(temp_df)):
            if (temp_df['Average Velocity'].iloc[i] < 0.5) & (location_work[i] < 10):
                loc_work_flag.append(1)
            else:
                loc_work_flag.append(0)

        temp_df['Work_location_distance'] = loc_work_flag  # Less than 10 meters away from home

    # Part 4: HDBSCAN Clustering Stops
    import hdbscan

    values = temp_df[['latitude', 'longitude']].values
    rads = np.radians(values)

    earth_radius_km = 6371
    epsilon = (0.005 / earth_radius_km)   #calculate 5 meter epsilon threshold
    min_cluster_size = 5
    metric = 'haversine'
    cluster_selection_method = 'eom'

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, cluster_selection_epsilon=epsilon, cluster_selection_method = cluster_selection_method)

    c = clusterer.fit(rads)

    cluster_labels = c.labels_ # https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html VERIFY THIS DOCUMENTATION
    cluster_outliers = c.outlier_scores_

    # Identifying large clusters (clusters with 20 or more data points)
    large_clusters = pd.Series(cluster_labels).value_counts(sort=True)[pd.Series(cluster_labels).value_counts() > 20].index.tolist()

    temp_df['cluster_labels'] = cluster_labels
    temp_df['cluster_outliers'] = cluster_outliers

    large_cluster = []

    for i in range(len(temp_df)):
        if temp_df.cluster_labels.iloc[i] in large_clusters:
            large_cluster.append(1)
        else:
            large_cluster.append(0)

    temp_df['large_cluster'] = large_cluster

    # Directional Change
    temp1 = temp_df.Direction.diff()

    temp1.fillna(0,inplace=True)

    directional_change = []

    for val in temp1:
        if val >=0:
            directional_change.append(1)
        elif val < 0:
            directional_change.append(-1)

    temp_df['directional_change'] = directional_change

    # Abrupt Changes
    abrupt_dir_changes = temp_df.Direction.diff() / temp_df.time_difference

    abrupt_dir_changes.fillna(0,inplace=True)

    temp_df['abrupt_directional_change'] = abrupt_dir_changes

    abrupt_changes = []

    for x in temp_df['abrupt_directional_change']:

        if x < np.percentile(abrupt_dir_changes,2.5):
            abrupt_changes.append(1)

        elif x > np.percentile(abrupt_dir_changes,97.5):
            abrupt_changes.append(1)

        else:
            abrupt_changes.append(0)

    temp_df['abrupt_change_in_direction'] = abrupt_changes

    flags = []

    for i in range(len(temp_df)):

        if ((temp_df['cluster_labels'].iloc[i] != -1) & (temp_df['large_cluster'].iloc[i] == 1) & (temp_df['abrupt_change_in_direction'].iloc[i] == 1)
           & (temp_df['Average Velocity'].iloc[i] < 0.2) & (temp_df.time_difference.iloc[i] > 120)):

            flags.append(1)
        else:
            flags.append(0)

    temp_df['Cluster_trip_stops'] = flags

    temp_df.drop(['cluster_labels','cluster_outliers','directional_change',
                  'abrupt_directional_change','abrupt_change_in_direction'],axis=1,inplace=True)

    fill_nans = 0

    temp_df.fillna(fill_nans,inplace=True)


# Section 6: Segmenting Time Series into Trips and Passing through Isolation Forest

    if 'Study_location_distance' in temp_df.columns:
        stop_list = temp_df[(temp_df['Application_stop'] == 1)|((temp_df['signal_loss_stops'] == 1))
                           |((temp_df['Home_location_distance'] == 1))|((temp_df['Study_location_distance'] == 1))
                           |((temp_df['Work_location_distance'] == 1))|((temp_df['Cluster_trip_stops'] == 1))
                           ].index.tolist()

    else:
        stop_list = temp_df[(temp_df['Application_stop'] == 1)|((temp_df['signal_loss_stops'] == 1))
                           |((temp_df['Home_location_distance'] == 1))|((temp_df['Work_location_distance'] == 1))
                            |((temp_df['Cluster_trip_stops'] == 1))
                           ].index.tolist()

# 1. Application_stop 
# 2. signal_loss_stops
# 3. Home_location_distance
# 4. Study_location_distance
# 5. Work_location_distance
# 6. Cluster_trip_stops
    stop_agg = []

    for x in temp_df.index:
        if x in stop_list:
            stop_agg.append(1)
        else:
            stop_agg.append(0)

    temp_df['Trip_ends'] = stop_agg # stops for trips end at 1 values.

# Part 7: Feeding in Trip Segments into Isolation Forest

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    from sklearn.ensemble import IsolationForest
    
    outliers = [] # Contains the outlier data indexes can be analyzed later to clean from dataset #
    predictions = [] # Contains all the prediction scores from the Isolation Forest #
    uuid_list = [] # Contains the USER UUIDs
    
    trip_length = 5
    
    import pprint
    
    stop = len(stop_list)
    
    for i in range(0,5):
        try:
            if (i == 0):
                isolation_df = temp_df.loc[:stop_list[i]]
                
                if len(isolation_df) < trip_length:                             # Ensures only analyzing trips of size greater than 5
                    continue
            
            else: 
                isolation_df = temp_df.loc[(stop_list[i-1]+1):stop_list[i]]
                
                if len(isolation_df) < trip_length:                             
                    continue
        
        except IndexError as e:                                                 # Ensures only analyzing trips of size greater than 5
            pass
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        isolation_numeric = isolation_df.select_dtypes(include=numerics)
        
         # Standard Scaler #
        
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(isolation_numeric)
        
        isolation_numeric = pd.DataFrame(scaled_df, columns=['latitude', 'longitude', 'instantaneous_speed', 'time_difference',
           'insta_speed_difference', 'instantaneous_acceleration',
           'insta_acceleration_difference', 'instantaneous_jerk', 'ClosedTransit',
           'Distance', 'Direction', 'Average Velocity', 'Average Acceleration',
           'Weekends', 'x_coordinate', 'y_coordinate', 'z_coordinate',
           'Application_stop', 'signal_loss_stops', 'Home_location_distance',
           'Work_location_distance', 'large_cluster', 'Cluster_trip_stops',
           'Trip_ends'])
        
                
        # Isolation Forest Parameters - derived from GridSearch #
        n_estimators = 100
        max_samples = 100
        contamination = 0.1
        bootstrap = True
        
        features_topredict = ['x_coordinate','y_coordinate','z_coordinate','Average Velocity','Average Acceleration']
        max_features = len(features_topredict)
        
        clf = IsolationForest(bootstrap = True, random_state=0, n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features= max_features).fit(isolation_numeric[features_topredict])
        
         # Specify feature columns names below: 
        
        testing = list(zip(isolation_numeric.x_coordinate,isolation_numeric.y_coordinate,isolation_numeric.z_coordinate,
                           isolation_numeric['Average Velocity'],isolation_numeric['Average Acceleration']))
        
        
        test_array = np.array(testing)
        
        
        prediction = clf.predict(test_array)
        
        predictions.append(prediction)
        
        isolation_numeric['anomaly']= prediction
        
        outlier = isolation_numeric.loc[isolation_numeric['anomaly']==-1]
           
        outliers.append(outlier.index)
        uuid_list.append(isolation_df.uuid.iloc[i])
    
print(outliers)
print(uuid_list)
    





