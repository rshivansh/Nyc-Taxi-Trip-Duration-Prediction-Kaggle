import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import timedelta


pd.set_option('display.max_columns', None)

def datetime_features(df):

	df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
	df.loc[:, 'pickup_date'] = df['pickup_datetime'].dt.date
	df.loc[:,'pickup_hour'] = df['pickup_datetime'].dt.hour
	df.loc[:,'pickup_min'] = df['pickup_datetime'].dt.minute
	df.loc[:,'pickup_weekday'] = df['pickup_datetime'].dt.weekday
	df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']
	df.loc[:, 'pickup_dayofyear'] = df['pickup_datetime'].dt.dayofyear
	df.loc[:, 'pickup_hour_weekofyear'] = df['pickup_datetime'].dt.weekofyear
	df.loc[:, 'pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
	df.loc[:, 'pickup_dayofyear'] = df['pickup_datetime'].dt.dayofyear
	return df

def ft_geo_center(df):
	df.loc[:, 'center_latitude'] = (df['pickup_latitude'].values + df['dropoff_latitude'].values) / 2
	df.loc[:, 'center_longitude'] = (df['pickup_longitude'].values + df['dropoff_longitude'].values) / 2
	return df

def haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
	pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = map(np.radians, (pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude))
	AVG_EARTH_RADIUS = 6371 
	lng_diff = dropoff_longitude - pickup_longitude
	lat_diff = dropoff_latitude - pickup_latitude
	distance = np.sin(lat_diff * 0.5) ** 2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(lng_diff * 0.5) ** 2
	haversine_distance = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(distance))
	return haversine_distance

def ft_degree(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
	lng_delta_rad = np.radians(dropoff_longitude - pickup_longitude)
	pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = map(np.radians, (pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude))
	y = np.sin(lng_delta_rad) * np.cos(dropoff_latitude)
	x = np.cos(pickup_latitude) * np.sin(dropoff_latitude) - np.sin(pickup_latitude) * np.cos(dropoff_latitude) * np.cos(lng_delta_rad)
	radian = np.arctan2(y,x)
	return np.degrees(radian)

def ft_pca(df,pca):
	df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]
	df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]
	df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
	df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
	return df

def clustering(df,kmeans):
	df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
	df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
	return df

def manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_distance(lat1, lng1, lat1, lng2)
    b = haversine_distance(lat1, lng1, lat2, lng1)
    return a + b

def ft_holidays(df):
	df['pickup_holiday'] = pd.to_datetime(df.pickup_datetime.dt.date).isin(holidays)
	df['pickup_holiday'] = df.pickup_holiday.map(lambda x: 1 if x == True else 0)

	# If day is before or after a holiday
	df['pickup_near_holiday'] = (pd.to_datetime(df.pickup_datetime.dt.date).isin(holidays + timedelta(days=1)) | pd.to_datetime(train.pickup_datetime.dt.date).isin(holidays - timedelta(days=1)))
	df['pickup_near_holiday'] = df.pickup_near_holiday.map(lambda x: 1 if x == True else 0)
	df['pickup_businessday'] = pd.to_datetime(df.pickup_datetime.dt.date).isin(business_days)
	df['pickup_businessday'] = df.pickup_businessday.map(lambda x: 1 if x == True else 0)
	return df


def bagged_set_cv(X_ts,y_cs, seed, estimators, xt,yt=None):   
	baggedpred=np.array([ 0.0 for d in range(0, xt.shape[0])]) 
	for n in range (0, estimators):
		params = {    'objective': 'huber',
			'metric': 'rmse',
			'boosting': 'gbdt',
			'fair_c':1.5,
			'learning_rate': 0.2, 
			'verbose': 0,    
			'num_leaves': 60,
			'bagging_fraction': 0.95,    
			'bagging_freq': 1,    
			'bagging_seed': seed + n,    
			'feature_fraction': 0.6,    
			'feature_fraction_seed': seed + n,    
			'min_data_in_leaf': 10,
			'max_bin': 255, 
			'max_depth':10,    
			'reg_lambda': 20,    
			'reg_alpha':20,    
			'lambda_l2': 20,
			'num_threads':30
			}

		d_train = lgb.Dataset(X_ts,np.log1p(y_cs), free_raw_data=False)
		if type(yt)!=type(None):           
		   d_cv = lgb.Dataset(xt,np.log1p(yt), free_raw_data=False, reference=d_train)
		   model = lgb.train(params,d_train,num_boost_round=4000,
		                     valid_sets=d_cv,
		                     verbose_eval=True )                             
		else :
		   d_cv = lgb.Dataset(xt, free_raw_data=False)  
		   model = lgb.train(params,d_train,num_boost_round=4000)                   
		preds=np.expm1(model.predict(xt))          
		baggedpred+=preds            
		print("completed: " + str(n))
	baggedpred/= estimators     
	return baggedpred

train = pd.read_csv('train.csv') # 1458644 data points
test = pd.read_csv('test.csv')
test_orig=test.copy()
train = datetime_features(train)
test = datetime_features(test)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train_gby_passengerCount = train.groupby('passenger_count').sum()
train_gby_passengerCount.plot.bar()
plt.plot()

train.loc[:,'degree'] = ft_degree(train['pickup_latitude'].values,train['pickup_longitude'].values, train['dropoff_latitude'].values,train['dropoff_longitude'].values)
test.loc[:,'degree'] = ft_degree(test['pickup_latitude'].values,test['pickup_longitude'].values, test['dropoff_latitude'].values,test['dropoff_longitude'].values)


train.loc[:,'haversine_distance'] = haversine_distance(train['pickup_latitude'].values,train['pickup_longitude'].values, train['dropoff_latitude'].values,train['dropoff_longitude'].values)

test.loc[:,'haversine_distance'] = haversine_distance(test['pickup_latitude'].values,test['pickup_longitude'].values, test['dropoff_latitude'].values,test['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] = manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)


train = ft_geo_center(train)
test = ft_geo_center(test)


coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,train[['dropoff_latitude',
	'dropoff_longitude']].values,test[['pickup_latitude',
	'pickup_longitude']].values,
	test[['dropoff_latitude', 'dropoff_longitude']].values))
pca = PCA().fit(coords)

train = ft_pca(train,pca)
test = ft_pca(test,pca)

train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
train = clustering(train,kmeans)
test = clustering(test,kmeans)

fig, ax = plt.subplots(ncols=1, nrows=1)
N = 500000
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,
           c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()



fr1 = pd.read_csv('fastest_routes_train_part_1.csv',usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])#,'step_direction'
fr2 = pd.read_csv('fastest_routes_train_part_2.csv',usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps',])#'step_direction'
test_street_info = pd.read_csv('fastest_routes_test.csv',usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])#,'step_direction'
train_street_info = pd.concat((fr1, fr2))


train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')

"""
### Holidays
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays()
# Load business days
us_bd = CustomBusinessDay(calendar = USFederalHolidayCalendar())
# Set business_days equal to the work days in our date range.
business_days = pd.DatetimeIndex(start = train.pickup_datetime.min(),end = train.pickup_datetime.max(),freq = us_bd)
business_days = pd.to_datetime(business_days).date

train = ft_holidays(train)
test = ft_holidays(test)
"""

train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)  
feature_names = list(train.columns)
print(np.setdiff1d(train.columns, test.columns))

do_not_use_for_training = ['id', 'log_trip_duration', 'trip_duration', 'dropoff_datetime', 'pickup_date', 
                               'pickup_datetime']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print('We have %i features.' % len(feature_names))
train[feature_names].count()

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)
test['store_and_fwd_flag'] = test['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)

y =np.array(train['trip_duration'].values)
X = train[feature_names].values
test_id=np.array(test['id'].values)
X_test=test[feature_names].values  


print (" final shape of train " ,  X.shape)
print (" final shape of X_test " ,  X_test.shape)
print (" final shape of y " ,  y.shape)


seed = 1
path = '' 
outset="dm"
estimators=1

predictions = bagged_set_cv(X, y, seed, estimators, X_test, yt=None)
predictions=np.array(predictions)              
print("---- Score on Test data -------")
predictions = (predictions.reshape(-1,1))
print(predictions[:20],predictions.shape)
predictions_idx = test_orig['id'].to_numpy().reshape(-1,1)
best_predictions = np.hstack([predictions_idx,predictions])
best_predictions = pd.DataFrame(best_predictions,columns=['id','trip_duration'])
best_predictions =best_predictions.astype({'trip_duration':int})

best_predictions.to_csv('clean_result_huber.csv',index=False)

