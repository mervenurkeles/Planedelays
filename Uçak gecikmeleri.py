
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_01.csv')


print(df.head())




df.memory_usage().sum()
print(df.describe())

# Çizim yap
"""plt.figure(figsize=(12,6))
plt.plot(df['WEATHER_DELAY'])
plt.ylabel("Hava Durumu Gecikmesi")
plt.show()"""

# Eksik değerleri kontrol et
print(df.isna().sum())


passengers = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/T3_AIR_CARRIER_SUMMARY_AIRPORT_ACTIVITY_2019.csv')
print(passengers)

# uçak kuyruk numarasına göre uçakların üretim yılı ve yolcu kapasitesi
aircraft = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/B43_AIRCRAFT_INVENTORY.csv', encoding='latin1')
aircraft.drop_duplicates(subset='TAIL_NUM', inplace=True)
print(aircraft)


# inplace=True parametresi, 'ORIGIN_AIRPORT_ID' sütunundaki yinelenen havaalanı kimliklerini içeren satırları kaldırarak DataFrame'i günceller.
koordinat = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/AIRPORT_COORDINATES.csv', encoding='latin1')
koordinat.drop_duplicates(subset='ORIGIN_AIRPORT_ID', inplace=True)
print(koordinat)

# havayolu kodlarının ana Zamanında Raporlarla eşleştirilmesi için bir arama tablosu elde etmektir.
names = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/CARRIER_DECODE.csv', encoding='latin1')
names.drop_duplicates(inplace=True)
names.drop_duplicates(subset=['OP_UNIQUE_CARRIER'], inplace=True)
print(names)

# yolcu başına çalışan sayısını belirleyebilmemizi sağlar
calisanlar = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/P10_EMPLOYEES.csv')
calisanlar = calisanlar[['OP_UNIQUE_CARRIER', 'PASS_GEN_SVC_ADMIN', 'PASSENGER_HANDLING']]
calisanlar = calisanlar.groupby('OP_UNIQUE_CARRIER').sum().reset_index()
print(calisanlar)

# 2019'da havalimanı şehirlerinin ilk %90'ı için hava durumu raporu
weather_report = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/airport_weather_2019.csv')
pd.set_option('display.max_columns', None)
print(weather_report)


# Ana df'mizle bağlantı kurabilmemiz için havalimanı görünen adı da dahil olmak üzere şehirler ve havaalanları listemiz
sehirler = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/airports_list.csv')
print(sehirler)

# Hava durumu ve şehirler arasıdna bağlantı kur
weather_merge = pd.merge(sehirler, weather_report, how='left', on='NAME')
print(weather_merge)

# Önememlileri al (date, precipitation, snow, temp, wind)
weather = weather_merge[['DATE', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND', 'ORIGIN_AIRPORT_ID']]
print(weather_merge)

# Hava bilgisi olmayanları düşür
weather.drop(weather.loc[weather['ORIGIN_AIRPORT_ID'].isna()].index, axis=0, inplace=True)
print(weather.drop)

weather.loc[weather['TMAX'].isna()]

print(weather.loc[weather['TMAX'].isna()])

weather['TMAX'].fillna(round(weather.groupby('ORIGIN_AIRPORT_ID')['TMAX'].transform('mean'), 1), inplace=True)
weather['AWND'].fillna(round(weather.groupby('ORIGIN_AIRPORT_ID')['AWND'].transform('mean'), 1), inplace=True)
weather.fillna(0, inplace=True)

print(weather.isna().sum())

weather['DATE'] = pd.to_datetime(weather['DATE'])
weather['MONTH'] = pd.DatetimeIndex(weather['DATE']).month
weather['DAY_OF_MONTH'] = pd.DatetimeIndex(weather['DATE']).day
print(weather)


def month_cleanup(monthly_data, aircraft, koordinat, names, weather, passengers, calisanlar):
    
        
    # Ne kadar zamanda düzenleyeceğini görmek için
    start = time.time()
    
    # Veiler Temizlenir
    print("Dropping NaNs from Dep Time, Tail Num. Dropping Cancellations.")
    monthly_data.drop(monthly_data.loc[monthly_data['DEP_TIME'].isna()].index, axis=0, inplace=True)
    monthly_data.drop(monthly_data.loc[monthly_data['TAIL_NUM'].isna()].index, axis=0, inplace=True)
    monthly_data.drop(monthly_data.loc[monthly_data['CANCELLED']==1].index, axis=0, inplace=True)
 
    # FEATURE ENGINEERING - SEGMENT NUMBER
   
    print("Adding Flight Number Sequence - SEGMENT_NUMBER")
    monthly_data["SEGMENT_NUMBER"] = monthly_data.groupby(["TAIL_NUM", 'DAY_OF_MONTH'])["DEP_TIME"].rank("dense", ascending=True)
    
    # FEATURE ENGINEERING - CONCURRENT FLIGHTS
     
    print("Adding Concurrent Flights - CONCURRENT_FLIGHTS")
    monthly_data['CONCURRENT_FLIGHTS'] = monthly_data.groupby(['ORIGIN_AIRPORT_ID','DAY_OF_MONTH', 'DEP_TIME_BLK'])['OP_UNIQUE_CARRIER'].transform("count")
 
    # MERGING to get NUMBER_OF_SEATS
    print("Applying seat counts to flights - NUMBER_OF_SEATS")   
    # Merge aircraft info with main frame on tail number - get NUMBER_OF_SEATS 
    monthly_data = pd.merge(monthly_data, aircraft, how="left", on='TAIL_NUM')
    # Fill missing aircraft info with means
    monthly_data['NUMBER_OF_SEATS'].fillna((monthly_data['NUMBER_OF_SEATS'].mean()), inplace=True)
    # simplify data type of number of seats to reduce memory usage
    monthly_data['NUMBER_OF_SEATS'] = monthly_data['NUMBER_OF_SEATS'].astype('int16')

    # MERGING
    # Merge to get proper carrier name
    print("Applying Carrier Names - CARRIER_NAME")  
    monthly_data = pd.merge(monthly_data, names, how='left', on=['OP_UNIQUE_CARRIER'])
    
    # FEATURE ENGINEERING - AIRPORT_FLIGHTS_MONTH, AIRLINE_FLIGHTS_MONTH, AIRLINE_AIRPORT_FLIGHTS_MONTH
    # Add monthly flight statistics for carrier and airport
    print("Adding flight statistics for carrier and airport - AIRPORT_FLIGHTS_MONTH, AIRLINE_FLIGHTS_MONTH, AIRLINE_AIRPORT_FLIGHTS_MONTH")
    monthly_data['AIRPORT_FLIGHTS_MONTH'] = monthly_data.groupby(['ORIGIN_AIRPORT_ID'])['ORIGIN_CITY_NAME'].transform('count')
    monthly_data['AIRLINE_FLIGHTS_MONTH'] = monthly_data.groupby(['OP_UNIQUE_CARRIER'])['ORIGIN_CITY_NAME'].transform('count')
    monthly_data['AIRLINE_AIRPORT_FLIGHTS_MONTH'] = monthly_data.groupby(['OP_UNIQUE_CARRIER', 'ORIGIN_AIRPORT_ID'])['ORIGIN_CITY_NAME'].transform('count')
    
    # FEATURE ENGINEERING - AVG_MONTHLY_PASS_AIRPORT, AVG_MONTHLY_PASS_AIRLINE
    #Add monthly passenger statistics for carrier and airport
    print("Adding passenger statistics for carrier and airport - AVG_MONTHLY_PASS_AIRPORT, AVG_MONTHLY_PASS_AIRLINE")
    monthly_airport_passengers = pd.DataFrame(passengers.groupby(['ORIGIN_AIRPORT_ID'])['REV_PAX_ENP_110'].sum())
    monthly_data = pd.merge(monthly_data, monthly_airport_passengers, how='left', on=['ORIGIN_AIRPORT_ID'])
    monthly_data['AVG_MONTHLY_PASS_AIRPORT'] = (monthly_data['REV_PAX_ENP_110']/12).astype('int64')
    monthly_airline_passengers = pd.DataFrame(passengers.groupby(['OP_UNIQUE_CARRIER'])['REV_PAX_ENP_110'].sum())
    monthly_data = pd.merge(monthly_data, monthly_airline_passengers, how='left', on=['OP_UNIQUE_CARRIER'])
    monthly_data['AVG_MONTHLY_PASS_AIRLINE'] = (monthly_data['REV_PAX_ENP_110_y']/12).astype('int64')
    
    # MERGING
    # Add employee stats then FEATURE ENGINEER FLT_ATTENDANTS_PER_PASS, GROUND_SERV_PER_PASS
    print("Adding employee statistics for carrier - FLT_ATTENDANTS_PER_PASS, GROUND_SERV_PER_PASS")
    monthly_data = pd.merge(monthly_data, calisanlar, how='left', on=['OP_UNIQUE_CARRIER'])
    monthly_data['FLT_ATTENDANTS_PER_PASS'] = monthly_data['PASSENGER_HANDLING']/monthly_data['REV_PAX_ENP_110_y']
    monthly_data['GROUND_SERV_PER_PASS'] = monthly_data['PASS_GEN_SVC_ADMIN']/monthly_data['REV_PAX_ENP_110_y']
    
    # FEATURE ENGINEERING - PLANE AGE
    # Calculate age of plane
    print("Calculate Fleet Age - PLANE_AGE")
    monthly_data['MANUFACTURE_YEAR'].fillna((monthly_data['MANUFACTURE_YEAR'].mean()), inplace=True)
    monthly_data['PLANE_AGE'] = 2019 - monthly_data['MANUFACTURE_YEAR']

    # MERGING
    # Merge to get airport coordinates
    print("Adding airport coordinates - LATITUDE, LONGITUDE, DEPARTING_AIRPORT")
    monthly_data = pd.merge(monthly_data, koordinat, how='left', on=['ORIGIN_AIRPORT_ID'])
    monthly_data['LATITUDE'] = round(monthly_data['LATITUDE'], 3)
    monthly_data['LONGITUDE'] = round(monthly_data['LONGITUDE'], 3)

    # FEATURE ENGINEERING - PREVIOUS AIRPORT
    # Get previous airport for tail number
    print("Adding airports - PREVIOUS_AIRPORT")
    segment_temp = monthly_data[['DAY_OF_MONTH', 'TAIL_NUM', 'DISPLAY_AIRPORT_NAME', 'SEGMENT_NUMBER']]
    monthly_data = pd.merge_asof(monthly_data.sort_values('SEGMENT_NUMBER'), segment_temp.sort_values('SEGMENT_NUMBER'), on='SEGMENT_NUMBER', by=['DAY_OF_MONTH', 'TAIL_NUM'], allow_exact_matches=False)
    monthly_data['DISPLAY_AIRPORT_NAME_y'].fillna('NONE', inplace=True)
    monthly_data.rename(columns={"DISPLAY_AIRPORT_NAME_y": "PREVIOUS_AIRPORT", "DISPLAY_AIRPORT_NAME_x": "DEPARTING_AIRPORT"}, inplace=True)  
    
    # CLEANING  
    # Drop airports below the 10th percentile
    print("Dropping bottom 10% of airports")
    monthly_data.drop(monthly_data.loc[monthly_data['AIRPORT_FLIGHTS_MONTH'] < 1100].index, axis=0, inplace=True)
    
    # MERGING
    # Merge weather data
    print("Adding daily weather data - PRCP, SNOW, SNWD, SMAX, TMIN, AWND")
    monthly_data = pd.merge(monthly_data, weather, how='inner', on=['ORIGIN_AIRPORT_ID', 'MONTH', 'DAY_OF_MONTH'])

    
    # CLEANING
    # drop columns that we won't use
    print("Clean up unneeded columns")
    monthly_data.drop(columns = ['ORIGIN',  'DEST',  
                   'CRS_DEP_TIME', 'DEP_DELAY_NEW', 'CRS_ARR_TIME', 'ARR_TIME', 
                   'CANCELLED', 'CANCELLATION_CODE', 'CRS_ELAPSED_TIME', 'DISTANCE',
                   'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',
                  'ARR_DELAY_NEW', 'Unnamed: 32',  'ARR_TIME_BLK', 'ACTUAL_ELAPSED_TIME',
                  'DEST_AIRPORT_ID', 'DEST_CITY_NAME',  'OP_CARRIER_FL_NUM',  'OP_UNIQUE_CARRIER',
                       'AIRLINE_ID', 'DATE', 'DAY_OF_MONTH', 'TAIL_NUM','DEP_TIME',
                    'ORIGIN_AIRPORT_ID', 'ORIGIN_CITY_NAME',  'PASSENGER_HANDLING', 'REV_PAX_ENP_110_x', 'REV_PAX_ENP_110_y', 
                                 'PASS_GEN_SVC_ADMIN', 'MANUFACTURE_YEAR',
                                 ],
                    axis=1, inplace=True)
    
    # CLEANING
    # specify data types of various fields to reduce memory usage
    print("Cleaning up data types")
    monthly_data['MONTH'] = monthly_data['MONTH'].astype('object')
    monthly_data['DAY_OF_WEEK'] = monthly_data['DAY_OF_WEEK'].astype('object')
    monthly_data['DEP_DEL15'] = monthly_data['DEP_DEL15'].astype('int8')
    monthly_data['DISTANCE_GROUP'] = monthly_data['DISTANCE_GROUP'].astype('int8')
    monthly_data['SEGMENT_NUMBER'] = monthly_data['SEGMENT_NUMBER'].astype('int8')
    monthly_data['AIRPORT_FLIGHTS_MONTH'] = monthly_data['AIRPORT_FLIGHTS_MONTH'].astype('int64')
    monthly_data['AIRLINE_FLIGHTS_MONTH'] = monthly_data['AIRLINE_FLIGHTS_MONTH'].astype('int64')
    monthly_data['AIRLINE_AIRPORT_FLIGHTS_MONTH'] = monthly_data['AIRLINE_AIRPORT_FLIGHTS_MONTH'].astype('int64')
    monthly_data['PLANE_AGE'] = monthly_data['PLANE_AGE'].astype('int32')
    
    # reset index
    monthly_data.reset_index(inplace=True, drop=True)
    
    # print elapsed time
    print(f'Elapsed Time: {time.time() - start}')
    
    print("FINISHED")
    
    # return cleaned file
    return monthly_data

df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_01.csv')
month01 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_02.csv')
month02 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_03.csv')
month03 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_04.csv')
month04 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_05.csv')
month05 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_06.csv')
month06 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_07.csv')
month07 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_08.csv')
month08 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_09.csv')
month09 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_10.csv')
month10 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_11.csv')
month11 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)
df = pd.read_csv('C:/Users/90538/Desktop/tez/kaggle_veriler/ONTIME_REPORTING_12.csv')
month12 = month_cleanup(df, aircraft, koordinat, names, weather, passengers, calisanlar)

# COMBINE MASTER FILE
all_data = pd.concat([month01, month02, month03, month04, month05, month06, month07, month08, month09, month10, month11, month12]).reset_index(drop=True)
all_data.to_pickle("data/pkl/train_val.pkl")
all_data.to_csv('data/train_val.csv', index=False)


# Datalar bölünür
train, test = train_test_split(all_data, test_size=.3, random_state=42, stratify=all_data['DEP_DEL15'])

# Test et
print(train.head())
print(test.head())


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Features ve target belirleyin
features = [ 'MONTH', 'LATITUDE','DAY_OF_WEEK', 'LONGITUDE','PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND']
target = 'DEP_DEL15'



# Train seti
X_train = train[features]
y_train = train[target]

# Test seti
X_test = test[features]
y_test = test[target]


# Karar ağacı modelini oluşturun
model = DecisionTreeClassifier(random_state=42)

# Modeli eğitin
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = model.predict(X_test)



# Modelin performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.countplot(x='DEP_DEL15', data=train)
plt.title('DEP_DEL15 Dağılımı in Train Data')
plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Modelinizi eğittikten sonra
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# Karar ağacını görselleştir
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Not Delayed', 'Delayed'], rounded=True, fontsize=10)
plt.show()


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Modelinizi eğittikten sonra
model = DecisionTreeClassifier(random_state=42, max_depth=3)  # Max derinlik 3 olarak ayarlandı, bu değeri ihtiyaca göre değiştirebilirsiniz
model.fit(X_train, y_train)


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# İlk 20 veri noktasını kullanarak modeli eğit
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train.head(2000), y_train.head(2000))

# Karar ağacını görselleştir
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Not Delayed', 'Delayed'],
          rounded=True, fontsize=10, max_depth=3)
plt.show()



from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Uygun bir aralıkta max_depth değerlerini belirleyin
max_depth_values = range(1, 15)

# Boş listeler oluşturun
train_scores = []
test_scores = []

# Her bir max_depth değeri için modeli eğit ve skorları kaydet
for depth in max_depth_values:
    dt_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    # Eğitim seti için doğruluk skorunu kaydet
    train_scores.append(dt_classifier.score(X_train, y_train))
    
    # Test seti için doğruluk skorunu kaydet
    test_scores.append(dt_classifier.score(X_test, y_test))

# Skorları grafiğe çizin
plt.figure(figsize=(20, 5))
plt.plot(max_depth_values, train_scores, label="Train")
plt.plot(max_depth_values, test_scores, label="Test")
plt.xlabel('Max Depth Değeri')
plt.ylabel('Accuracy Score')
plt.legend()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veriyi birleştirin
all_data = pd.concat([month01, month02, month03, month04, month05, month06, month07, month08, month09, month10, month11, month12]).reset_index(drop=True)
all_data.to_pickle("data/pkl/train_val.pkl")
all_data.to_csv('data/train_val.csv', index=False)

# Veriyi eğitim ve test setlerine bölün
train, test = train_test_split(all_data, test_size=.3, random_state=42, stratify=all_data['DEP_DEL15'])

# Features ve target'ı belirleyin
features = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND']
target = 'DEP_DEL15'

# Eğitim seti
X_train = train[features]
y_train = train[target]

# Test seti
X_test = test[features]
y_test = test[target]

# Lojistik regresyon modelini oluşturun
lr_model = LogisticRegression(random_state=42)

# Modeli eğitin
lr_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred_lr = lr_model.predict(X_test)

# Modelin performansını değerlendirin
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
class_report_lr = classification_report(y_test, y_pred_lr)

print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(f'Logistic Regression Confusion Matrix:\n{conf_matrix_lr}')
print(f'Logistic Regression Classification Report:\n{class_report_lr}')


import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Karar Ağacı Tahminleri
y_pred_tree = model.predict(X_test)

# Confusion Matrix oluşturun
cm_tree = confusion_matrix(y_test, y_pred_tree)

# Heatmap ile görselleştirin
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Delayed', 'Delayed'], yticklabels=['Not Delayed', 'Delayed'])
plt.title('Karar Ağacı Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değerler')
plt.show()


# Lojistik Regresyon Tahminleri
y_pred_lr = lr_model.predict(X_test)

# Confusion Matrix oluşturun
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Heatmap ile görselleştirin
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Delayed', 'Delayed'], yticklabels=['Not Delayed', 'Delayed'])
plt.title('Lojistik Regresyon Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değerler')
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Lojistik regresyon modelinin tahmin olasılıklarını alın
y_probs = lr_model.predict_proba(X_test)[:, 1]

# ROC eğrisini oluşturun
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# ROC eğrisini görselleştirin
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

