import pandas as pd
import numpy as np
import streamlit as st
import folium
import geopandas
import plotly.express as px
import datetime
from datetime import datetime as datetime1

from folium.plugins import MarkerCluster
from streamlit_folium import folium_static


desired_width=150
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)
pd.set_option( 'display.float_format', lambda x: '%.2f' % x )


st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)

    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile


def set_feature(data):
    # add new features
    data['price_sqft'] = data['price'] / data['sqft_lot']
    # transform date
    data['date'] = pd.to_datetime(data['date'])

    return data




data = pd.read_csv('kc_house_data.csv')

# def data_overview(data, geofile):
# ----- Average Price per Year
data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
print(data.loc[data['bathrooms'] == 0])
st.sidebar.title('Filters')

# filters and selections
f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

min_price = int(data['price'].min())
max_price = int(data['price'].max())
avg_price = int(data['price'].mean())

min_year_built = int(data['yr_built'].min())
max_year_built = int(data['yr_built'].max())

min_date = datetime1.strptime(data['date'].min(), '%Y-%m-%d')
max_date = datetime1.strptime(data['date'].max(), '%Y-%m-%d')

# min_date = min_date1 + datetime.timedelta(days=1)
# max_date = max_date1 + datetime.timedelta(days=1)


st.sidebar.title('Commercial Options')

st.sidebar.subheader('Select Max Price')
f_price = st.sidebar.slider('Price', min_price, max_price, max_price)

st.sidebar.subheader('Select Max Year Built')
f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, max_year_built)

st.sidebar.subheader('Select Max Date')
f_date = st.sidebar.slider('Date', min_date, max_date, max_date)
f_date = pd.to_datetime(f_date).strftime('%Y-%m-%d')

st.sidebar.title('Attribute Options')

on_off_attributes = st.sidebar.checkbox('Check to use the filters below')

f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                  sorted(set(data['bedrooms'].unique())))

f_bathrooms = st.sidebar.selectbox('Max number of bathrooms',
                                   sorted(set(data['bathrooms'].unique())))

f_floors = st.sidebar.selectbox('Max number of floors',
                                sorted(set(data['floors'].unique())))

f_waterview = st.sidebar.checkbox('Only houses with Water View')


st.title('Data Overview')

df = data

if (f_zipcode != []) & (f_attributes != []):
    df_map = data.loc[data['zipcode'].isin(f_zipcode), :]
    #Commercial Options
    df_map = df_map.loc[df_map['yr_built'] <= f_year_built]
    df_map = df_map.loc[df_map['date'] <= f_date]
    df_map = df_map.loc[df_map['price'] <= f_price]
    #Attributes Options
    if on_off_attributes:
        df_map = df_map.loc[df_map['bedrooms'] <= f_bedrooms]
        df_map = df_map.loc[df_map['bathrooms'] <= f_bathrooms]
        df_map = df_map.loc[df_map['floors'] <= f_floors]
        if f_waterview:
            df_map = df_map.loc[df_map['waterfront'] == 1]

    data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    # Commercial Options
    if 'yr_built' in data:
        data = data.loc[data['yr_built'] <= f_year_built]
    if 'date' in data:
        data = data.loc[data['date'] <= f_date]
    if 'price' in data:
        data = data.loc[data['price'] <= f_price]
    # Attributes Options
    if on_off_attributes:
        if 'bedrooms' in data:
            data = data.loc[data['bedrooms'] <= f_bedrooms]
        if 'bathrooms' in data:
            data = data.loc[data['bathrooms'] <= f_bathrooms]
        if 'floors' in data:
            data = data.loc[data['floors'] <= f_floors]
        if 'waterfront' in data:
            if f_waterview:
                data = data.loc[data['waterfront'] == 1]

elif (f_zipcode != []) & (f_attributes == []):
    df_map = data.loc[data['zipcode'].isin(f_zipcode), :]
    # Commercial Options
    df_map = df_map.loc[df_map['yr_built'] <= f_year_built]
    df_map = df_map.loc[df_map['date'] <= f_date]
    df_map = df_map.loc[df_map['price'] <= f_price]
    # Attributes Options
    if on_off_attributes:
        df_map = df_map.loc[df_map['bedrooms'] <= f_bedrooms]
        df_map = df_map.loc[df_map['bathrooms'] <= f_bathrooms]
        df_map = df_map.loc[df_map['floors'] <= f_floors]
        if f_waterview:
            df_map = df_map.loc[df_map['waterfront'] == 1]

    data = data.loc[data['zipcode'].isin(f_zipcode), :]
    # Commercial Options
    if 'yr_built' in data:
        data = data.loc[data['yr_built'] <= f_year_built]
    if 'date' in data:
        data = data.loc[data['date'] <= f_date]
    if 'price' in data:
        data = data.loc[data['price'] <= f_price]
    # Attributes Options
    if on_off_attributes:
        if 'bedrooms' in data:
            data = data.loc[data['bedrooms'] <= f_bedrooms]
        if 'bathrooms' in data:
            data = data.loc[data['bathrooms'] <= f_bathrooms]
        if 'floors' in data:
            data = data.loc[data['floors'] <= f_floors]
        if 'waterfront' in data:
            if f_waterview:
                data = data.loc[data['waterfront'] == 1]

elif (f_zipcode == []) & (f_attributes != []):
    df_map = data.loc[:, :]
    # Commercial Options
    df_map = df_map.loc[df_map['yr_built'] <= f_year_built]
    df_map = df_map.loc[df_map['date'] <= f_date]
    df_map = df_map.loc[df_map['price'] <= f_price]
    # Attributes Options
    if on_off_attributes:
        df_map = df_map.loc[df_map['bedrooms'] <= f_bedrooms]
        df_map = df_map.loc[df_map['bathrooms'] <= f_bathrooms]
        df_map = df_map.loc[df_map['floors'] <= f_floors]
        if f_waterview:
            df_map = df_map.loc[df_map['waterfront'] == 1]

    data = data.loc[:, f_attributes]
    # Commercial Options
    if 'yr_built' in data:
        data = data.loc[data['yr_built'] <= f_year_built]
    if 'date' in data:
        data = data.loc[data['date'] <= f_date]
    if 'price' in data:
        data = data.loc[data['price'] <= f_price]
    # Attributes Options
    if on_off_attributes:
        if 'bedrooms' in data:
            data = data.loc[data['bedrooms'] <= f_bedrooms]
        if 'bathrooms' in data:
            data = data.loc[data['bathrooms'] <= f_bathrooms]
        if 'floors' in data:
            data = data.loc[data['floors'] <= f_floors]
        if 'waterfront' in data:
            if f_waterview:
                data = data.loc[data['waterfront'] == 1]

else:
    data = data.loc[data['yr_built'] <= f_year_built]
    data = data.loc[data['date'] <= f_date]
    data = data.loc[data['price'] <= f_price]
    if on_off_attributes:
        data = data.loc[data['bedrooms'] <= f_bedrooms]
        data = data.loc[data['bathrooms'] <= f_bathrooms]
        data = data.loc[data['floors'] <= f_floors]
        if f_waterview:
            data = data.loc[data['waterfront'] == 1]

    df_map = data



st.dataframe(data)
st.subheader('Results: {}'.format(data['id'].count()))

st.title('Region Overview (Map)')

c1, c2 = st.columns((1, 1))

print(df_map)
print(data)

# check_id = df_map.loc[df_map['id']]
check = df_map.empty
if check == False:
    # Base Map - Folium
    density_map = folium.Map(location=[df_map['lat'].mean(),
                                       df_map['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df_map.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}.  ID: {2} Zipcode: {3} Features:'
                            ' {4} sqft, {5} bedrooms, {6} bathrooms,'
                            ' year built: {7}'.format(row['price'],
                                                      row['date'],
                                                      row['id'],
                                                      row['zipcode'],
                                                      row['sqft_living'],
                                                      row['bedrooms'],
                                                      row['bathrooms'],
                                                      row['yr_built'])).add_to(marker_cluster)


    with c1:
        folium_static(density_map)

else:
    st.subheader('Data not found, change filters to see the map!')



def data_overview2(data):

    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_sqft', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/SQFT']

    st.header('Average Values')
    st.dataframe(df)


def sell_recommendation(data):
    data['date'] = pd.to_datetime(data['date'])
    # group by zipcodes - median
    g_zipcode = data[['price', 'zipcode']].groupby('zipcode').median().reset_index()

    # merge
    test1 = pd.merge(data, g_zipcode, on='zipcode', how='inner')

    # rename columns and organize by ascending zipcodes
    test1.rename(columns = {'price_x':'price', 'price_y':'region_median_price'}, inplace = True)
    test1 = test1.sort_values('zipcode', ascending=True)

    # convert to integer
    test1.price = test1.price.astype(int)
    test1.region_median_price = test1.region_median_price.astype(int)

    # create new column
    test1['status'] = 'x'

    # populate new column
    for i in range(len(test1)):
        if (test1.loc[i, 'price'] <= test1.loc[i, 'region_median_price']) & (test1.loc[i, 'condition'] >= 2):
            test1.loc[i, 'status'] = 'buy'
        else:
            test1.loc[i, 'status'] = 'dont buy'


    # get seasons
    # spring = range(3, 5) = 2
    # summer = range(6, 8) = 3
    # fall = range(9, 11) = 4
    # winter = everything else = 1

    # create and populate season number column
    for i in test1:
        test1['season'] = test1['date'].dt.month%12 // 3 + 1

    # replace season numbers for season str name
    for i in range(len(test1)):
        if test1.loc[i, 'season'] == 2:
            test1.loc[i, 'season'] = 'spring'
        elif test1.loc[i, 'season'] == 3:
            test1.loc[i, 'season'] = 'summer'
        elif test1.loc[i, 'season'] == 4:
            test1.loc[i, 'season'] = 'fall'
        else:
            test1.loc[i, 'season'] = 'winter'


    # group by seasons - median
    # g_season = test1[['price', 'season']].groupby('season').median().reset_index()
    g_season2 = test1[['price', 'season', 'zipcode']].groupby(['zipcode', 'season']).median().reset_index()


    # merge
    test2 = pd.merge(test1, g_season2, on='zipcode', how='inner')

    test2.price_y = test2.price_y.astype(int)

    for i in range(len(test2)):
        if test2.loc[i, 'price_x'] >= test2.loc[i, 'price_y']:
            test2.loc[i, 'sell_rec'] = test2.loc[i, 'price_x'] * 1.1

        elif test2.loc[i, 'price_x'] < test2.loc[i, 'price_y']:
            test2.loc[i, 'sell_rec'] = test2.loc[i, 'price_x'] * 1.3


    test2.rename(columns = {'price_x':'buy_price', 'price_y':'region_season_median_price', 'season_x':'posted_season', 'season_y':'PERIOD'}, inplace = True)

    test2.sell_rec = test2.sell_rec.astype(int)
    st.title('Buy / Sell Recommendation')


    min_price = int(test2['buy_price'].min() + 1)
    max_price = int(test2['buy_price'].max() + 1)
    avg_price = int(test2['buy_price'].mean())

    count = 0

    f_buy = st.checkbox('Only buy status')
    f_price = st.slider('Select Max Buy Price', min_price, max_price, avg_price, key = count)
    f_id = st.multiselect('Search for Specific IDs', test2['id'].unique())
    f_zipcode = st.multiselect('Search for Zipcode', test2['zipcode'].unique())

    if f_buy:
        test2 = test2.loc[test2['status'] == 'buy']

    if f_zipcode != []:
        test2 = test2.loc[test2['zipcode'].isin(f_zipcode), :]

    test2 = test2.loc[test2['buy_price'] < f_price]

    if f_id != []:
        test2 = test2.loc[test2['id'].isin(f_id), :]

    st.dataframe(test2[['id', 'zipcode', 'PERIOD', 'region_season_median_price', 'buy_price', 'sell_rec', 'status']])
    st.subheader('Results: {}'.format(test2['id'].count()))



#
# test3 = test2.loc[test2['status'] == 'buy']
# test4 = test3.loc[test3['sell'] == 'good_sell']
# st.title('TEST2 only buy and good sell')
# st.dataframe(test4[['id', 'zipcode', 'season_y', 'price_y', 'price_x', 'season_x', 'sell', 'status']])
# st.header(test4['id'].count())
# test2['']
# for i in range(len(test1)):
#     if test1.loc[i, 'season'] == 2:
#         test1.loc[i, 'season'] = 'spring'
#     elif test1.loc[i, 'season'] == 3:
#         test1.loc[i, 'season'] = 'summer'
#     elif test1.loc[i, 'season'] == 4:
#         test1.loc[i, 'season'] = 'fall'
#     else:
#         test1.loc[i, 'season'] = 'winter'



# print(test2[test2['id'] == 2817850290])


# c1, c2 = st.columns((1, 1))
#
# c1.header('NEW DF')
# c1.dataframe(test2)
# c2.dataframe(region_and_season)






if __name__ == '__main__':
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    # Transform
    data = set_feature(data)

    # data_overview(data, geofile)
    data_overview2(data)

    sell_recommendation(data)


# data = pd.read_csv('kc_house_data.csv')
# go = data.loc[data['yr_built'] == 1900]
# print(go['yr_built'])