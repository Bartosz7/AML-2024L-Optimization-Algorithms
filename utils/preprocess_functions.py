import pandas as pd
from scipy.io import arff
from sklearn import preprocessing 
from utils.preprocess_helpers import(
    one_hot_encode,
    impute_water,
    vif_train_test_split
)

RANDOM_STATE = 123

def preprocess_booking(filename="data/booking.csv"):
    """Preprocessing for booking.csv dataset."""
    booking = pd.read_csv(filename).drop(["Booking_ID", "date of reservation"], axis=1)
    booking["market segment type"] = 1*(booking["market segment type"] == "Online")
    booking["booking status"] = 1*(booking["booking status"] == "Canceled")
    label_encoder = preprocessing.LabelEncoder() 
    booking["room type"] = label_encoder.fit_transform(booking["room type"]) 
    booking = one_hot_encode(booking)

    return vif_train_test_split(
        df=booking, 
        target_col_name="booking status",
        dataset_name="booking"
    )

def preprocess_churn(filename="data/churn.csv"):
    """Preprocessing for churn.csv dataset."""
    churn = pd.read_csv(filename)
    churn["FrequentFlyer"] = 1*(churn["FrequentFlyer"] == "Yes")
    churn["BookedHotelOrNot"] = 1*(churn["BookedHotelOrNot"] == "Yes")
    churn["AccountSyncedToSocialMedia"] = 1*(churn["AccountSyncedToSocialMedia"] == "Yes")
    churn.loc[churn["AnnualIncomeClass"] == "Low Income", "AnnualIncomeClass"] = 0
    churn.loc[churn["AnnualIncomeClass"] == "Middle Income", "AnnualIncomeClass"] = 1
    churn.loc[churn["AnnualIncomeClass"] == "High Income", "AnnualIncomeClass"] = 2
    churn.AnnualIncomeClass = churn.AnnualIncomeClass.astype(int)
    
    return vif_train_test_split(
        df=churn, 
        target_col_name="Target", 
        dataset_name="churn"
    )

def preprocess_employee(filename='data/employee.csv'):
    """Preprocessing for employee.csv dataset."""
    df = pd.read_csv(filename)
    df['EducationBachelors'] = 1 * (df['Education'] == 'Bachelors')
    df['EducationMasters'] = 1 * (df['Education'] == 'Masters')
    df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    df['EverBenched'] = df['EverBenched'].map({'No': 0, 'Yes': 1})
    df.drop(['Education', 'City'], axis=1, inplace=True)
    
    return vif_train_test_split(
        df=df, 
        target_col_name="LeaveOrNot", 
        dataset_name="employee"
    )

def preprocess_challenger(filename='data/challenger_lol.csv'):
    """Preprocessing for challenger_lol.csv dataset."""
    df = pd.read_csv(filename)
    df.drop('gameId', axis=1, inplace=True)
    for col in ['blue', 'red']:
        for lane in ['BOT_LANE', 'MID_LANE', 'TOP_LANE']:
            df[f'{col}FirstTowerLane_{lane}'] = df[f'{col}FirstTowerLane'].apply(lambda x: int(lane in x))
        for dragon in ['AIR_DRAGON', 'WATER_DRAGON', 'FIRE_DRAGON', 'EARTH_DRAGON']:
            df[f'{col}DragnoType_{dragon}'] = df[f'{col}DragnoType'].apply(lambda x: int(lane in x))
        df.drop(f'{col}FirstTowerLane', axis=1, inplace=True)
        df.drop(f'{col}DragnoType', axis=1, inplace=True)
    
    return vif_train_test_split(
        df=df, 
        target_col_name="blueWins",
        dataset_name="challenger lol"
    )

def preprocess_jungle(filename='data/jungle_chess.arff'):
    """Preprocessing for jungle_chess.arff dataset."""
    df = arff.loadarff(filename)
    df = pd.DataFrame(df[0])
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    df = df[df['class'] != 'd']
    df[['highest_strength', 'closest_to_den', 'fastest_to_den', 'class']] = df.copy()[['highest_strength', 'closest_to_den', 'fastest_to_den', 'class']].applymap(lambda x: int(x == 'w'))
    df = pd.concat([df, pd.get_dummies(df[['white_piece0_advanced', 'black_piece0_advanced']], drop_first=True)], axis=1)
    df.drop(['white_piece0_advanced', 'black_piece0_advanced'], axis=1, inplace=True)
    df = df.apply(pd.to_numeric)
    
    return vif_train_test_split(
        df=df, 
        target_col_name="class",
        dataset_name="jungle"
    )

def preprocess_water(filename="data/water_quality.csv"):
    """Preprocessing for water_quality.csv dataset."""
    water = pd.read_csv(filename)
    
    return vif_train_test_split(
        df=water, 
        target_col_name="is_safe",
        dataset_name="water",
        additional_preprocess=impute_water
    )