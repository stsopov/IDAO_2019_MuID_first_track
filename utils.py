import os
from itertools import repeat
import numpy as np
import pandas as pd

SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']

TRAIN_COLUMNS = ["label", "weight"]

FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T", "FOI_hits_DT", 
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_DZ", "FOI_hits_S"]

ID_COLUMN = "id"

N_STATIONS = 4
FEATURES_PER_STATION = 6
N_FOI_FEATURES = N_STATIONS*FEATURES_PER_STATION
# The value to use for stations with missing hits
# when computing FOI features
EMPTY_FILLER = 1000

# Examples on working with the provided files in different ways

# hdf is all fine - but it requires unpickling the numpy arrays
# which is not guranteed
def load_train_hdf(path):
    return pd.concat([
        pd.read_hdf(os.path.join(path, "train_part_%i_v2.hdf" % i))
        for i in (1, 2)], axis=0, ignore_index=True)


def load_data_csv(path, feature_columns):
    train = pd.concat([
        pd.read_csv(os.path.join(path, "train_part_%i_v2.csv.gz" % i),
                    usecols= [ID_COLUMN] + feature_columns + TRAIN_COLUMNS, nrows = 125000,
                    index_col=ID_COLUMN)
        for i in (1, 2)], axis=0, ignore_index=True)
    test = pd.read_csv(os.path.join(path, "test_public_v2.csv.gz"),
                       usecols=[ID_COLUMN] + feature_columns, index_col=ID_COLUMN)
    return train, test


def parse_array(line, dtype=np.float32):
    return np.fromstring(line[1:-1], sep=" ", dtype=dtype)


def load_full_test_csv(path):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    test = pd.read_csv(os.path.join(path, "test_public_v2.csv.gz"),
                       index_col="id", converters=converters,
                       dtype=types,
                       usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
    return test


def find_closest_hit_per_station(row):
    result = np.empty(N_FOI_FEATURES, dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]
    
    for station in range(4):
        hits = (row["FOI_hits_S"] == station)
        if not hits.any():
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
        else:
            x_distances_2 = (row["Lextra_X[%i]" % station] - row["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - row["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            closest_hit = np.argmin(distances_2)
            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = row["FOI_hits_T"][hits][closest_hit]
            closest_z_per_station[station] = row["FOI_hits_Z"][hits][closest_hit]
            closest_dx_per_station[station] = row["FOI_hits_DX"][hits][closest_hit]
            closest_dy_per_station[station] = row["FOI_hits_DY"][hits][closest_hit]
    return result

def meanTargetEncoder(dataset, feature_for_encoding, label, alpha = 10):
    # new columns for dataset
    new_col_name = feature_for_encoding + '_mean_label'
    mean_general_value = np.mean(dataset[dataset.is_train == 1].label)
    
    # evaluation stat
    stat = dataset[dataset.is_train == 1].groupby(feature_for_encoding)[label].agg(['mean', 'count'], index=False)
    
    # adding to dataset
    stat[new_col_name] = (mean_general_value * alpha +  stat['count'] * stat['mean']) / (stat['count'] + alpha)
    stat = stat.drop(columns = ['mean', 'count'])
    dataset = pd.merge(dataset, stat, on = feature_for_encoding, how = 'left')
    dataset[new_col_name] = dataset[new_col_name].fillna(mean_general_value)
    dataset = dataset.sort_index()
    
    return dataset

def preparing_p_and_matched_hits(dataset):
    if dataset['P_type'] == 4.5:
        if (((dataset['cl_hits_0']) != 1000) and (dataset['cl_hits_1'] != 1000)):
            return 1
        else:
            return 0
    elif dataset['P_type'] == 8.0:
        if (((dataset['cl_hits_0']) != 1000) and (dataset['cl_hits_1'] != 1000) and 
            ((dataset['cl_hits_2'] != 1000) or (dataset['cl_hits_3'] != 1000))):
            return 1
        else:
            return 0
    else:
        if (((dataset['cl_hits_0']) != 1000) and (dataset['cl_hits_1'] != 1000) 
            and (dataset['cl_hits_2'] != 1000) and (dataset['cl_hits_3'] != 1000)):
            return 1
        else:
            return 0
        
def prep_P(P):
    return P.apply(lambda x: 2 if x < 3 else 4.5 if x < 6 else 8 if x < 10 else 11)
       
def find_relative_diff_btwn_closest_extra(row):
    result = np.empty(2, dtype=np.float32)
    N = 0
    metric1 = 0
    for i in [0, 1, 2, 3]:
        if row['cl_hits_' + str(i)] != 1000:
            metric1 += (row['cl_hits_' + str(i)] / row['cl_hits_' + str(i + 16)]**2 + 
                        row['cl_hits_' + str(i + 4)] / row['cl_hits_' + str(i + 20)]**2)
                
            N += 1
        
    result[0] = N
    result[1] = metric1 / N
    
    return result


def find_relative_diff_btwn_matched_extra(row):
    result = np.empty(2, dtype=np.float32)
    N = 0
    metric1 = 0
    for i in [0, 1, 2, 3]:
        if row['MatchedHit_TYPE[' + str(i) + ']'] != 0:
            metric1 += (((row['MatchedHit_X[' + str(i) + ']'] - row['Lextra_X[' + str(i) + ']'])**2) / row['MatchedHit_DX[' + str(i) + ']']**2 + 
                        ((row['MatchedHit_Y[' + str(i) + ']'] - row['Lextra_Y[' + str(i) + ']'])**2) / row['MatchedHit_DY[' + str(i) + ']']**2)
                
            N += 1
        
    result[0] = N
    result[1] = metric1 / N
    
    return result


def find_relative_diff_btwn_closest_extra_another_pad_var(row):
    result = np.empty(2, dtype=np.float32)
    N = 0
    metric1 = 0
    for i in [0, 1, 2, 3]:
        if row['cl_hits_' + str(i)] != 1000:
            metric1 += (row['cl_hits_' + str(i)] / (row['MatchedHit_DX[' + str(i) + ']']**2) + 
                        row['cl_hits_' + str(i + 4)] / (row['MatchedHit_DY[' + str(i) + ']']**2))
                
            N += 1
        
    result[0] = N
    result[1] = metric1 / N
    
    return result



def find_features_foi_per_station(row):
    EMPTY_FILLER = 1000
    
    result = np.empty(8, dtype=np.float32)
    avg_dist = result[0:4]
    n_hits = result[4:8]
    
    for station in range(4):
        hits = (row["FOI_hits_S"] == station)
        
        if not hits.any():
            avg_dist[station] = EMPTY_FILLER
            n_hits[station] = 0
        else:
            x_distances_2 = (row["Lextra_X[%i]" % station] - row["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - row["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            
            N_hits_value = sum(hits)
            
            avg_dist[station] = np.mean(np.sqrt(distances_2))
            n_hits[station] = N_hits_value

    return result