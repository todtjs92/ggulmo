from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


# train ,test 아에 따로 만들어주구나 . user , test 따로 되도록. 이건 정리를 해야겠따 .

def load_csv(csvfile):
    df = pd.read_csv(csvfile)
    return df

def load_pickle(pickle_file):
    df = pd.read_pickle(pickle_file)
    return df



def train_valid_test_split(df):
    '''
    :param df:
    :return:
    df_train , df_valid , df_test
    '''
    df_train , df_temp = train_test_split(df,test_size =0.2 , random_state=1,shuffle=True, stratify= df['label'] )
    df_valid , df_test = train_test_split(df_temp , test_size=0.5, random_state = 1 , shuffle = True , stratify = df_temp['label'] )

    return df_train , df_valid ,  df_test


def split_feature_label(df):
    '''
    split feature and label X, Y
    :param df:
    :return:
    df_feat , df_label
    '''

    df_feat = df.iloc[:, :-1]
    df_feat = df_feat.values



    df_label = df.iloc[:,-1]
    df_label = df_label.values

    return df_feat , df_label


