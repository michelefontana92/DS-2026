import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,TargetEncoder
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
"""
def fit_scalers(df_orig, categorical_cols, numerical_cols, label_encoder_cols, target_col):
    scalers = {}

    # TargetEncoder + StandardScaler per categoriche
    for cat in categorical_cols:
        te = TargetEncoder()
        te.fit(df_orig[[cat]], df_orig[target_col])
        se = StandardScaler()
        # scalo il risultato del target encoder
        encoded = te.transform(df_orig[[cat]])
        se.fit(encoded)
        scalers[cat] = {"te": te, "sc": se}

    # StandardScaler numeriche
    for num in numerical_cols:
        scalers[num] = StandardScaler().fit(df_orig[[num]])

    # Label encoder
    for label in label_encoder_cols:
        scalers[label] = LabelEncoder().fit(df_orig[label])

    return scalers

def encode_categorical(df, cat, enc):
    te, sc = enc["te"], enc["sc"]
    encoded = te.transform(df[[cat]])
    scaled = sc.transform(encoded)
    #df = df.drop(columns=[cat])
    #print('Categorical encoding for', cat, 'with columns:', df.columns.to_list())
    df[cat] = scaled
    return df
"""


"""
def fit_scalers(df_orig, categorical_cols,
                numerical_cols, label_encoder_cols,target_col=None):
    scalers = {}
    for cat in categorical_cols:
        scalers[cat] = OneHotEncoder(sparse_output=False).fit(df_orig[[cat]])
    for cat in numerical_cols:
        scalers[cat] = StandardScaler().fit(df_orig[[cat]])
    for label in label_encoder_cols:
        scalers[label] = LabelEncoder().fit(df_orig[[label]])
    return scalers


def encode_categorical(df_orig, attribute, ohe):
    df = df_orig.copy()
    cols = df.columns.to_list()
    found_idx = -1
    for i, c in enumerate(cols):
        if c == attribute:
            found_idx = i
            break
    cols_before = cols[:found_idx]
    cols_after = cols[found_idx+1:]
    df_scaled = df.copy()
    p = ohe.transform(df_scaled[[attribute]])
    features = ohe.get_feature_names_out()
    cols_scaled = [c for c in cols_before]
    cols_scaled = cols_scaled + list(features)
    cols_scaled = cols_scaled + cols_after
    ohe_df = pd.DataFrame(p, columns=features, dtype=int)

    df_scaled = pd.concat([df_scaled.drop(attribute, axis=1), ohe_df], axis=1)[
        cols_scaled]

    return df_scaled
"""

from category_encoders import TargetEncoder

def fit_scalers(df_orig, categorical_cols, numerical_cols, label_encoder_cols, target_col,
                n_splits=5, random_state=0, smoothing=20.0):
    """
    TargetEncoder OOF + StandardScaler per categoriche, SEMPRE 1 colonna per feature (anche multiclass).
    StandardScaler per numeriche.
    LabelEncoder per label_encoder_cols.
    """
    scalers = {}

    # y deve essere numerico (0..K-1) per multiclass; se non lo è, lo mappiamo
    y_raw = df_orig[target_col].to_numpy()
    if not np.issubdtype(y_raw.dtype, np.number):
        classes, y = np.unique(y_raw, return_inverse=True)
        scalers["__y_map__"] = {c: i for i, c in enumerate(classes.tolist())}
    else:
        y = y_raw.astype(int)

    # Stratified CV per classificazione (binaria o multiclass)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # --- categoriche: OOF TE (1 feature) + scaler ---
    for cat in categorical_cols:
        oof = np.zeros((len(df_orig), 1), dtype=np.float64)

        for tr_idx, te_idx in cv.split(df_orig, y):
            te = TargetEncoder(cols=[cat], smoothing=smoothing)
            te.fit(df_orig.iloc[tr_idx][[cat]], y[tr_idx])
            oof[te_idx, 0] = te.transform(df_orig.iloc[te_idx][[cat]]).to_numpy().ravel()

        sc = StandardScaler().fit(oof)

        te_full = TargetEncoder(cols=[cat], smoothing=smoothing)
        te_full.fit(df_orig[[cat]], y)

        scalers[cat] = {"te": te_full, "sc": sc, "type": "single"}

    # --- numeriche: scaler ---
    for num in numerical_cols:
        scalers[num] = StandardScaler().fit(df_orig[[num]])

    # --- label encoders ---
    for col in label_encoder_cols:
        le = LabelEncoder()
        le.fit(df_orig[col].astype(str))
        scalers[col] = le

    return scalers


def encode_categorical(df, cat, enc):
    """
    Stessa firma di prima: prende df, nome colonna cat, e dizionario enc.
    Sostituisce df[cat] con il target-encoding scalato (1 colonna).
    """
    df = df.copy()
    te, sc = enc["te"], enc["sc"]
    encoded = te.transform(df[[cat]]).to_numpy()          # [N,1]
    scaled = sc.transform(encoded).ravel()                # [N]
    df[cat] = scaled
    return df

def encode_numerical(df_orig, attribute, sc):
    df = df_orig.copy()
    df[attribute] = sc.transform(df[[attribute]])
    return df


def encode_label(df_orig, attribute, lb):
    df = df_orig.copy()
    df[attribute] = lb.transform(df[[attribute]])
    return df


def encode_dataset(df, cat_cols, num_cols, label_cols, scalers):
    df_enc = df.copy()
    for cat in cat_cols:
        df_enc = encode_categorical(df_enc, cat, scalers[cat])
    for num in num_cols:
        df_enc = encode_numerical(df_enc, num, scalers[num])
    for label in label_cols:
        df_enc = encode_label(df_enc, label, scalers[label])
    return df_enc


def decode_categorical(df_orig, attribute, ohe):
    df = df_orig.copy()
    cols = df.columns.to_list()
    found = False
    cols_before = []
    cols_after = []
    for _, c in enumerate(cols):
        if c.startswith(attribute):
            found = True
        else:
            if not found:
                cols_before.append(c)
            else:
                cols_after.append(c)
    target_cols = [c for c in df.columns if c.startswith(attribute)]
    decoding = pd.DataFrame(ohe.inverse_transform(
        df[target_cols]), columns=[attribute])
    cols = cols_before + [attribute]+cols_after
    df_decoded = pd.concat(
        [df.drop(target_cols, axis=1), decoding], axis=1)[cols]
    return df_decoded


def decode_numerical(df_orig, attribute, sc):
    df = df_orig.copy()
    df[attribute] = sc.inverse_transform(df[[attribute]])
    return df


def decode_label(df_orig, attribute, lb):
    df = df_orig.copy()
    df[attribute] = lb.inverse_transform(df[[attribute]])
    return df


def decode_dataset(df, cat_cols, num_cols, label_cols, scalers):
    df_dec = df.copy()
    for cat in cat_cols:
        df_dec = decode_categorical(df_dec, cat, scalers[cat])
    for num in num_cols:
        df_dec = decode_numerical(df_dec, num, scalers[num])
    for label in label_cols:
        df_enc = decode_label(df_dec, label, scalers[label])
    return df_enc
