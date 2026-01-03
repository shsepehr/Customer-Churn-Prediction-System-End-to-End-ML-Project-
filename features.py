
def add_features(df):
    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
    return df
