def de_duplication(df):
    return df.drop_duplicates()

def noise_remover(df):
    # removing special characters, lower Case
    df['interaction_content'] = df['interaction_content'].str.replace(r'[^\w\s]', '', regex=True).str.lower()
    return df

def get_input_data():
    import pandas as pd
    df1 = pd.read_csv("data/AppGallery.csv")
    df2 = pd.read_csv("data/Purchasing.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    return df