import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    return df

def clean_data(df):
    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove negative quantities
    df = df[df['Quantity'] > 0]

    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    return df