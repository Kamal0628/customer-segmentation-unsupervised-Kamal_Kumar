def create_rfm(df):
    snapshot_date = df['InvoiceDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    })

    rfm.columns = ['Recency', 'Frequency', 'TotalQuantity', 'AvgPrice']

    # Monetary
    rfm['Monetary'] = rfm['TotalQuantity'] * rfm['AvgPrice']

    return rfm.reset_index()