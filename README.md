# Customer Segmentation - Unsupervised Learning

A comprehensive Python project for customer segmentation using multiple unsupervised machine learning algorithms. This project analyzes customer behavior data and segments customers into distinct groups based on RFM (Recency, Frequency, Monetary) analysis.

## 📋 Overview

Customer segmentation is a critical strategy in customer relationship management (CRM). This project implements and compares four popular unsupervised learning algorithms to identify distinct customer segments from transactional data. The analysis helps businesses understand customer behavior patterns and tailor marketing strategies accordingly.

## ✨ Features

- **Data Preprocessing**: Handles missing values, data validation, and format conversion
- **Feature Engineering**: RFM (Recency, Frequency, Monetary) analysis with mathematical transformations
- **Dimensionality Reduction**: PCA for visualization and analysis
- **Multiple Clustering Algorithms**:
  - **K-Means**: Partitioning-based clustering
  - **Hierarchical Clustering**: Agglomerative hierarchical clustering
  - **DBSCAN**: Density-based spatial clustering
  - **Gaussian Mixture Model (GMM)**: Probabilistic clustering approach
- **Evaluation Metrics**: Silhouette Score and Davies-Bouldin Index
- **Visualization**: 2D cluster visualization using PCA components

## 📊 Dataset

The project uses the **Online Retail Dataset** (`online_retail.csv`) containing:
- **500K+ transactions** from an online retail company
- **4,400+ customers** across multiple countries
- Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

**Data Location**: `data/raw/online_retail.csv`

## 🗂️ Project Structure

```
customer-segmentation-unsupervised/
├── main.py                          # Main entry point - runs full pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/
│   └── raw/
│       └── online_retail.csv        # Raw customer transaction data
└── src/
    ├── __init__.py
    ├── data_preprocessing.py         # Data loading and cleaning functions
    ├── feature_engineering.py        # RFM feature extraction
    ├── evaluation.py                 # Clustering evaluation metrics
    ├── utils.py                      # Utility functions (plotting, etc.)
    └── clustering/
        ├── __init__.py
        ├── kmeans.py                 # K-Means implementation
        ├── hierarchical.py           # Hierarchical clustering implementation
        ├── dbscan.py                 # DBSCAN implementation
        └── gmm.py                    # Gaussian Mixture Model implementation
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-unsupervised.git
   cd customer-segmentation-unsupervised
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and metrics
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scipy** - Scientific computing

## 🚀 Usage

### Running the Full Pipeline

Execute the main script to run all clustering algorithms:

```bash
python main.py
```

This will:
1. Load the online retail dataset
2. Clean and preprocess the data
3. Perform RFM analysis
4. Scale features using StandardScaler
5. Reduce dimensionality with PCA
6. Run K-Means, Hierarchical, DBSCAN, and GMM algorithms
7. Evaluate each model using Silhouette Score and Davies-Bouldin Index
8. Generate cluster visualizations

### Example Output

```
--- KMEANS ---
{'silhouette': 0.42, 'davies_bouldin': 0.85}

--- HIERARCHICAL ---
{'silhouette': 0.39, 'davies_bouldin': 0.92}

--- DBSCAN ---
{'silhouette': 0.28, 'davies_bouldin': 1.15}

--- GMM ---
{'silhouette': 0.41, 'davies_bouldin': 0.88}
```

## 🔍 Key Concepts

### RFM Analysis
- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Number of transactions
- **Monetary (M)**: Total spending value

### Clustering Algorithms

| Algorithm | Type | Best For | Pros | Cons |
|-----------|------|----------|------|------|
| **K-Means** | Partitioning | Medium-large datasets | Fast, scalable | Requires k specification |
| **Hierarchical** | Agglomerative | Understanding structure | Dendrogram visualization | Computationally expensive |
| **DBSCAN** | Density-based | Non-convex clusters | No k specification | Sensitive to parameters |
| **GMM** | Probabilistic | Soft clustering | Probability assignments | Higher computational cost |

### Evaluation Metrics

- **Silhouette Score** (higher is better, range: -1 to 1): Measures cluster cohesion and separation
- **Davies-Bouldin Index** (lower is better): Average similarity ratio of clusters

## 📈 Workflow

```
Raw Data → Clean → RFM Analysis → Scale → PCA → Clustering → Evaluate → Visualize
```

1. **Data Preprocessing**: Remove nulls, negative quantities, convert dates
2. **Feature Engineering**: Calculate RFM metrics per customer
3. **Scaling**: Standardize features for fair algorithm comparison
4. **Dimensionality Reduction**: PCA transforms to 2D for visualization
5. **Clustering**: Apply all four algorithms
6. **Evaluation**: Compare cluster quality metrics
7. **Visualization**: Plot 2D cluster distributions

## 🎯 Results Interpretation

- **High Silhouette Score + Low Davies-Bouldin**: Well-separated, compact clusters
- **Algorithm Comparison**: Compare metrics across algorithms to select best approach
- **Cluster Visualization**: Observe cluster distributions in PCA space
- **Business Insights**: Use clusters for targeted marketing strategies

## 🔧 Customization

### Modify Number of Clusters
Edit `main.py`:
```python
k_labels, _ = run_kmeans(rfm_scaled, k=5)  # Change 4 to desired number
```

### Adjust DBSCAN Parameters
Edit `main.py` or directly call:
```python
run_dbscan(rfm_scaled, eps=0.3, min_samples=10)
```

### Change PCA Components
```python
pca = PCA(n_components=3)  # For 3D visualization
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### To contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

If you have any questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dataset source: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Built with scikit-learn, pandas, and other open-source libraries
