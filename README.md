# Customer Segmentation KMeans Project

## 🎯 Project Overview
End-to-end unsupervised ML project implementing **KMeans clustering** for mall customer segmentation using real-world dataset. Segments 200 customers into 3 actionable groups based on Age, Gender, Annual Income, and Spending Score - perfect for targeted marketing campaigns.[1]

**Business Value**: Identifies high-value customers (high spenders) vs. conservative spenders for personalized strategies. Silhouette Score: **0.34** (good cluster separation).[1]

## 🛠 Technical Implementation

### Data Pipeline
```
Raw Data → Preprocessing → Feature Engineering → Clustering → Evaluation → Deployment
```
- **Preprocessing**: Gender encoding (Male=1, Female=0), dropped CustomerID[1]
- **Elbow Method**: Optimal K=3 determined via SSE plot[1]
- **Features**: `['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']`

### Cluster Profiles[1]
| Cluster | Segment | Size | Key Traits | Marketing Strategy |
|---------|---------|------|------------|-------------------|
| **0** | Low Income Low Spending | ~67 customers | Conservative, low earners | Budget promotions, loyalty programs |
| **1** | **High Income High Spending** | ~81 customers | **VIP Segment** - Big spenders | Premium offers, exclusive deals |
| **2** | Young Low Income High Spending | ~52 customers | Young impulse buyers | Trendy products, flash sales |

## 📊 Key Results & Visualizations

### Cluster Visualization[1]
```
Income vs Spending Score scatter plot clearly separates 3 clusters
- Cluster 1 (Red): High Income + High Spending (Prime Target)
- Cluster 0 (Blue): Low Income + Low Spending  
- Cluster 2 (Green): Low Income + High Spending (Young)
```

**Silhouette Analysis**: Score 0.34 indicates well-separated, cohesive clusters[1]

## 🚀 Production-Ready Features

### Real-time Prediction API[1]
```python
def predict_customer_segment(gender, age, income, spending_score):
    """Predict customer segment for new customers"""
    new_data = np.array([[gender, age, income, spending_score]])
    cluster = kmeans.predict(new_data)[0]
    labels = {
        0: 'Low Income Low Spending',
        1: 'High Income High Spending', 
        2: 'Young Low Income High Spending'
    }
    return labels.get(cluster, 'Unknown Segment')

# Example
result = predict_customer_segment(1, 28, 50, 60)  # Young Male, returns "Young Low Income High Spending"
```

### Model Persistence
```python
import pickle
# Save
pickle.dump(kmeans, open('kmeans.pkl', 'wb'))
# Load 
kmeans = pickle.load(open('kmeans.pkl', 'rb'))
```

## 🧪 Quick Start

# 1. Clone & Install
git clone https://github.com/wittyswayam/Mall-Customer-Segmentation-Using-Kmeans.git
cd Mall-Customer-Segmentation-Using-Kmeans
pip install -r requirements.txt

# 2. Run Analysis
jupyter notebook Customer-Clustering-Kmeans-Mall-Customer-Segmentation-Data.ipynb

# 3. Test Prediction
python predict.py


**requirements.txt**:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
pickle
```

## 🔬 Advanced Extensions (For Portfolio Enhancement)

1. **Hyperparameter Tuning**: GridSearchCV for KMeans parameters
2. **Alternative Algorithms**: DBSCAN, GMM, Hierarchical Clustering comparison
3. **Feature Engineering**: PCA dimensionality reduction, interaction terms
4. **Deployment**: Flask/FastAPI REST API + Streamlit dashboard
5. **Evaluation**: Davies-Bouldin Index, Calinski-Harabasz Score

## 📈 Business Impact Metrics
```
- Identified 81/200 (40.5%) High-Value Customers (Cluster 1)
- Young High Spenders (Cluster 2): 52 customers for trend-based campaigns  
- Conservative segment (Cluster 0): Loyalty program targets
```
