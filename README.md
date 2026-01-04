# Wholesale Customer Clustering (KMeans & DBSCAN)

Clustering analysis of wholesale customer spending data using **KMeans** and **DBSCAN**.
Pipeline includes outlier removal, standardization, elbow method, PCA visualization, and clustering metrics. :contentReference[oaicite:6]{index=6}

## What’s inside
- Outlier removal using IQR :contentReference[oaicite:7]{index=7}
- Feature standardization :contentReference[oaicite:8]{index=8}
- KMeans clustering + elbow method :contentReference[oaicite:9]{index=9}
- DBSCAN clustering :contentReference[oaicite:10]{index=10}
- PCA visualization for clusters :contentReference[oaicite:11]{index=11}
- Clustering evaluation metrics (recommended: Silhouette / Davies–Bouldin / Calinski–Harabasz)

> Repo currently contains a main script and the dataset file referenced in the original README. :contentReference[oaicite:12]{index=12}

---

## Project Structure (recommended)
```text
.
├── data/
│   └── wholesale_customers.csv
├── notebooks/
│   └── 01_wholesale_clustering.ipynb
├── scripts/
│   └── cluster.py
├── assets/
│   └── (plots / screenshots)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

## ▶️ How to Run

1. Clone the repository 
   git clone https://github.com/Amr-Belal-77/Wholesale_clustering.git
   cd Wholesale_clustering

2. Install.
   pip install -r requirements.txt

3. Run the script
   python scripts/cluster.py



