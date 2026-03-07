# unsupervised learning module (clustering + PCA)

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _preprocess_features(df: pd.DataFrame):
    # numeric + categorical handling for clustering
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ],
        remainder='drop'
    )
    return pre

def run_clustering(df: pd.DataFrame, k_min=2, k_max=10, random_state=42):
    # try multiple k, select best based on silhouette score

    pre = _preprocess_features(df)

    # preprocess -> array 
    X = pre.fit_transform(df)

    best_k = None
    best_score = -1
    best_model = None
    best_labels = None
    all_scores = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)

        # If all points in 1 cluster (rare), skip
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)
        all_scores.append({'k': k, 'silhouette': float(score)})

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels

    return {
        'best_k': best_k,
        'best_score': float(best_score),
        'scores': all_scores,
        'best_cluster_model': best_model,
        'labels': best_labels
    }

def run_pca(df: pd.DataFrame, n_components=2, random_state=42):
    # PCAembeddings (useful for visualization or dimensionality reduction)

    pre = _preprocess_features(df)
    X = pre.fit_transform(df)

    pca = PCA(n_components=n_components, random_state=42)
    emb = pca.fit_transform(X)

    return {
        'pca_model': pca,
        'components': emb,
    }
def generate_cluster_report(df, labels, save_path='output/cluster_report.csv'):
    import pandas as pd
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_Copy = df.copy()
    df_Copy['cluster'] = labels

    size_df =(
         df_Copy['cluster']
        .value_counts()
        .sort_index()
        .reset_index()
    )
    size_df.columns = ['cluster', 'count']
    size_df['percentage'] = (size_df['count'] / len(df_Copy) * 100).round(2)

    numerical_cols = df_Copy.select_dtypes(include='number').columns.tolist()
    if 'cluster' in numerical_cols:
        numerical_cols.remove('cluster')

    mean_df = (
        df_Copy
        .groupby('cluster')[numerical_cols]
        .mean()
        .reset_index()
    )

    report = pd.merge(size_df, mean_df, on='cluster')

    report.to_csv(save_path, index=False)

    return report

def interpret_clusters(report_df):
    interpretations = []

    for _, row in report_df.iterrows():
        cluster_id = int(row['cluster'])
        size_pct = row['percentage']

        age = row.get('age', None)
        bmi = row.get('bmi', None)
        charges = row.get('charges', None)

        text = f'Cluster {cluster_id} represents {size_pct}% of data.'

        if age is not None:
            text += f' Average age is {age:.1f}.'
        if bmi is not None:
            text += f' Average BMI is {bmi:.1f}.'
        if charges is not None:
            text += f' Average charges are {charges:.2f}.'

        interpretations.append(text)

    return interpretations

# smart Business Interpretation
def interpret_clusters_smart(df, report_df, top_n=3, save_path='output/cluster_interpretation.txt'):
    '''
    Business-smart interpretation
    - cluster means vs overall means
    - auto tags.: High/Low Age, BMI, Charges (if columns exist)
    - writes a readable text report
    '''
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Overall means only for numeric columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    overall = df[num_cols].mean(numeric_only=True)

    def tag(feature, cluster_mean, overall_mean):
        if overall_mean == 0 or overall_mean is None:
            return None
        diff_pct = ((cluster_mean - overall_mean) / abs(overall_mean)) * 100

        # threshold: +/- 8% => noticeable
        if diff_pct >= 8:
            return f'HIGH {feature} (+{diff_pct:.1f}%)'
        if diff_pct <= -8:
            return f'LOW {feature} ({diff_pct:.1f}%)'
        return None
    
    lines = []
    lines.append('=== Cluster Business Interpretation ===\n')

    # If report_df already has these columns
    for _, row in report_df.iterrows():
        cid = int(row['cluster'])
        pct = float(row.get('percentage', 0))

        lines.append(f'Cluster {cid} | Size: {pct:.2f}%')
        insights = []

        # Check common business features if present in report_df
        for col, name in [('age', 'Age'), ('bmi', 'BMI'), ('charges', 'Charges'), ('children', 'Children')]:
            if col in report_df.columns and col in overall.index:
                cm = float(row[col])
                om = float(overall[col])
                t = tag(name, cm, om)
                if t:
                    insights.append(t)

        # fallback: top deviating numeric columns (optional)
        # find strongest deviations among numeric cols present in report_df
        numeric_in_report = [c for c in report_df.columns if c in overall.index and c not in ['cluster', 'count', 'percentage']]
        devs = []
        for c in numeric_in_report:
            cm = float(row[c])
            om = float(overall[c])
            if om != 0:
                devs.append((c, ((cm -om) / abs(om)) * 100))
        devs = sorted(devs, key=lambda x: abs(x[1]), reverse=True)

        if not insights and devs:
            # take top_n deviations if business features not avaliable

            for c, dp in devs[ : top_n]:
                if dp >= 8:
                    insights.append(f'HIGH {c} (+{dp:.1f}%)')
                elif dp <= -8:
                    insights.append(f'LOW {c} ({dp:.1f}%)')

        if insights:
            lines.append(' Insights:"+" |'.join(insights))
        else:
            lines.append('  Insights: No strong deviations found (clusters are similar).')

        lines.append('') # blank line

    # write file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return lines

# Auto Clustring naming
def auto_name_cluster(report_df):
    '''
    option-1: Auto Cluster Naming
    Logic: charges / bmi / age comparing with this and gives simple business-friendly name
    '''

    names = {}

    for _, row in report_df.iterrows():
        cid = int(row['cluster'])

        age = row.get('age', None)
        bmi = row.get('bmi', None)
        charges = row.get('charges', None)

        # default fallback name
        cname = f'segment {cid}'

        # simple naming rules
        if charges is not None:
            if charges >= 20000:
                cname = "High Premium Segment"
            elif charges >= 12000:
                cname = 'Mild Premium Segment'
            else:
                cname = 'Low Premium Segment'

        # extra boost naming based on BMI + Age
        if bmi is not None and age is not None:
            if bmi >= 30 and age >= 40:
                cname += '(High Risk)'
            elif bmi >= 30:
                cname += '(BMI Risk)'
            elif age >= 45:
                cname += '(Age Risk)'

        names[cid] = cname
    return names

def generate_business_recommendations(cluster_names, report_df, save_path='output/business_recommendation.txt'):
    '''
    option-3: Business Recommendation Generator
    Comparing -> Cluster name + mean values and generate suggestion text
      '''
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = []
    lines.append('=== Business Recommendations (Auto Generated) ===\n')

    for _, row in report_df.iterrows():
        cid = int(row['cluster'])
        cname = cluster_names.get(cid, f'Segment {cid}')
        age = row.get('age', None)
        bmi = row.get('bmi', None)
        charges = row.get('charges', None)

        lines.append(f'Cluster {cid}: {cname}')

        # Smart suggestions
        if charges is not None and charges >= 20000:
            lines.append('- Recommendation: Offer premium plans, upsell high coverage polices.')
            lines.append('- Action: Focus on retention + personalized suport (high value customer).')

        if bmi is not None and bmi >= 30:
            lines.append('- Recommendation: Add wellness programs+ preventive health offers.')
            lines.append('- Action: Risk reduction through fitness/health incentives.')

        if age is not None and age >= 45:
            lines.append('- Recommendation: Senior-friendly policy bundles + long-term coverage options.')
            lines.append('- Action: Target with long-term plans, claim support benefits.')

        # fallback
        if (charges is None and bmi is None and age is None):
            lines.append('- Recommendation: No strong signals; test marketing offers based on other features.')

        lines.append('') # blank line

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return lines