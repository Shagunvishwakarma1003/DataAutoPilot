import os
import pandas as pd 
import numpy as np 

def make_eda_report(df: pd.DataFrame, target: str | None = None, out_dir: str= 'output/eda'):
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'eda_report.txt')

    lines = []
    lines.append('=== AUTO EDA REPORT ===\n')
    lines.append(f'Shape: {df.shape}\n')

    #Dtype summary
    lines.append('\n---Column Types ---\n')
    lines.append(df.dtypes.astype(str).value_counts().to_string())
    lines.append('\n')

    # Missing Values
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100). round(2)
    miss_df = pd.DataFrame({'missing': miss, 'missing_%': miss_pct}).sort_values('missing', ascending=False)

    lines.append('\nMissing Values (Top)---\n')
    lines.append(miss_df[miss_df['missing'] > 0]. head(30).to_string())
    lines.append('\n')

    # Basic numeric stats
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        lines.append('\n--- Numeric Summary ---\n')
        lines.append(num_df.describe().to_string())
        lines.append('\n')
    
    # Correlation
    if not num_df.empty and num_df.shape[1] >=2:
        corr = num_df.corr(numeric_only=True)
        lines.append('\n--- Correlation (Top abs with target if available) ---\n')
        if target is not None and target in corr.columns:
            s = corr[target].drop(target).abs().sort_values(ascending=False).head(15)
        else:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().abs().sort_values(ascending=False).head(15)
            lines.append(pairs.to_string())
        lines.append('\n')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n.join(lines)')

    return report_path