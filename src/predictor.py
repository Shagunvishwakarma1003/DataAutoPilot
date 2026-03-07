
import os 
import pandas as pd

def  predict_to_csv(model, df: pd.DataFrame, target: str |
                    None, save_path: str):
    os.makedirs(os.path.dirname(save_path),
                exist_ok=True)
    
    x = df.copy()
    if target and target in x.columns:
        x = x.drop(columns=[target])

    preds = model.predict(x)
    out = df.copy()
    out['prediction'] = preds
    out.to_csv(save_path, index=False)
    return out