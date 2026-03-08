# src/report_generator.py
import os
import json
import pdfkit
from datetime import datetime
from typing import Any, Dict, Optional


def _read_text(path: str, max_chars: int = 12000) -> str:
    if not path or (not os.path.exists(path)):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return txt[:max_chars] + ("\n\n...[truncated]..." if len(txt) > max_chars else "")


def _safe(s: Any) -> str:
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _img_block(title: str, path: str, output_dir: str) -> str:
    if path and os.path.exists(path):
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        return f"""
        <div class="card">
          <h3>{_safe(title)}</h3>
          <img src="{_safe(rel)}" alt="{_safe(title)}" />
          <p class="muted">{_safe(rel)}</p>
        </div>
        """
    return f"""
    <div class="card">
      <h3>{_safe(title)}</h3>
      <p class="muted">Not found: {_safe(path)}</p>
    </div>
    """


def generate_html_report(
    output_dir: str,
    task_type: str,
    dataset_path: str,
    target: str,
    data_shape: tuple,
    best_model_name: str,
    best_metric: Dict[str, Any],
    all_results: list,
    artifacts: Optional[Dict[str, str]] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Creates output/report.html + output/run_summary.json
    """

    os.makedirs(output_dir, exist_ok=True)
    artifacts = artifacts or {}
    notes = notes or {}

    # Defaults (if not provided)
    eda_txt = artifacts.get(
        "eda_report_txt", os.path.join(output_dir, "eda", "eda_report.txt")
    )
    corr_png = artifacts.get(
        "corr_heatmap_png", os.path.join(output_dir, "eda", "correlation_heatmap.png")
    )
    fi_png = artifacts.get(
        "feature_importance_png", os.path.join(output_dir, "feature_importance.png")
    )
    pi_png = artifacts.get(
        "permutation_importance_png",
        os.path.join(output_dir, "permutation_importance.png"),
    )
    shap_summary_png = artifacts.get(
        "shap_summary_png", os.path.join(output_dir, "shap", "shap_summary.png")
    )
    shap_waterfall_png = artifacts.get(
        "shap_waterfall_png", os.path.join(output_dir, "shap", "shap_waterfall.png")
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eda_text = _read_text(eda_txt)

    meta = {
        "generated_at": now,
        "task_type": task_type,
        "dataset_path": dataset_path,
        "target": target,
        "shape": list(data_shape),
        "best_model": best_model_name,
        "best_metric": best_metric,
        "notes": notes,
        "artifacts": artifacts,
        "all_results": all_results,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Results table
    task = task_type.lower()
    if task == "classification":
        header = "<th>Model</th><th>Acc</th><th>Prec</th><th>Recall</th><th>F1</th><th>CV_F1</th>"
        rows = ""
        for r in all_results:
            rows += (
                "<tr>"
                f"<td>{_safe(r.get('model',''))}</td>"
                f"<td>{_safe(r.get('accuracy',''))}</td>"
                f"<td>{_safe(r.get('precision',''))}</td>"
                f"<td>{_safe(r.get('recall',''))}</td>"
                f"<td>{_safe(r.get('f1_score',''))}</td>"
                f"<td>{_safe(r.get('cv_f1',''))}</td>"
                "</tr>"
            )
    else:
        header = "<th>Model</th><th>MAE</th><th>RMSE</th><th>R2</th><th>CV_RMSE</th>"
        rows = ""
        for r in all_results:
            rows += (
                "<tr>"
                f"<td>{_safe(r.get('model',''))}</td>"
                f"<td>{_safe(r.get('mae',''))}</td>"
                f"<td>{_safe(r.get('rmse',''))}</td>"
                f"<td>{_safe(r.get('r2',''))}</td>"
                f"<td>{_safe(r.get('cv_rmse',''))}</td>"
                "</tr>"
            )

    # Notes html
    notes_html = "<p class='muted'>No notes</p>"
    if notes:
        notes_html = (
            "<ul>"
            + "".join(
                [f"<li><b>{_safe(k)}</b>: {_safe(v)}</li>" for k, v in notes.items()]
            )
            + "</ul>"
        )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Universal ML Detector - AutoML Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background:#0b1220; color:#e8eefc; }}
    .muted {{ color:#a9b4d0; font-size: 12px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ background:#121a2c; border:1px solid #22304e; border-radius:14px; padding:14px; }}
    img {{ width:100%; border-radius:12px; border:1px solid #22304e; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ border-bottom:1px solid #22304e; padding:8px; text-align:left; }}
    th {{ color:#cfe0ff; }}
    pre {{ background:#0e1627; padding:10px; border-radius:10px; overflow:auto; }}
    .tag {{ display:inline-block; background:#1b2b4a; border:1px solid #2a3d62; padding:4px 8px; border-radius:999px; margin-right:8px; }}
  </style>
</head>
<body>

  <h1>DataAutoPilot Analysis Report</h1>
  <p class="muted">Generated at: {now}</p>

  <div class="card">
    <span class="tag">{_safe(task_type.upper())}</span>
    <span class="tag">Target: {_safe(target)}</span>
    <span class="tag">Shape: {_safe(data_shape)}</span>
    <p class="muted">Dataset: {_safe(dataset_path)}</p>
  </div>

  <div class="card">
    <h2>Best Model</h2>
    <p><b>{_safe(best_model_name)}</b></p>
    <pre>{_safe(json.dumps(best_metric, indent=2))}</pre>
  </div>

  <div class="card">
    <h2>Leaderboard</h2>
    <table>
      <thead><tr>{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Notes</h2>
    {notes_html}
  </div>

  <h2>Artifacts</h2>
  <div class="grid">
    {_img_block("Correlation Heatmap", corr_png, output_dir)}
    {_img_block("Feature Importance", fi_png, output_dir)}
    {_img_block("Permutation Importance", pi_png, output_dir)}
    {_img_block("SHAP Summary", shap_summary_png, output_dir)}
    {_img_block("SHAP Waterfall", shap_waterfall_png, output_dir)}
  </div>

  <div class="card">
    <h2>EDA Text Summary (snippet)</h2>
    <pre>{_safe(eda_text) if eda_text else "EDA report not found."}</pre>
  </div>

</body>
</html>
"""

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # ✅ Convert HTML → PDF
    pdf_path = os.path.join(output_dir, "report.pdf")

    config = pdfkit.configuration(
        wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    )
    options = {"enable-local-file-access": None}
    pdfkit.from_file(report_path, pdf_path, configuration=config, options=options)
    print("✅ PDF report saved:", pdf_path)

    return report_path
