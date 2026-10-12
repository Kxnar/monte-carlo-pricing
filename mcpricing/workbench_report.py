"""Render the workbench's saved evidence without running any estimators."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path


LABELS = {
    "n_requested": "Physical paths / samples", "mean": "Mean estimate",
    "mean_estimate": "Mean estimate", "mean_standard_error": "Mean reported SE",
    "standard_error": "Reported SE", "reference_standard_error": "Shared reference SE",
    "reference_se": "Shared reference SE", "rmse": "Observed deviation RMSE",
    "observed_deviation_rmse": "Observed deviation RMSE", "pooled_standard_error": "Pooled native SE",
    "rmse_upper_95": "RMSE upper 95%", "coverage_95": "95% interval coverage",
    "coverage": "95% interval coverage", "observed_break_even": "Observed crossing (calls)",
    "mean_cumulative_seconds": "Cumulative seconds", "timing_difference_upper_95": "Time difference upper 95% (s)",
    "production_seconds": "Production (s)", "tuning_seconds": "Tuning (s)",
    "decision_seconds": "Decision (s)", "setup_seconds": "Setup (s)",
    "peak_memory_bytes": "Peak traced bytes", "price_se": "Price SE", "delta_se": "Delta SE",
    "vega_se": "Vega SE", "vega": "Vega / unit volatility",
    "estimator_coverage": "Estimator-only intervals against stochastic reference",
    "combined_uncertainty_coverage": "Combined native/reference uncertainty coverage",
}


def _label(key):
    if " / " in key:
        return " / ".join(_label(part) for part in key.split(" / "))
    return LABELS.get(key, key.replace("_", " ").capitalize())


def _format(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "unavailable"
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return str(value)


def _flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = f"{prefix} / {key}" if prefix else key
        if isinstance(item, dict):
            out.update(_flatten(item, name))
        else:
            out[name] = item
    return out


def _tables(title, value):
    """Expose every saved summary field, with readable nested sections."""
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        def has_records(item):
            if isinstance(item, list):
                return any(isinstance(child, dict) for child in item)
            if isinstance(item, dict):
                return any(has_records(child) for child in item.values())
            return False
        if any(has_records(row) for row in value):
            result = []
            for index, row in enumerate(value):
                identity = row.get("case_id", row.get("horizon", index+1))
                result.extend(_tables(f"{title} · {identity}", row))
            return result
        rows = [_flatten(row) for row in value]
        if rows:
            return [(title, rows)]
        return [(title, [{"status": "No saved observations"}])]
    if isinstance(value, dict):
        scalar = {key: item for key, item in value.items() if not isinstance(item, (dict, list))}
        result = [(title, [{"field": _label(key), "value": item} for key, item in scalar.items()])] if scalar else []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                result.extend(_tables(f"{title} · {_label(key)}", item))
        return result
    if isinstance(value, list):
        return [(title, [{"value": item} for item in value])]
    return [(title, [{"value": value}])]


def _table_html(rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        return "<p>No saved observations.</p>"
    head = "".join(f"<th>{html.escape(_label(key))}</th>" for key in fields)
    body = "".join("<tr>"+"".join(f"<td>{html.escape(_format(row.get(key)))}</td>" for key in fields)+"</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _table_md(rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    def escape(value):
        return _format(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", "<br>")
    if not fields:
        return "No saved observations."
    lines = ["| "+" | ".join(escape(_label(key)) for key in fields)+" |",
             "| "+" | ".join("---" for _ in fields)+" |"]
    lines += ["| "+" | ".join(escape(row.get(key)) for key in fields)+" |" for row in rows]
    return "\n".join(lines)


def _raw_inventory(directory):
    inventory, failures, control_calls = [], [], {}
    counts = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            count = 0
            statuses = {}
            for row in reader:
                count += 1
                if "suite" in fields and "status" in fields:
                    suite = row["suite"]
                    counts[suite] = counts.get(suite, 0)+1
                    status = row.get("status") or "missing status"
                    statuses[status] = statuses.get(status, 0)+1
                    if status != "ok":
                        failures.append(dict(file=path.name, row=count+1, suite=suite, case_id=row.get("case_id"), status=status))
                    if suite == "A" and row.get("method") == "geometric_cv" and row.get("coefficient_id"):
                        coefficient = row["coefficient_id"]
                        key = (coefficient, row.get("n_requested"))
                        control_calls.setdefault(key, []).append(float(row.get("production_seconds") or 0.))
            inventory.append(dict(file=path.name, rows=count, statuses=statuses or None))
    return inventory, counts, failures, control_calls


def _asian_costs(controls, calls):
    rows = []
    if not isinstance(controls, list):
        return rows
    for control in controls:
        coefficient = control.get("coefficient_id")
        shared = sum(len(times) for (identifier, _), times in calls.items() if identifier == coefficient)
        pilot = control.get("seconds")
        if not shared or not isinstance(pilot, (int, float)):
            continue
        for (identifier, n), times in calls.items():
            if identifier != coefficient:
                continue
            production = sum(times)/len(times)
            rows.append(dict(case_id=control.get("case_id"), group=control.get("group"), n_requested=n,
                             mean_production_seconds=production, full_pilot_seconds=pilot,
                             single_call_including_full_pilot_seconds=production+pilot,
                             production_calls_sharing_fit=shared,
                             amortized_seconds_per_call=production+pilot/shared))
    return rows


def _validation_failures(value, path="validation"):
    failures = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{path}.{key}"
            if item is False and (key in ("passed", "pass", "ok") or key.endswith("_pass") or key.endswith("_passed")):
                failures.append(dict(check=name, result="failed"))
            elif isinstance(item, (dict, list)):
                failures.extend(_validation_failures(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_validation_failures(item, f"{path}[{index}]"))
    return failures


def render_workbench_report(directory):
    """Create report.md/report.html from saved JSON and CSV evidence only.

    Rerendering is allowed in a nonempty experiment directory. No raw artifact
    is modified, and no estimator, random generator or reference is called.
    """
    directory = Path(directory)
    summary_path = directory/"summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain an object")
    artifacts = {}
    for path in sorted(directory.glob("*.json")):
        artifacts[path.name] = json.loads(path.read_text(encoding="utf-8"))
    validation = artifacts.get("validation.json", summary.get("validation", {}))
    environment = artifacts.get("environment.json", artifacts.get("metadata.json", {}))
    configuration = artifacts.get("configuration.json", artifacts.get("config.json", {}))
    inventory, raw_counts, raw_failures, control_calls = _raw_inventory(directory)
    checks = _validation_failures(validation)
    expected = summary.get("expected_counts", {})
    recorded = summary.get("actual_counts", {})
    count_rows = [{"suite": suite, "expected": expected.get(suite), "saved summary": recorded.get(suite),
                   "raw CSV rows": raw_counts.get(suite),
                   "counts match": expected.get(suite) == recorded.get(suite) == raw_counts.get(suite)}
                  for suite in sorted((set(expected)|set(recorded)|set(raw_counts)) & {"A", "B", "C"})]
    mismatches = [row for row in count_rows if not row["counts match"]]
    passed = validation.get("passed", summary.get("validation", {}).get("passed")) if isinstance(validation, dict) else None
    state = "PASS" if passed is True and not checks and not raw_failures and not mismatches else "FAIL" if passed is False or checks or raw_failures or mismatches else "UNVERIFIED"
    mode = configuration.get("stage", configuration.get("mode", "saved run")) if isinstance(configuration, dict) else "saved run"
    sections = []
    def section(title, note, data):
        sections.append((title, note, _tables(title, data)))
    section("Evidence and validation", "Counts are read from the saved raw CSVs. A failed or missing gate limits the result; failures remain visible.",
            {"raw_counts": count_rows, "saved_expected_counts": expected, "saved_actual_counts": recorded,
             "validation": validation, "raw_failures": raw_failures, "failed_checks": checks})
    section("A · Arithmetic Asian accuracy", "RMSE measures observed deviation from a stochastic QuantLib reference, not exact truth. The shared reference SE is not divided by the number of native repeats. Estimator-only interval coverage compares against that stochastic reference while omitting reference error; it is not ordinary exact-target CI coverage. Combined native/reference uncertainty coverage is the primary uncertainty comparison. Coverage has Wilson intervals; group bootstrap intervals with three pilot groups are exploratory. A one-fixing control can have tiny residual SE and does not imply infinite efficiency.",
            {**summary.get("A", {}), "pilot_cost_accounting": _asian_costs(artifacts.get("A-controls.json", []), control_calls)})
    section("B · Joint portfolio price and risk", "Signed leg quantities retain the full cross-leg covariance for price, delta and vega SEs. Vega is per unit absolute volatility. Hedge and scenario P&L are first-order GBM sensitivities; they are not VaR. Common-union singleton and batch comparisons use the same workload; own-grid comparisons have different workloads.", summary.get("B", {}))
    section("C · Reuse at equal observed accuracy", "A crossing requires the upper 95% bootstrap RMSE to meet the frozen target, budget reselection within the bootstrap, and a timing difference interval favouring reuse. Qualification is for pooled RMSE over a horizon, not a per-call accuracy guarantee. The first observed crossing is exploratory and has no familywise 95% claim across horizons or cases. No observed crossing is a valid outcome. Quadrature is a measured warm deterministic baseline; its refinement diagnostic is not a stochastic SE.", summary.get("C", {}))
    section("Costs and memory", "Production, setup, pilot tuning, reuse decisions, references and reporting are separate costs. Charging a full pilot to one call differs from amortising it across a workload. Memory is an untimed tracemalloc allocation measurement at fixed block size, not process RSS. Times describe this machine and implementation.",
            {"costs": summary.get("costs", {}), "memory": artifacts.get("memory.json", {}),
             "portfolio_timing": artifacts.get("B-timings.json", {}), "portfolio_memory": artifacts.get("B-memory.json", {})})
    section("Reproducibility", "The frozen configuration and source hashes identify the evaluated code and environment. Earlier reconstruction results remain separate. This report is regenerated from saved artifacts without resampling.",
            {"configuration_sha256": summary.get("configuration_sha256"), "configuration": configuration,
             "environment": environment, "raw_files": inventory})
    hashes = [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted(directory.iterdir()) if path.is_file() and path.name not in ("report.html", "report.md")]
    section("Artifact hashes", "SHA-256 digests cover the saved input files; generated reports are excluded so regeneration is stable.", hashes)
    intro = "A bounded synthetic pricing workbench: independent Asian references, portfolio risk, and finite-workload proposal reuse. Results are descriptive measurements, not promised performance or market calibration."
    md = ["# Pricing workbench", "", intro, "", f"Validation: **{state}** · Stage: **{_format(mode)}**", ""]
    body = []
    for index, (title, note, tables) in enumerate(sections):
        md += [f"## {title}", "", note, ""]
        content = []
        for table_title, rows in tables:
            md += [f"### {table_title}", "", _table_md(rows), ""]
            content += [f"<h3>{html.escape(table_title)}</h3>", _table_html(rows)]
        body.append(f'<section id="s{index}"><h2>{html.escape(title)}</h2><p class="note">{html.escape(note)}</p>'+"".join(content)+"</section>")
    details = "".join(f'<details><summary>{html.escape(name)}</summary><pre>{html.escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))}</pre></details>' for name, value in artifacts.items())
    nav = "".join(f'<a href="#s{index}">{html.escape(title.split(" · ")[0])}</a>' for index, (title, _, _) in enumerate(sections))
    css = """*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#182a36;font:15px/1.6 system-ui,sans-serif}header{background:#142f41;color:#fff;padding:44px max(24px,calc((100vw - 1240px)/2))}header p{max-width:820px;color:#d5e0e7}.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#9cd4d0}h1{font-size:clamp(30px,4vw,48px);line-height:1.15;margin:12px 0 18px}h2{font-size:25px;line-height:1.3}h3{font-size:16px;margin-top:30px}.badge{display:inline-block;border:1px solid #86bdb8;border-radius:6px;padding:5px 12px;margin-right:12px}nav{max-width:1240px;margin:0 auto;padding:18px 24px;display:flex;gap:18px;flex-wrap:wrap}a{color:#076e75;text-decoration:none}main{max-width:1240px;margin:auto;padding:0 24px 40px}section{background:white;border:1px solid #dfe6eb;border-radius:10px;margin:18px 0;padding:24px}.note{color:#49616f;max-width:1000px}.table-wrap{overflow-x:auto;max-height:560px;border:1px solid #e2e8ec;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e6ecef;vertical-align:top;max-width:340px;overflow-wrap:anywhere}th{background:#edf3f5;position:sticky;top:0;white-space:nowrap}tr:nth-child(even){background:#fafcfd}details{margin:12px 0;border:1px solid #dfe6eb;border-radius:6px;padding:12px}summary{cursor:pointer;color:#076e75;font-weight:600}pre{font:12px/1.6 ui-monospace,monospace;overflow:auto;max-height:500px;white-space:pre-wrap;overflow-wrap:anywhere}footer{color:#607480;margin:24px 0}@media(max-width:600px){main{padding:0 12px}section{padding:16px}header{padding:30px 20px}th,td{padding:8px}}@media print{nav{display:none}.table-wrap{max-height:none;overflow:visible}section{break-inside:avoid}details{display:none}header{padding:20px;color:#182a36;background:white}header p{color:#49616f}}"""
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Pricing workbench · saved evidence</title><style>'+css+'</style></head><body><header><div class="eyebrow">Monte Carlo pricing lab / saved evidence</div><h1>Pricing workbench</h1><p>'+html.escape(intro)+'</p><span class="badge">Validation '+state+'</span><span>'+html.escape(_format(mode))+'</span></header><nav>'+nav+'</nav><main>'+"".join(body)+'<section><h2>Saved JSON details</h2>'+details+'</section><footer>Rendered from saved artifacts. No sampling or reference evaluation occurred during report generation.</footer></main></body></html>'
    (directory/"report.md").write_text("\n".join(md), encoding="utf-8")
    (directory/"report.html").write_text(document, encoding="utf-8")
    return {"report_markdown": str(directory/"report.md"), "report_html": str(directory/"report.html"),
            "validation_status": state, "raw_counts": raw_counts}
