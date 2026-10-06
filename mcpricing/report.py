"""Build shareable Markdown, HTML and SVG reports from saved experiment data."""
import csv
import html
import json
import math
from pathlib import Path

COLORS={'IID':'#2563eb','IS':'#15803d','AV':'#9333ea','CV':'#db2777','MH':'#b45309'}


def figure(rows,scenario):
    strikes=sorted({float(r['strike']) for r in rows if r['scenario']==scenario})
    width,height=370*len(strikes),400
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Monte Carlo RMSE by sample budget for {scenario}">',
        '<rect width="100%" height="100%" fill="white"/>','<g font-family="Arial, sans-serif" fill="#18243b">',
        f'<text x="24" y="27" font-size="19" font-weight="bold">{scenario.capitalize()}: convergence by payoff budget</text>',
        '<text x="24" y="49" font-size="12">Logarithmic axes. RMSE versus numerical integration. Lower is better.</text>']
    selected=[r for r in rows if r['scenario']==scenario]
    ymin=min(float(r['rmse']) for r in selected)*.8
    ymax=max(float(r['rmse']) for r in selected)*1.2
    for panel,strike in enumerate(strikes):
        group=[r for r in selected if float(r['strike'])==strike]
        xmin=min(int(r['n']) for r in group)
        xmax=max(int(r['n']) for r in group)
        if xmax==xmin:
            xmin,xmax=xmin/2,xmax*2
        left,top,w,h=panel*370+64,90,278,222
        sx=lambda x:left+w*math.log(x/xmin)/math.log(xmax/xmin)
        sy=lambda y:top+h-h*math.log(y/ymin)/math.log(ymax/ymin)
        parts.append(f'<text x="{left}" y="77" font-size="14" font-weight="bold">Strike {strike:g}</text>')
        for i in range(5):
            y=ymin*(ymax/ymin)**(i/4)
            py=sy(y)
            parts.append(f'<line x1="{left}" x2="{left+w}" y1="{py}" y2="{py}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="{left-7}" y="{py+4}" text-anchor="end" font-size="10">{y:.3g}</text>')
        for n in sorted({int(r['n']) for r in group}):
            parts.append(f'<text x="{sx(n)}" y="{top+h+20}" text-anchor="middle" font-size="10">{n:,}</text>')
        parts.append(f'<text x="{left+w/2}" y="{top+h+39}" text-anchor="middle" font-size="11">Payoff evaluations</text>')
        for i,(method,color) in enumerate(COLORS.items()):
            points=sorted([r for r in group if r['method']==method],key=lambda r:int(r['n']))
            coords=' '.join(f'{sx(int(r["n"])):.2f},{sy(float(r["rmse"])):.2f}' for r in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            for r in points:
                parts.append(f'<circle cx="{sx(int(r["n"])):.2f}" cy="{sy(float(r["rmse"])):.2f}" r="3.5" fill="{color}"/>')
            parts.append(f'<text x="{left+i*55}" y="383" fill="{color}" font-size="12">{method}</text>')
    return '\n'.join(parts+['</g></svg>'])


def render_report(directory):
    directory=Path(directory)
    with (directory/'summary.csv').open(encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    n=max(int(r['n']) for r in rows)
    intro=(f'{meta["total_estimates"]:,} estimates across {len(meta["scenarios"])} synthetic distributions, '
           f'{len(meta["strikes"])} strikes and {len(meta["sizes"])} payoff budgets. '
           f'{meta["pilot_groups"]} independent pilot groups and {meta["repeats_per_group"]} held-out repeats per group. '
           f'Tables below use {n:,} payoff evaluations per estimate.')
    lines=['# Benchmark report','',intro,'',
        'These are synthetic pricing experiments, not trading returns or market-calibrated results. '
        'Each proposal and control coefficient is fitted on separate pilot samples. '
        'The original 6.1x figure came from a smaller experiment; this report evaluates the whole tuning procedure.','']
    htmlparts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Monte Carlo pricing benchmark</title><style>body{font:15px/1.65 system-ui,sans-serif;color:#18243b;max-width:1180px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:42px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce3ed;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f1f5f9}svg{max-width:100%;height:auto}.scroll{overflow:auto}.note{background:#f1f5f9;padding:16px;border-radius:8px}</style>',
        '<h1>Monte Carlo pricing benchmark</h1>',f'<p>{html.escape(intro)}</p>',
        '<p class="note">Synthetic terminal pricing models. Results measure estimator accuracy and cost, not market performance.</p>']
    headers=['Strike','Method','RMSE','Empirical VRF (95% interval)','Population VRF','Coverage','Runtime ms','Pilot ms','Efficiency incl. pilot']
    for scenario in meta['scenarios']:
        svg=figure(rows,scenario)
        (directory/f'{scenario}-convergence.svg').write_text(svg,encoding='utf-8')
        lines.extend([f'## {scenario.capitalize()}','',f'![RMSE convergence]({scenario}-convergence.svg)','',
            '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        htmlparts.extend([f'<h2>{html.escape(scenario.capitalize())}</h2>',svg,'<div class="scroll"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'])
        for r in rows:
            if r['scenario']!=scenario or int(r['n'])!=n:
                continue
            interval=f'{float(r["variance_ratio_vs_iid"]):.2f} ({float(r["variance_ratio_ci_low"]):.2f}-{float(r["variance_ratio_ci_high"]):.2f})'
            values=[f'{float(r["strike"]):g}',r['method'],f'{float(r["rmse"]):.5f}',interval,
                f'{float(r["population_variance_ratio"]):.2f}' if r['population_variance_ratio'] else 'N/A',
                f'{float(r["coverage_95"])*100:.0f}%',f'{float(r["mean_seconds"])*1000:.2f}',
                f'{float(r["mean_pilot_seconds"])*1000:.2f}',f'{float(r["efficiency_including_one_pilot"]):.2f}']
            lines.append('| '+' | '.join(values)+' |')
            htmlparts.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in values)+'</tr>')
        lines.append('')
        htmlparts.append('</tbody></table></div>')
    notes=[
        'VRF = IID estimator variance / method estimator variance at the same payoff budget. Above 1 means lower variance. AV uses half as many independent pair averages, with two payoffs per pair.',
        'Population VRF is obtained by integrating payoff moments under the model, averaged over the frozen pilot choices. It checks the empirical comparison without relying on another finite set of Monte Carlo estimates. MH has no population VRF here because its serial autocovariances are not integrated.',
        'Bootstrap intervals resample pilot groups and held-out repeats. Five pilot groups still give limited evidence about tuning variability; the intervals are exploratory, not simultaneous confidence bands over the grid.',
        'Coverage is the observed fraction of nominal 95% intervals containing the reference. It has sampling uncertainty. MH uses batch means; IID formulas must not be used on correlated chain draws.',
        'Efficiency incl. pilot = (IID variance x IID runtime) / (method variance x (runtime + one full method-specific pilot)). Above 1 is better. This is a variance-times-cost comparison, not an observed speedup at equal RMSE. Reusing a pilot amortises its cost; per-call figures excluding pilots are in summary.csv.',
        'Pilot time includes all 12 IS candidate trials, all six MH step trials, or the CV coefficient fit, respectively. Shared quadrature, imports and report writing are excluded from estimator timing. IS weight ESS and MH payoff ESS are different diagnostics and should not be directly compared.',
        'The benchmark samples a one-maturity terminal law. No calibrated volatility surface, dynamic hedging, transaction costs or realised trading profit is claimed. Quadrature is the practical preferred solver for this simple one-dimensional contract.',
        'Antithetic sampling is performed within each mixture component. Between-component randomness can outweigh the within-component negative covariance, so it is not guaranteed to beat IID for these mixtures.',
        'Hardware timing is machine-specific and sub-millisecond measurements are noisy. The code stores source hashes, random seeds, parameters, raw trials and environment information. Reordering or selecting a subset of scenarios leaves their random streams unchanged.'
    ]
    lines.extend(['## Reading the results','']+['- '+note for note in notes]+['','## Reproduce','',
        '```sh','python -m mcpricing benchmark --output local-results','```','',
        'Use the exact arguments in metadata.json for non-default runs. Existing result directories are never overwritten.'])
    htmlparts.extend(['<h2>Reading the results</h2><ul>']+[f'<li>{html.escape(note)}</li>' for note in notes]+['</ul></html>'])
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (directory/'report.html').write_text('\n'.join(htmlparts),encoding='utf-8')
