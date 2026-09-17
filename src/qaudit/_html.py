"""Offline HTML presentation shared by audit reports and the synthetic demo.

All report content is embedded as inert JSON and rendered with ``textContent``.
The document has no remote scripts, fonts, styles, images, or data requests.
"""
from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .report import AuditReport


def report_view(report: AuditReport) -> dict[str, Any]:
    """Compute presentation facts before lossless JSON key encoding.

    ``details`` can use the serializer's typed-mapping representation when
    keys collide. Rendering must not infer reserved flags from that encoded
    structure, nor silently bless a malformed mutable result contract.
    """
    from .report import _result_contract_problem

    problems = [f"result[{i}]: {problem}" for i, result in enumerate(report.results)
                if (problem := _result_contract_problem(result)) is not None]
    flags = [{key: result.details.get(key) is True
              for key in ("not_judged", "unresolved", "superseded")}
             if isinstance(result.details, dict) else {}
             for result in report.results]
    judged = sum(result.status.value == "pass"
                 and not flag.get("not_judged") and not flag.get("unresolved")
                 for result, flag in zip(report.results, flags))
    if problems or report.errors:
        status = "ERROR"
    elif report.failures or report.warnings:
        status = "FLAGGED"
    elif not judged:
        status = "INCOMPLETE"
    else:
        status = "CLEAN"
    return {"status": status, "judged_passes": judged if not problems else 0,
            "result_flags": flags, "contract_problems": problems}


def render_html(payload: dict[str, Any], *, title: str) -> str:
    """Render a report bundle as one portable, interactive HTML document."""
    # A script element ends at </script> even when that text occurs inside a
    # JSON string. Escape HTML-significant characters before embedding data.
    data = (json.dumps(payload, ensure_ascii=True, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e"))
    return (_PAGE.replace("__PAYLOAD__", data, 1)
            .replace("__TITLE__", html.escape(title, quote=True), 1))


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="author" content="Chung Shing Mars Wong">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--paper:#f4f5f1;--ink:#182721;--muted:#55645c;--line:#d8dfd5;--green:#135d48;--mint:#e1eee4;--amber:#835400;--red:#aa3034}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit}button,a,input,select,summary{-webkit-tap-highlight-color:transparent}a{color:var(--green)}button{cursor:pointer}
:focus-visible{outline:3px solid #146ec6;outline-offset:4px}[hidden]{display:none!important}
.shell{max-width:1260px;margin:auto;padding:0 40px}header{border-bottom:1px solid var(--line)}.masthead{display:flex;justify-content:space-between;gap:20px;align-items:center;min-height:86px}
.brand{font-size:18px;line-height:1.3;font-weight:650;max-width:720px}.edition{flex-shrink:0}.edition{font:11px/1.4 ui-monospace,SFMono-Regular,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:1.3px}
.hero{padding:48px 0 30px;display:grid;grid-template-columns:1.5fr 1fr;gap:54px;align-items:end}.eyebrow{font-size:11px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:var(--green)}h1{font-size:clamp(32px,4vw,50px);letter-spacing:-2px;font-weight:620;line-height:1.08;margin:12px 0 0}h2{font-size:21px;letter-spacing:-.5px;margin:0}h3{font-size:16px;margin:0}p{margin:10px 0}.lead{color:var(--muted);max-width:520px}.notice{border:1px solid var(--line);border-left:3px solid var(--green);background:#ecf0e9;padding:13px 17px;color:#3f5247;font-size:13px;border-radius:0 8px 8px 0}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:24px 0 32px}.metric{padding:20px 24px;background:#fff}.metric-label{font-size:11px;font-weight:650;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted)}.metric-value{font-size:32px;font-weight:600;letter-spacing:-1px;line-height:1.3;margin:8px 0 3px}.metric-hint{font-size:12px;color:var(--muted)}
.section-head{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:24px 0 14px}.section-head p{font-size:12px;color:var(--muted);margin:0}.panel{background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;text-align:left}caption{text-align:left;padding:14px 18px;font-size:12px;color:var(--muted)}th{background:#eef1eb;color:var(--muted);font-size:10px;letter-spacing:1px;text-transform:uppercase;font-weight:700;white-space:nowrap}td,th{padding:12px 18px;border-bottom:1px solid #e4e8e1}tbody tr:last-child td{border-bottom:0}td{font-size:13px;vertical-align:top}.case-link{border:0;padding:0;background:none;text-align:left;font-weight:650;color:var(--green)}.case-link:hover{text-decoration:underline}.case-row[aria-selected=true]{background:#f0f7ef}.count{font:12px ui-monospace,SFMono-Regular,monospace;white-space:nowrap}.badge{display:inline-block;border-radius:4px;padding:3px 7px;font-size:10px;letter-spacing:.4px;font-weight:750;white-space:nowrap}.pass,.clean{background:#e4f0e7;color:#225e40}.warn,.flagged{background:#fff0d1;color:#805409}.fail,.error{background:#f9e4e3;color:#a12630}.skip,.incomplete{background:#eaedf0;color:#515d66}.subtle{color:var(--muted);font-size:12px}.selected{display:grid;grid-template-columns:1.5fr 1fr;gap:24px;padding:24px}.description{white-space:pre-line;color:var(--muted);font-size:14px}.expected{font-size:12px;padding:0;list-style:none;margin-bottom:0}.expected li{padding:5px 0;overflow-wrap:anywhere}.expected strong{margin-right:8px}.expected .hit{color:var(--green)}.expected .miss{color:var(--red)}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;padding:18px 24px;background:#f7f9f4;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}label{display:flex;flex-direction:column;gap:5px;font-size:11px;font-weight:650;color:var(--muted)}select,input{height:37px;background:white;border:1px solid #bcc8be;border-radius:6px;padding:6px 10px;color:var(--ink);font-size:13px}input{min-width:230px}button.secondary{border:1px solid #bcc8be;border-radius:6px;background:white;padding:8px 12px;font-size:12px;color:var(--ink)}.results-info{margin:0;padding:12px 24px;font-size:12px;color:var(--muted)}.finding{border-top:1px solid #e4e8e1;padding:17px 24px}.finding-head{display:flex;justify-content:space-between;gap:12px}.check-id{font:12px/1.6 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;font-weight:600}.finding p{font-size:13px;white-space:pre-line;overflow-wrap:anywhere}.remediation{background:#f5f7f2;border-left:2px solid #9cb39f;padding:8px 12px}.severity{color:var(--muted);font-size:10px;white-space:nowrap;text-transform:uppercase;letter-spacing:.6px}.flags{font-size:11px;font-weight:650;color:var(--amber);margin-left:8px}details{font-size:12px;margin-top:10px}summary{cursor:pointer;color:var(--green);font-weight:600}pre{background:#f0f3ed;max-height:280px;overflow:auto;padding:15px;border-radius:6px;font:11px/1.65 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.provenance{padding:20px 24px;border-top:1px solid var(--line);background:#f7f9f4}.empty{padding:24px;color:var(--muted)}footer{display:flex;justify-content:space-between;gap:20px;padding:26px 0 40px;font-size:11px;color:var(--muted)}.skip-link{position:absolute;left:12px;top:-100px;background:white;padding:10px}.skip-link:focus{top:12px}.no-script{margin:30px auto;padding:25px;max-width:800px}
@media(max-width:760px){.masthead{align-items:flex-start;flex-direction:column;gap:8px;padding-top:18px;padding-bottom:18px}.brand{font-size:16px}.shell{padding:0 18px}.hero{grid-template-columns:1fr;gap:16px;padding-top:32px}.metrics{grid-template-columns:1fr 1fr}.metric{padding:16px}.selected{grid-template-columns:1fr;padding:20px}.controls{padding:16px;gap:10px}input{min-width:180px;max-width:100%}.edition{font-size:9px}.finding{padding:16px}.section-head{align-items:start;flex-direction:column;gap:4px}td,th{padding:11px 13px}footer{flex-direction:column;gap:5px}.finding-head{flex-wrap:wrap}}
@media print{.controls,.skip-link,.secondary{display:none}.shell{padding:0;max-width:none}.hero{padding-top:15px}.metrics{margin:15px 0}.finding,details,.selected{break-inside:avoid}.panel{overflow:visible}.table-wrap{overflow:visible}pre{max-height:none}.case-row[aria-selected=true]{background:transparent}body{background:white}footer{padding-bottom:0}}
</style>
</head>
<body>
<a class="skip-link" href="#evidence">Skip to audit evidence</a>
<header><div class="shell masthead"><div class="brand">Quantitative Backtest Validation &amp; Bias Detection</div><span class="edition" id="edition">Backtest validation</span></div></header>
<main class="shell">
<section class="hero"><div><div class="eyebrow" id="eyebrow">Audit evidence</div><h1 id="title">Backtest validation report</h1></div><p class="lead" id="intro">Inspect the findings, understand the evidence, and trace every result to the inputs and configuration that produced it.</p></section>
<p class="notice" id="notice">A clean advisory result is not deployment approval. Skipped and unresolved checks are disclosed below; use report.gate(...) to enforce the coverage your workflow requires.</p>
<section class="metrics" aria-label="Report overview" id="metrics"></section>
<section id="cases-section" aria-labelledby="cases-title"><div class="section-head"><h2 id="cases-title">Detection matrix</h2><p>Select a case to inspect its evidence</p></div><div class="panel table-wrap"><table><caption id="matrix-caption">Expected flags count detector prefixes, not independently estimated detection rates.</caption><thead><tr><th scope="col">Synthetic case</th><th scope="col">Expected flags</th><th scope="col">Audit status</th><th scope="col">Pass</th><th scope="col">Warn / fail</th><th scope="col">Skip / error</th></tr></thead><tbody id="cases"></tbody></table></div></section>
<section id="evidence" aria-labelledby="evidence-heading"><div class="section-head"><h2 id="evidence-heading">Audit evidence</h2><p>Failures and errors appear first</p></div><div class="panel"><div class="selected"><div><h3 id="case-name"></h3><p class="description" id="case-description"></p><p class="subtle" id="sample"></p></div><div id="expected-wrap"><h3>Planted defects</h3><ul class="expected" id="expected"></ul></div></div>
<div class="controls"><label for="status">Status<select id="status"><option value="all">All statuses</option><option value="findings">Warnings, failures &amp; errors</option><option value="pass">Pass</option><option value="warn">Warning</option><option value="fail">Failure</option><option value="skip">Skipped</option><option value="error">Error</option></select></label><label for="family">Check family<select id="family"><option value="all">All families</option></select></label><label for="search">Search evidence<input id="search" type="search" placeholder="Check ID, finding, remediation…"></label><button class="secondary" id="reset" type="button">Reset filters</button></div>
<p class="results-info" id="results-info" role="status" aria-live="polite"></p><div id="findings"></div><details class="provenance"><summary>Provenance &amp; reproducibility</summary><p>Versions, configuration, input fingerprints and timings from this run. Hashes identify content; they are not a digital signature or independent data verification.</p><pre id="provenance"></pre></details></div></section>
<footer><span id="generated"></span><span>Self-contained report · works offline · no external requests</span></footer>
</main>
<noscript><p class="no-script">JavaScript is required to explore this report. The complete report data remains embedded in the document as JSON.</p></noscript>
<script id="qaudit-data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
(() => {
  const data = JSON.parse(document.getElementById("qaudit-data").textContent);
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = String(text); if (cls) node.className = cls; return node; };
  const counts = report => { const out = {pass:0,warn:0,fail:0,skip:0,error:0}; for (const r of report.results) out[r.status]++; return out; };
  const verdict = c => c.presentation.status;
  const badge = status => el("span", status.toUpperCase(), "badge " + status.toLowerCase());
  const metric = (label,value,hint) => { const node=el("div",undefined,"metric"); node.append(el("div",label,"metric-label"),el("div",value,"metric-value"),el("div",hint,"metric-hint")); $("metrics").append(node); };
  const isDemo = data.schema === "qaudit.demo.v1";
  const all = data.cases.flatMap(c => c.report.results);
  $("edition").textContent = "v" + data.qaudit_version + " / " + (isDemo ? "synthetic demonstration" : "audit report");
  $("generated").textContent = "Generated " + data.generated_at;
  $("cases-section").hidden = !isDemo;
  if (isDemo) {
    $("eyebrow").textContent = "Research integrity / demonstration";
    $("title").textContent = "Synthetic backtest audit results";
    $("intro").textContent = "Backtests with planted defects and a clean control. Select a case to inspect the checks, findings and remediation.";
    $("notice").textContent = "Synthetic data only. This demonstrates known failure modes; it does not estimate real-world detection rates or establish future trading performance. Demo guards: " + (data.demo_passed ? "PASSED." : "FAILED - inspect errors and missed flags.");
    const hits = data.cases.flatMap(c => Object.values(c.expected_flags));
    const clean = data.cases.find(c => c.name === "clean");
    metric("Synthetic cases",data.cases.length,"Deterministic market and probe seeds");
    metric("Expected flags caught",hits.filter(Boolean).length + " / " + hits.length,"WARN or FAIL in each expected prefix");
    metric("Clean control",clean ? verdict(clean) : "Not run",clean ? "Skips and coverage shown in the evidence" : "Run the clean case to include this guard");
    metric("Check errors",all.filter(r => r.status === "error").length,"A crashed check cannot validate a backtest");
  } else {
    $("title").textContent = data.title;
    const c=counts(data.cases[0].report);
    metric("Audit status",verdict(data.cases[0]),"Advisory result; apply your deployment gate");
    metric("Findings",c.warn+c.fail+c.error,"Warnings, failures and check errors");
    metric("Judged passes",data.cases[0].presentation.judged_passes,"Excludes unresolved and unjudged passes");
    metric("Skipped checks",c.skip,"Missing or intentionally superseded coverage");
  }
  let selected = 0;
  const rows=[];
  data.cases.forEach((c,i) => {
    const row=el("tr",undefined,"case-row"), button=el("button",c.name,"case-link");
    button.type="button"; button.setAttribute("aria-controls","evidence"); button.addEventListener("click",()=>{ selectCase(i); $("evidence").scrollIntoView({block:"start"}); });
    const name=el("td"); name.append(button); row.append(name);
    const hits=Object.values(c.expected_flags || {}), n=counts(c.report), status=el("td"); status.append(badge(verdict(c)));
    row.append(el("td",hits.length ? hits.filter(Boolean).length+" / "+hits.length : "Control","count"),status,el("td",n.pass,"count"),el("td",n.warn+" / "+n.fail,"count"),el("td",n.skip+" / "+n.error,"count"));
    $("cases").append(row); rows.push(row);
  });
  function renderFindings() {
    const c=data.cases[selected], status=$("status").value, family=$("family").value, query=$("search").value.toLowerCase().trim();
    const order={error:0,fail:1,warn:2,pass:3,skip:4}, severity={critical:0,high:1,medium:2,low:3,info:4};
    const filtered=c.report.results.map((r,i)=>({...r,view_flags:c.presentation.result_flags[i]})).filter(r => (status==="all" || (status==="findings" ? ["warn","fail","error"].includes(r.status) : r.status===status)) && (family==="all" || r.check.split(".")[0]===family) && (!query || [r.check,r.message,r.remediation,JSON.stringify(r.details)].join(" ").toLowerCase().includes(query))).sort((a,b)=>order[a.status]-order[b.status] || severity[a.severity]-severity[b.severity] || a.check.localeCompare(b.check));
    $("findings").replaceChildren();
    $("results-info").textContent=filtered.length+" of "+c.report.results.length+" checks shown · "+c.name;
    if(!filtered.length) $("findings").append(el("p","No checks match these filters.","empty"));
    for(const r of filtered) {
      const article=el("article",undefined,"finding"), head=el("div",undefined,"finding-head"), identity=el("div");
      identity.append(el("span",r.check,"check-id"));
      const flags=[r.view_flags.unresolved && "Unresolved",r.view_flags.not_judged && "Not judged",r.view_flags.superseded && "Superseded"].filter(Boolean);
      if(flags.length) identity.append(el("span",flags.join(" · "),"flags"));
      const tags=el("div"); tags.append(el("span",r.severity+" ","severity"),badge(r.status)); head.append(identity,tags);
      article.append(head,el("p",r.message));
      if(r.remediation) article.append(el("p","Remediation: "+r.remediation,"remediation"));
      if(Object.keys(r.details).length) { const detail=el("details"); detail.append(el("summary","Measured evidence"),el("pre",JSON.stringify(r.details,null,2))); article.append(detail); }
      $("findings").append(article);
    }
  }
  function selectCase(i) {
    selected=i; const c=data.cases[i]; rows.forEach((row,j)=>row.setAttribute("aria-selected",String(i===j)));
    $("case-name").textContent=c.name; $("case-description").textContent=c.description + (c.presentation.contract_problems.length ? "\nReport structure invalid: " + c.presentation.contract_problems.join("; ") : "");
    const m=c.report.meta; $("sample").textContent=m.n_periods != null ? m.n_periods+" periods × "+m.n_assets+" assets" : "Sample dimensions are not recorded in this report.";
    $("expected-wrap").hidden=!isDemo; $("expected").replaceChildren();
    for(const [prefix,hit] of Object.entries(c.expected_flags || {})) { const li=el("li"); li.append(el("strong",hit ? "CAUGHT" : "MISSED",hit ? "hit" : "miss"),el("span",prefix)); $("expected").append(li); }
    if(!Object.keys(c.expected_flags || {}).length) $("expected").append(el("li","No defects planted. This case is the false-positive control."));
    $("provenance").textContent=JSON.stringify(m.provenance || m,null,2);
    $("family").replaceChildren(); const option=el("option","All families"); option.value="all"; $("family").append(option);
    for(const family of [...new Set(c.report.results.map(r=>r.check.split(".")[0]))].sort()) { const opt=el("option",family); opt.value=family; $("family").append(opt); }
    $("status").value="all"; $("search").value=""; renderFindings();
  }
  $("status").addEventListener("change",renderFindings); $("family").addEventListener("change",renderFindings); $("search").addEventListener("input",renderFindings);
  $("reset").addEventListener("click",()=>{ $("status").value="all"; $("family").value="all"; $("search").value=""; renderFindings(); });
  selectCase(0);
})();
</script>
</body>
</html>
'''
