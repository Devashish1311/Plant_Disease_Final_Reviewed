const $ = id => document.getElementById(id);
let DATA = null;
const fmt = x => x == null ? "—" : (100 * x).toFixed(2) + "%";
function classScores(cm, labels) {
  return labels.map((label, i) => {
    const tp = cm[i][i];
    const actual = cm[i].reduce((a,b)=>a+b,0);
    const predicted = cm.reduce((a,row)=>a+row[i],0);
    const precision = predicted ? tp/predicted : 0;
    const recall = actual ? tp/actual : 0;
    const f1 = (precision+recall) ? 2*precision*recall/(precision+recall) : 0;
    return {label, precision, recall, f1, support: actual};
  });
}
function render() {
  const fs = $("featureSelect").value, model = $("modelSelect").value;
  const row = DATA.metrics.find(x => x.feature_set === fs && x.model === model);
  if (!row) return;
  $("metricCards").innerHTML = [
    ["Macro-F1", row.macro_f1], ["Accuracy", row.accuracy],
    ["Macro precision", row.macro_precision], ["Macro recall", row.macro_recall]
  ].map(([name,val]) => `<div class="metric"><strong>${fmt(val)}</strong><span>${name}</span></div>`).join("");
  const all = DATA.metrics.filter(x => x.feature_set === fs);
  $("bars").innerHTML = all.map(x => `<div class="barrow"><span>${x.model}</span><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,100*x.macro_f1))}%"></div></div><b>${fmt(x.macro_f1)}</b></div>`).join("");
  const key = `${fs}__${model}`, cm = DATA.confusion_matrices[key], labels = DATA.classes;
  if (cm) {
    let html = `<table class="matrix"><thead><tr><th>Actual ↓ / Predicted →</th>${labels.map(x=>`<th>${x}</th>`).join("")}</tr></thead><tbody>`;
    cm.forEach((r,i)=>{html += `<tr><th>${labels[i]}</th>${r.map(v=>`<td>${v}</td>`).join("")}</tr>`});
    $("matrix").innerHTML = html + "</tbody></table>";
    $("classReport").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test support</th></tr></thead><tbody>${
      classScores(cm,labels).map(x=>`<tr><td>${x.label}</td><td>${fmt(x.precision)}</td><td>${fmt(x.recall)}</td><td>${fmt(x.f1)}</td><td>${x.support}</td></tr>`).join("")
    }</tbody></table></div>`;
  } else {
    $("matrix").textContent = "Confusion matrix unavailable.";
    $("classReport").textContent = "Per-class metrics unavailable.";
  }
  $("bestParams").textContent = row.best_params ? "Best parameters selected using training-only cross-validation: " + JSON.stringify(row.best_params) : "Majority-class baseline; no hyperparameters tuned.";
}
async function init() {
  try {
    const res = await fetch("results.json", {cache:"no-store"});
    if (!res.ok) throw new Error("results.json not found");
    DATA = await res.json();
    $("status").hidden = true; $("explorer").hidden = false;
    [...new Set(DATA.metrics.map(x=>x.feature_set))].forEach(v=>$("featureSelect").add(new Option(v,v)));
    [...new Set(DATA.metrics.map(x=>x.model))].forEach(v=>$("modelSelect").add(new Option(v,v)));
    $("featureSelect").value = "Colour + GLCM + LBP";
    $("modelSelect").value = "Random Forest";
    $("featureSelect").addEventListener("change",render);
    $("modelSelect").addEventListener("change",render);
    render();
  } catch(e) {
    $("status").hidden = false;
    $("status").textContent = "The results file could not be loaded. Make sure results.json is published beside index.html.";
  }
}
init();