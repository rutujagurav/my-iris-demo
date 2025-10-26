// app.js (multi-model version)
// Loads docs/models.json and provides inference for multiple model types (linear, rf, mlp)

let exportData = null;

async function loadModels() {
  const r = await fetch('models.json');
  exportData = await r.json();
  console.log("Loaded models:", Object.keys(exportData.models));
}
loadModels();

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b, 0);
  return exps.map(e => e / sum);
}

function scaleFeatures(features) {
  // features: raw numbers
  const mean = exportData.scaler_mean;
  const scale = exportData.scaler_scale;
  return features.map((v,i) => (Number(v) - mean[i]) / scale[i]);
}

// Linear model scoring (logistic, svm exported as linear)
function scoreLinear(coef, intercept, xs) {
  // coef: [n_classes][n_features] OR [n_features] for binary
  // intercept: [n_classes] or scalar
  const nClasses = coef.length;
  const scores = [];
  for (let c = 0; c < nClasses; c++) {
    let s = intercept[c] || 0;
    const row = coef[c];
    for (let i = 0; i < row.length; i++) s += row[i] * xs[i];
    scores.push(s);
  }
  return softmax(scores);
}

// Random forest prediction: traverse each tree, accumulate class counts
function predictTree(treeNodes, xs) {
  let node = 0;
  while (true) {
    const n = treeNodes[node];
    const left = n.left, right = n.right;
    if (left === -1 && right === -1) { // leaf
      return n.value; // class counts
    }
    const f = n.feature;
    const th = n.threshold;
    if (xs[f] <= th) {
      node = left;
    } else {
      node = right;
    }
  }
}

function predictRF(rfModel, xs) {
  // rfModel.trees is an array of node lists. Each node has .value (counts)
  const nClasses = rfModel.n_classes || exportData.class_names.length;
  const agg = new Array(nClasses).fill(0);
  rfModel.trees.forEach(tree => {
    const counts = predictTree(tree, xs);
    for (let i = 0; i < counts.length; i++) agg[i] += counts[i];
  });
  // normalize
  const total = agg.reduce((a,b)=>a+b, 0);
  return agg.map(v => v / total);
}

// MLP inference (dense net with ReLU or logistic final)
function mlpPredict(mlpModel, xs_raw) {
  // xs_raw expected to be scaled features
  // mlpModel.coefs: [W0, W1,...], intercepts: [b0, b1,...]
  let a = xs_raw.slice(); // as array
  for (let layer = 0; layer < mlpModel.coefs.length; layer++) {
    const W = mlpModel.coefs[layer];      // W: rows = in_dim, cols = out_dim in sklearn's coefs: shape (in, out)
    const b = mlpModel.intercepts[layer]; // b: out_dim
    const out = new Array(b.length).fill(0);
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < a.length; i++) {
        s += a[i] * W[i][j];
      }
      out[j] = s;
    }
    // activation: all hidden layers relu; final layer linear (softmax applied outside)
    if (layer < mlpModel.coefs.length - 1) {
      for (let j = 0; j < out.length; j++) out[j] = Math.max(0, out[j]); // ReLU
    }
    a = out;
  }
  return softmax(a);
}

// Main predict function for a single model
function predictWithModel(modelName, xs_scaled) {
  const model = exportData.models[modelName];
  if (!model) return null;
  if (model.type === 'linear') {
    return scoreLinear(model.coef, model.intercept, xs_scaled);
  } else if (model.type === 'rf') {
    return predictRF(model, xs_scaled);
  } else if (model.type === 'mlp') {
    return mlpPredict(model, xs_scaled);
  } else {
    console.warn("Unknown model type", modelName, model.type);
    return null;
  }
}

function probsToText(probs) {
  const names = exportData.class_names;
  const lines = [];
  const maxIdx = probs.indexOf(Math.max(...probs));
  for (let i = 0; i < probs.length; i++) {
    const star = (i === maxIdx) ? "←" : "";
    lines.push(`${names[i]}: ${(probs[i]*100).toFixed(1)}% ${star}`);
  }
  return lines.join('\n');
}

// UI wiring & rendering
document.getElementById('predict').addEventListener('click', () => {
  if (!exportData) {
    alert("Models not loaded yet — wait a sec.");
    return;
  }
  const sl = document.getElementById('sepal_length').value;
  const sw = document.getElementById('sepal_width').value;
  const pl = document.getElementById('petal_length').value;
  const pw = document.getElementById('petal_width').value;
  const raw = [sl, sw, pl, pw];
  const xs = scaleFeatures(raw);

  // models to show based on checkboxes
  const showLog = document.getElementById('show_logistic').checked;
  const showRF  = document.getElementById('show_rf').checked;
  const showSVM = document.getElementById('show_svm').checked;
  const showMLP = document.getElementById('show_mlp').checked;
  const doEnsemble = document.getElementById('ensemble').checked;

  const outputsDiv = document.getElementById('models-output');
  outputsDiv.innerHTML = '';

  // collect probs for ensemble
  const probList = [];
  const modelOrder = [];
  if (showLog) { modelOrder.push('logistic'); }
  if (showRF)  { modelOrder.push('random_forest'); }
  if (showSVM) { modelOrder.push('svm'); }
  if (showMLP) { modelOrder.push('mlp'); }

  modelOrder.forEach(name => {
    const probs = predictWithModel(name, xs);
    if (!probs) return;
    probList.push(probs);
    const idx = probs.indexOf(Math.max(...probs));
    const el = document.createElement('div');
    el.className = 'model-block';
    el.innerHTML = `<strong>${name}</strong> → <em>${exportData.class_names[idx]}</em>
      <pre class="probs">${probs.map(p => (p*100).toFixed(1)+'%').join('  ')}</pre>`;
    outputsDiv.appendChild(el);
  });

  // ensemble
  const ensembleDiv = document.getElementById('ensemble-output');
  const ensembleTitle = document.getElementById('ensemble-title');
  if (doEnsemble && probList.length > 0) {
    // average probs element-wise
    const n = probList.length;
    const summed = new Array(probList[0].length).fill(0);
    probList.forEach(p => {
      for (let i = 0; i < p.length; i++) summed[i] += p[i];
    });
    const avg = summed.map(v => v / n);
    const idx = avg.indexOf(Math.max(...avg));
    ensembleTitle.style.display = 'block';
    ensembleDiv.style.display = 'block';
    ensembleDiv.innerHTML = `<strong>Ensemble (avg)</strong> → <em>${exportData.class_names[idx]}</em>
      <pre class="probs">${avg.map(p => (p*100).toFixed(1)+'%').join('  ')}</pre>`;
    // set reference image for ensemble winner
    setReferenceImage(idx);
  } else {
    ensembleTitle.style.display = 'none';
    ensembleDiv.style.display = 'none';
    // if not ensemble, show reference image from first shown model's prediction (fallback)
    if (probList.length > 0) {
      const first = probList[0];
      const idx = first.indexOf(Math.max(...first));
      setReferenceImage(idx);
    }
  }

  document.getElementById('results-multi').hidden = false;
  document.getElementById('result').hidden = false;
});

function setReferenceImage(classIndex) {
  const imageNames = ['images/setosa.jpg','images/versicolor.jpg','images/virginica.jpg'];
  const img = document.getElementById('pred-image');
  img.src = imageNames[classIndex];
  img.alt = exportData.class_names[classIndex];
}

document.getElementById('clear').addEventListener('click', () => {
  document.getElementById('sepal_length').value = '';
  document.getElementById('sepal_width').value = '';
  document.getElementById('petal_length').value = '';
  document.getElementById('petal_width').value = '';
  document.getElementById('models-output').innerHTML = '';
  document.getElementById('results-multi').hidden = true;
  document.getElementById('result').hidden = true;
});

// presets (re-use same presets)
const PRESETS = {
  setosa:       [5.1, 3.5, 1.4, 0.2],
  versicolor:   [6.0, 2.9, 4.2, 1.3],
  virginica:    [6.5, 3.0, 5.5, 2.0]
};
document.querySelectorAll('.preset').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const name = e.currentTarget.dataset.name;
    const p = PRESETS[name];
    document.getElementById('sepal_length').value = p[0];
    document.getElementById('sepal_width').value  = p[1];
    document.getElementById('petal_length').value = p[2];
    document.getElementById('petal_width').value  = p[3];
    document.getElementById('predict').click();
  });
});
