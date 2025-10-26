// app.js - loads model_params.json and performs client-side prediction
let model = null;

async function loadModel() {
  const r = await fetch('model_params.json');
  model = await r.json();
  console.log("Loaded model:", model.class_names);
}
loadModel();

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b, 0);
  return exps.map(e => e / sum);
}

function predictFromFeatures(features) {
  // features: array length 4 (raw input)
  // apply scaler: (x - mean) / scale
  const mean = model.scaler_mean;
  const scale = model.scaler_scale;
  const xs = features.map((v, i) => (Number(v) - mean[i]) / scale[i]);

  // linear scores = coef dot xs + intercept
  const coefs = model.coef; // shape [3][4]
  const intercept = model.intercept; // length 3

  const scores = coefs.map((row, c) => {
    let s = intercept[c];
    for (let i = 0; i < row.length; i++) s += row[i] * xs[i];
    return s;
  });

  const probs = softmax(scores);
  const best = probs.indexOf(Math.max(...probs));
  return {
    classIndex: best,
    className: model.class_names[best],
    probs: probs
  };
}

// Example presets (typical values from iris data)
const PRESETS = {
  setosa:       [5.1, 3.5, 1.4, 0.2],
  versicolor:   [6.0, 2.9, 4.2, 1.3],
  virginica:    [6.5, 3.0, 5.5, 2.0]
};

// Helper to fill inputs and (optionally) auto-predict
function applyPreset(name, autoPredict = true) {
  const p = PRESETS[name];
  if (!p) return;
  document.getElementById('sepal_length').value = p[0];
  document.getElementById('sepal_width').value  = p[1];
  document.getElementById('petal_length').value = p[2];
  document.getElementById('petal_width').value  = p[3];
  if (autoPredict) document.getElementById('predict').click();
}

// UI wiring
document.getElementById('predict').addEventListener('click', () => {
  if (!model) {
    alert("Model not loaded yet. Wait a second and retry.");
    return;
  }
  const sl = document.getElementById('sepal_length').value;
  const sw = document.getElementById('sepal_width').value;
  const pl = document.getElementById('petal_length').value;
  const pw = document.getElementById('petal_width').value;

  const features = [sl, sw, pl, pw];
  const out = predictFromFeatures(features);

  document.getElementById('prediction-text').textContent =
    `${out.className} (class index ${out.classIndex})`;

  // show probabilities
  const probsDiv = document.getElementById('prediction-probs');
  probsDiv.innerHTML = '';
  for (let i = 0; i < model.class_names.length; i++) {
    const p = document.createElement('div');
    p.textContent = `${model.class_names[i]}: ${(out.probs[i]*100).toFixed(1)}%`;
    probsDiv.appendChild(p);
  }

  // image mapping - expects images in docs/images/
  const imageNames = ['images/setosa.jpg', 'images/versicolor.jpg', 'images/virginica.jpg'];
  const img = document.getElementById('pred-image');
  img.src = imageNames[out.classIndex];
  img.alt = model.class_names[out.classIndex];

  document.getElementById('result').hidden = false;
});

document.getElementById('clear').addEventListener('click', () => {
  document.getElementById('sepal_length').value = '';
  document.getElementById('sepal_width').value = '';
  document.getElementById('petal_length').value = '';
  document.getElementById('petal_width').value = '';
  document.getElementById('result').hidden = true;
});

// wire preset buttons
document.querySelectorAll('.preset').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const name = e.currentTarget.dataset.name;
    applyPreset(name, true);
  });
});
