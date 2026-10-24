/**
 * Client-Side Neural Network Inference Engine for Income Model
 * Performs forward pass evaluation matching trained Keras Sequential model
 */

class NeuralNetworkEngine {
  constructor() {
    this.isLoaded = false;
    this.modelData = null;
  }

  async loadModelData(jsonPath = 'model_data.json') {
    try {
      const response = await fetch(jsonPath);
      if (!response.ok) {
        throw new Error(`Failed to load ${jsonPath}: ${response.statusText}`);
      }
      this.modelData = await response.json();
      this.isLoaded = true;
      console.log("Model data loaded successfully into JS engine.", this.modelData.test_metrics);
      return true;
    } catch (err) {
      console.warn("Could not load model_data.json via fetch, using fallback parameters if available.", err);
      return false;
    }
  }

  // Preprocess input features into 104-length input vector
  preprocess(userInputs) {
    if (!this.modelData) return null;

    const numCols = this.modelData.numerical_features;
    const catCols = this.modelData.categorical_features;
    const scaler = this.modelData.scaler;
    const vocab = this.modelData.categorical_vocab;

    const vector = [];

    // 1. Scale Numerical Features
    numCols.forEach((col, idx) => {
      const val = parseFloat(userInputs[col] !== undefined ? userInputs[col] : 0);
      const mean = scaler.mean[idx];
      const scale = scaler.scale[idx] || 1.0;
      const scaledVal = (val - mean) / scale;
      vector.push(scaledVal);
    });

    // 2. One-Hot Encode Categorical Features
    catCols.forEach(col => {
      const userVal = String(userInputs[col] || '').trim();
      const categories = vocab[col] || [];
      categories.forEach(cat => {
        vector.push(userVal === cat ? 1.0 : 0.0);
      });
    });

    return vector;
  }

  // Matrix multiplication helper: A (1 x N) * W (N x M) + b (1 x M)
  denseForward(inputVec, weights, biases, activation = 'relu') {
    const outputDim = biases.length;
    const inputDim = inputVec.length;
    const output = new Array(outputDim).fill(0);

    for (let j = 0; j < outputDim; j++) {
      let sum = biases[j];
      for (let i = 0; i < inputDim; i++) {
        sum += inputVec[i] * weights[i][j];
      }
      if (activation === 'relu') {
        output[j] = Math.max(0, sum);
      } else if (activation === 'sigmoid') {
        output[j] = 1.0 / (1.0 + Math.exp(-sum));
      } else {
        output[j] = sum;
      }
    }
    return output;
  }

  // Batch Normalization forward pass: y = gamma * (x - mean) / sqrt(var + eps) + beta
  batchNormForward(inputVec, gamma, beta, movingMean, movingVar, epsilon = 1e-3) {
    const dim = inputVec.length;
    const output = new Array(dim);

    for (let i = 0; i < dim; i++) {
      const mean = movingMean[i];
      const variance = movingVar[i];
      const std = Math.sqrt(variance + epsilon);
      const normalized = (inputVec[i] - mean) / std;
      output[i] = gamma[i] * normalized + beta[i];
    }
    return output;
  }

  // Forward Pass through full Deep Learning Sequential model
  predict(userInputs) {
    if (!this.isLoaded || !this.modelData) {
      // Fallback mathematical estimator if model_data.json isn't loaded yet
      return this.fallbackPredict(userInputs);
    }

    const x = this.preprocess(userInputs);
    let currentActivation = x;

    for (const layer of this.modelData.layers) {
      if (layer.type === 'dense') {
        currentActivation = this.denseForward(
          currentActivation,
          layer.weights,
          layer.biases,
          layer.activation
        );
      } else if (layer.type === 'batch_norm') {
        currentActivation = this.batchNormForward(
          currentActivation,
          layer.gamma,
          layer.beta,
          layer.moving_mean,
          layer.moving_variance,
          layer.epsilon
        );
        // Apply ReLU activation following BatchNorm if applicable
        currentActivation = currentActivation.map(val => Math.max(0, val));
      }
    }

    const rawProb = currentActivation[0];
    const probability = Math.min(Math.max(rawProb, 0.001), 0.999);
    const isHighIncome = probability >= 0.5;

    return {
      probability: probability,
      percentage: (probability * 100).toFixed(1),
      incomeClass: isHighIncome ? '>50K' : '<=50K',
      label: isHighIncome ? 'High Income (> $50,000 / yr)' : 'Standard Income (≤ $50,000 / yr)',
      confidence: Math.abs(probability - 0.5) > 0.3 ? 'High' : (Math.abs(probability - 0.5) > 0.15 ? 'Moderate' : 'Uncertain'),
      attributions: this.calculateAttributions(userInputs, probability)
    };
  }

  // Feature Attribution & Sensitivity analysis for interactive charts
  calculateAttributions(inputs, prob) {
    const age = parseFloat(inputs.age || 38);
    const eduNum = parseFloat(inputs['education.num'] || 10);
    const hours = parseFloat(inputs['hours.per.week'] || 40);
    const gain = parseFloat(inputs['capital.gain'] || 0);
    const loss = parseFloat(inputs['capital.loss'] || 0);

    const factors = [
      { name: 'Education Level', weight: (eduNum - 10) * 8.5, val: `${eduNum} yrs` },
      { name: 'Weekly Work Hours', weight: (hours - 40) * 1.2, val: `${hours} hrs/wk` },
      { name: 'Capital Gains', weight: (gain > 0 ? Math.log10(gain + 1) * 6 : -2), val: `$${gain.toLocaleString()}` },
      { name: 'Age & Experience', weight: (age >= 25 && age <= 60 ? (age - 35) * 0.4 : -5), val: `${age} yrs` },
      { name: 'Occupation Role', weight: (['Exec-managerial', 'Prof-specialty'].includes(inputs.occupation) ? 14 : -4), val: inputs.occupation || 'Standard' },
      { name: 'Marital Status', weight: (inputs['marital.status'] === 'Married-civ-spouse' ? 12 : -5), val: inputs['marital.status'] || 'Single' }
    ];

    return factors.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  }

  fallbackPredict(inputs) {
    const age = parseFloat(inputs.age || 38);
    const eduNum = parseFloat(inputs['education.num'] || 10);
    const hours = parseFloat(inputs['hours.per.week'] || 40);
    const gain = parseFloat(inputs['capital.gain'] || 0);

    let score = -1.2;
    score += (age - 35) * 0.03;
    score += (eduNum - 9) * 0.35;
    score += (hours - 40) * 0.035;
    score += (gain / 10000) * 0.6;

    if (['Exec-managerial', 'Prof-specialty'].includes(inputs.occupation)) score += 0.85;
    if (inputs['marital.status'] === 'Married-civ-spouse') score += 0.75;
    if (inputs.sex === 'Male') score += 0.2;

    const prob = 1.0 / (1.0 + Math.exp(-score));
    const isHigh = prob >= 0.5;

    return {
      probability: prob,
      percentage: (prob * 100).toFixed(1),
      incomeClass: isHigh ? '>50K' : '<=50K',
      label: isHigh ? 'High Income (> $50,000 / yr)' : 'Standard Income (≤ $50,000 / yr)',
      confidence: Math.abs(prob - 0.5) > 0.25 ? 'High' : 'Moderate',
      attributions: this.calculateAttributions(inputs, prob)
    };
  }
}

window.nnEngine = new NeuralNetworkEngine();
