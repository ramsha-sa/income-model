/**
 * Interactive Application Controller for Income Model Dashboard
 * Handles input events, live predictions, preset loading, and Chart.js visualizations
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize Neural Network Engine
  await window.nnEngine.loadModelData('model_data.json');

  // DOM Input References
  const ageInput = document.getElementById('ageInput');
  const ageVal = document.getElementById('ageVal');
  const sexInput = document.getElementById('sexInput');
  const raceInput = document.getElementById('raceInput');
  const countryInput = document.getElementById('countryInput');

  const educationInput = document.getElementById('educationInput');
  const eduNumVal = document.getElementById('eduNumVal');
  const occupationInput = document.getElementById('occupationInput');
  const workclassInput = document.getElementById('workclassInput');
  const maritalInput = document.getElementById('maritalInput');
  const hoursInput = document.getElementById('hoursInput');
  const hoursVal = document.getElementById('hoursVal');

  const gainInput = document.getElementById('gainInput');
  const lossInput = document.getElementById('lossInput');

  const btnReset = document.getElementById('btnReset');
  const presetChips = document.querySelectorAll('.chip-btn');

  // Chart Instances
  let radarChart, scenarioChart, waterfallChart, eduBarChart, gaugeChart;

  // Preset Data Configuration
  const presets = {
    executive: {
      age: 45, sex: 'Male', race: 'White', country: 'United-States',
      education: 'Bachelors', occupation: 'Exec-managerial', workclass: 'Private',
      marital: 'Married-civ-spouse', hours: 48, gain: 5000, loss: 0
    },
    tech: {
      age: 32, sex: 'Male', race: 'Asian-Pac-Islander', country: 'United-States',
      education: 'Masters', occupation: 'Prof-specialty', workclass: 'Private',
      marital: 'Never-married', hours: 40, gain: 2500, loss: 0
    },
    entrepreneur: {
      age: 40, sex: 'Male', race: 'White', country: 'United-States',
      education: 'Bachelors', occupation: 'Exec-managerial', workclass: 'Self-emp-inc',
      marital: 'Married-civ-spouse', hours: 55, gain: 15024, loss: 0
    },
    student: {
      age: 22, sex: 'Female', race: 'White', country: 'United-States',
      education: 'Some-college', occupation: 'Adm-clerical', workclass: 'Private',
      marital: 'Never-married', hours: 25, gain: 0, loss: 0
    }
  };

  // Helper to extract form data
  function getUserInputs() {
    const selectedEduOption = educationInput.options[educationInput.selectedIndex];
    const eduNum = parseInt(selectedEduOption.getAttribute('data-num') || 10);

    return {
      age: parseInt(ageInput.value),
      fnlwgt: 189778, // median weight
      'education.num': eduNum,
      'capital.gain': parseFloat(gainInput.value || 0),
      'capital.loss': parseFloat(lossInput.value || 0),
      'hours.per.week': parseInt(hoursInput.value),
      workclass: workclassInput.value,
      education: educationInput.value,
      'marital.status': maritalInput.value,
      occupation: occupationInput.value,
      relationship: maritalInput.value === 'Married-civ-spouse' ? 'Husband' : 'Not-in-family',
      race: raceInput.value,
      sex: sexInput.value,
      'native.country': countryInput.value
    };
  }

  // Update UI Text Display Labels
  function updateInputLabels() {
    ageVal.textContent = `${ageInput.value} yrs`;
    hoursVal.textContent = `${hoursInput.value} hrs/wk`;

    const selectedEduOption = educationInput.options[educationInput.selectedIndex];
    const eduNum = selectedEduOption.getAttribute('data-num') || 10;
    eduNumVal.textContent = `Grade: ${eduNum} (${educationInput.value})`;
  }

  // Initialize Chart.js Visualizations
  function initCharts() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // 1. Donut Gauge Chart
    const ctxGauge = document.getElementById('gaugeCanvas').getContext('2d');
    gaugeChart = new Chart(ctxGauge, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [78.5, 21.5],
          backgroundColor: ['#06b6d4', 'rgba(255, 255, 255, 0.08)'],
          borderWidth: 0,
          borderRadius: 10
        }]
      },
      options: {
        cutout: '80%',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } }
      }
    });

    // 2. Radar Chart (User Profile vs High-Income Benchmark)
    const ctxRadar = document.getElementById('radarChart').getContext('2d');
    radarChart = new Chart(ctxRadar, {
      type: 'radar',
      data: {
        labels: ['Education', 'Work Hours', 'Cap Gains', 'Experience', 'Role Seniority'],
        datasets: [
          {
            label: 'Current Profile',
            data: [75, 80, 20, 65, 85],
            backgroundColor: 'rgba(6, 182, 212, 0.25)',
            borderColor: '#06b6d4',
            pointBackgroundColor: '#06b6d4'
          },
          {
            label: 'High Earner Avg (> $50K)',
            data: [85, 85, 60, 80, 85],
            backgroundColor: 'rgba(139, 92, 246, 0.15)',
            borderColor: '#8b5cf6',
            borderDash: [4, 4],
            pointBackgroundColor: '#8b5cf6'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            pointLabels: { color: '#94a3b8', font: { size: 11, weight: '600' } },
            ticks: { display: false, max: 100, min: 0 }
          }
        },
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } }
        }
      }
    });

    // 3. What-If Scenario Optimizer Bar Chart
    const ctxScenario = document.getElementById('scenarioChart').getContext('2d');
    scenarioChart = new Chart(ctxScenario, {
      type: 'bar',
      data: {
        labels: ['Current Profile', 'Optimized (+10h/Edu)', 'Average High Earner'],
        datasets: [{
          label: 'High Income Probability (%)',
          data: [78.5, 92.0, 85.0],
          backgroundColor: ['#06b6d4', '#10b981', '#8b5cf6'],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { callback: v => v + '%' }
          },
          x: { grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });

    // 4. Feature Contribution Waterfall Bar Chart
    const ctxWaterfall = document.getElementById('waterfallChart').getContext('2d');
    waterfallChart = new Chart(ctxWaterfall, {
      type: 'bar',
      data: {
        labels: ['Role', 'Education', 'Marital', 'Hours', 'Gains'],
        datasets: [{
          label: 'Impact Score',
          data: [14, 12, 10, 4, -2],
          backgroundColor: ctx => ctx.raw >= 0 ? '#10b981' : '#f43f5e',
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.06)' } },
          y: { grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });

    // 5. Census Income Probability by Education Level
    const ctxEdu = document.getElementById('eduBarChart').getContext('2d');
    eduBarChart = new Chart(ctxEdu, {
      type: 'bar',
      data: {
        labels: ['HS Grad', 'Some College', 'Bachelors', 'Masters', 'Doctorate'],
        datasets: [{
          label: '% Earning > $50K',
          data: [15.9, 19.0, 41.5, 55.6, 73.8],
          backgroundColor: 'rgba(6, 182, 212, 0.4)',
          borderColor: '#06b6d4',
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { callback: v => v + '%' }
          },
          x: { grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // Primary Update Method - Triggered on any input change
  function updatePrediction() {
    updateInputLabels();
    const inputs = getUserInputs();
    const result = window.nnEngine.predict(inputs);

    // Update Hero UI Elements
    document.getElementById('gaugePercentage').textContent = `${result.percentage}%`;
    document.getElementById('predictionTitle').textContent = `${result.incomeClass} Income Probability`;
    document.getElementById('predictionDesc').textContent = result.label;

    const tagEl = document.getElementById('predictionTag');
    if (result.incomeClass === '>50K') {
      tagEl.className = 'prediction-tag tag-high';
      tagEl.innerHTML = `<span>●</span> Predicted Class: > $50,000 / yr (${result.percentage}%)`;
    } else {
      tagEl.className = 'prediction-tag tag-standard';
      tagEl.innerHTML = `<span>●</span> Predicted Class: ≤ $50,000 / yr (${result.percentage}%)`;
    }

    document.getElementById('confidenceText').textContent = `${result.confidence} Confidence`;
    document.getElementById('confidenceFill').style.width = `${Math.min(100, Math.max(20, result.probability * 100))}%`;

    // 1. Update Gauge Chart
    if (gaugeChart) {
      gaugeChart.data.datasets[0].data = [result.probability * 100, (1.0 - result.probability) * 100];
      gaugeChart.data.datasets[0].backgroundColor = [
        result.incomeClass === '>50K' ? '#06b6d4' : '#f59e0b',
        'rgba(255, 255, 255, 0.08)'
      ];
      gaugeChart.update('none');
    }

    // 2. Update Radar Chart (Current Profile Normalize)
    if (radarChart) {
      const eduNorm = Math.min(100, (inputs['education.num'] / 16) * 100);
      const hoursNorm = Math.min(100, (inputs['hours.per.week'] / 60) * 100);
      const gainNorm = Math.min(100, (inputs['capital.gain'] / 15000) * 100);
      const ageNorm = Math.min(100, (inputs.age / 65) * 100);
      const roleNorm = ['Exec-managerial', 'Prof-specialty'].includes(inputs.occupation) ? 90 : 45;

      radarChart.data.datasets[0].data = [eduNorm, hoursNorm, gainNorm, ageNorm, roleNorm];
      radarChart.update('none');
    }

    // 3. Update What-If Scenario Chart
    if (scenarioChart) {
      const optInputs = Object.assign({}, inputs);
      optInputs['hours.per.week'] = Math.min(60, inputs['hours.per.week'] + 10);
      optInputs['education.num'] = Math.min(16, inputs['education.num'] + 2);
      const optResult = window.nnEngine.predict(optInputs);

      scenarioChart.data.datasets[0].data = [
        parseFloat(result.percentage),
        parseFloat(optResult.percentage),
        85.0
      ];
      scenarioChart.update('none');
    }

    // 4. Update Waterfall Chart
    if (waterfallChart && result.attributions) {
      waterfallChart.data.labels = result.attributions.map(a => a.name);
      waterfallChart.data.datasets[0].data = result.attributions.map(a => a.weight);
      waterfallChart.update('none');
    }
  }

  // Preset Load Helper
  function applyPreset(presetKey) {
    const p = presets[presetKey];
    if (!p) return;

    ageInput.value = p.age;
    sexInput.value = p.sex;
    raceInput.value = p.race;
    countryInput.value = p.country;
    educationInput.value = p.education;
    occupationInput.value = p.occupation;
    workclassInput.value = p.workclass;
    maritalInput.value = p.marital;
    hoursInput.value = p.hours;
    gainInput.value = p.gain;
    lossInput.value = p.loss;

    presetChips.forEach(c => c.classList.remove('active'));
    document.querySelector(`[data-preset="${presetKey}"]`)?.classList.add('active');

    updatePrediction();
  }

  // Attach Event Listeners
  const allInputs = [
    ageInput, sexInput, raceInput, countryInput, educationInput,
    occupationInput, workclassInput, maritalInput, hoursInput,
    gainInput, lossInput
  ];

  allInputs.forEach(input => {
    input.addEventListener('input', updatePrediction);
    input.addEventListener('change', updatePrediction);
  });

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetKey = chip.getAttribute('data-preset');
      applyPreset(presetKey);
    });
  });

  btnReset.addEventListener('click', () => {
    applyPreset('executive');
  });

  // Initial Run
  initCharts();
  updatePrediction();
});
