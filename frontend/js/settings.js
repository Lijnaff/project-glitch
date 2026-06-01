/** Settings, models, training, and documents page logic */

// ============ MODELS ============
async function renderModelList() {
  try {
    const models = await api.models.list();
    const modelsPage = document.getElementById('model-list');
    const trainSelect = document.getElementById('train-model');

    if (!models.length) {
      modelsPage.innerHTML = '<p class="empty-text">No models found. Place .gguf files in data/models/</p>';
    } else {
      modelsPage.innerHTML = models.map(function(m) {
        return '<div class="model-item">' +
          '<div class="model-info">' +
            '<div class="model-name">' + escHtml(m.name) + '</div>' +
            '<div class="model-meta"><span>' + m.size_gb + ' GB</span><span>' + (m.quantization || 'unknown') + '</span><span>ctx: ' + m.context_length + '</span></div>' +
          '</div>' +
          '<div>' +
            '<span class="model-badge ' + (m.is_loaded ? 'loaded' : 'available') + '">' + (m.is_loaded ? 'Loaded' : 'Available') + '</span> ' +
            (!m.is_loaded ? '<button class="btn btn-outline" onclick="loadModel(\'' + escHtml(m.name).replace(/'/g, "\\'") + '\')">Load</button>' : '') +
          '</div>' +
        '</div>';
      }).join('');
    }

    if (trainSelect) {
      trainSelect.innerHTML = models.length
        ? models.map(function(m) { return '<option>' + escHtml(m.name) + '</option>'; }).join('')
        : '<option>No models available</option>';
    }
  } catch(e) { console.warn('Models error:', e); }
}

async function loadModel(name) {
  await api.models.load(name);
  renderModelList();
}

// ============ SETTINGS ============
document.getElementById('setting-temp').addEventListener('input', function() {
  document.getElementById('temp-val').textContent = this.value;
});
document.getElementById('setting-maxtok').addEventListener('input', function() {
  document.getElementById('maxtok-val').textContent = this.value;
});
document.getElementById('setting-ctx').addEventListener('input', function() {
  document.getElementById('ctx-val').textContent = this.value;
});

document.getElementById('save-settings-btn').addEventListener('click', async function() {
  const s = {
    temperature: parseFloat(document.getElementById('setting-temp').value),
    max_tokens: parseInt(document.getElementById('setting-maxtok').value),
    context_length: parseInt(document.getElementById('setting-ctx').value),
    system_prompt: document.getElementById('setting-sysprompt').value,
  };
  await api.models.updateSettings(s);
  settings.temperature = s.temperature;
  settings.max_tokens = s.max_tokens;
  alert('Settings saved!');
});

document.getElementById('test-connection-btn').addEventListener('click', async function() {
  const url = document.getElementById('setting-llm-url').value;
  const result = document.getElementById('connection-result');
  result.innerHTML = 'Testing...';
  try {
    const r = await fetch(url.replace(/\/completion$/, '/health') || url);
    const d = await r.json();
    result.innerHTML = '<span class="conn-dot ok"></span>OK — ' + JSON.stringify(d);
  } catch (e) {
    result.innerHTML = '<span class="conn-dot bad"></span>Failed — ' + e.message;
  }
});

// ============ TRAINING ============
document.getElementById('start-training-btn').addEventListener('click', async function() {
  const cfg = {
    base_model: document.getElementById('train-model').value,
    dataset_path: document.getElementById('train-dataset').value,
    epochs: parseInt(document.getElementById('train-epochs').value),
    learning_rate: parseFloat(document.getElementById('train-lr').value),
    batch_size: parseInt(document.getElementById('train-batch').value),
    lora_rank: parseInt(document.getElementById('train-lora').value),
  };
  try {
    const job = await api.training.start(cfg);
    alert('Training job ' + job.job_id + ' started!');
    renderTrainingJobs();
  } catch(e) {
    alert('Error starting training: ' + e.message);
  }
});

async function renderTrainingJobs() {
  try {
    const jobs = await api.training.list();
    const list = document.getElementById('training-jobs-list');
    if (!jobs.length) {
      list.innerHTML = '<p class="empty-text">No training jobs yet</p>';
      return;
    }
    list.innerHTML = jobs.map(function(j) {
      var cls = j.status === 'completed' ? 'green' : j.status === 'failed' ? 'red' : '';
      return '<div class="doc-item">' +
        '<div><div class="doc-name">Job ' + j.job_id + '</div>' +
        '<div class="doc-meta">' + j.status + ' &middot; Epoch ' + j.current_epoch + '/' + j.total_epochs + ' &middot; Loss: ' + (j.loss || '—') + '</div></div>' +
        '<div><div class="progress-bar" style="width:120px;display:inline-block"><div class="progress-bar-fill ' + cls + '" style="width:' + j.progress_pct + '%"></div></div>' +
        '<span style="font-size:0.8rem;margin-left:8px">' + Math.round(j.progress_pct) + '%</span></div>' +
      '</div>';
    }).join('');
  } catch(e) { console.warn('Training jobs error:', e); }
}

// ============ DOCUMENTS ============
async function renderDocs() {
  try {
    const docs = await api.docs.list();
    const list = document.getElementById('doc-list');
    if (!docs.length) {
      list.innerHTML = '<p class="empty-text">No documents uploaded yet</p>';
      return;
    }
    list.innerHTML = docs.map(function(d) {
      return '<div class="doc-item">' +
        '<div><div class="doc-name">📄 ' + escHtml(d.name) + '</div>' +
        '<div class="doc-meta">' + d.size_kb + ' KB &middot; ' + d.status + ' &middot; ' + d.chunks + ' chunks</div></div>' +
        '<button class="btn-icon btn-danger" onclick="deleteDoc(\'' + escHtml(d.name).replace(/'/g, "\\'") + '\')" title="Delete">🗑️</button>' +
      '</div>';
    }).join('');
  } catch(e) { console.warn('Docs error:', e); }
}

async function deleteDoc(name) {
  if (!confirm('Delete ' + name + '?')) return;
  await api.docs.delete(name);
  renderDocs();
}

// Document upload
var docDropzone = document.getElementById('doc-dropzone');
var docInput = document.getElementById('doc-file-input');

docDropzone.addEventListener('click', function() { docInput.click(); });
docDropzone.addEventListener('dragover', function(e) { e.preventDefault(); this.style.borderColor = 'var(--accent-cyan)'; });
docDropzone.addEventListener('dragleave', function() { this.style.borderColor = ''; });
docDropzone.addEventListener('drop', async function(e) {
  e.preventDefault();
  this.style.borderColor = '';
  for (var i = 0; i < e.dataTransfer.files.length; i++) {
    await api.docs.upload(e.dataTransfer.files[i]);
  }
  renderDocs();
});
docInput.addEventListener('change', async function() {
  for (var i = 0; i < this.files.length; i++) {
    await api.docs.upload(this.files[i]);
  }
  renderDocs();
  this.value = '';
});

function escHtml(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ============ BACKEND SWITCHER ============
document.getElementById('setting-backend').addEventListener('change', function() {
  var val = this.value;
  document.getElementById('hermes-settings').style.display = val === 'hermes' ? 'block' : 'none';
  document.getElementById('openrouter-settings').style.display = val === 'openrouter' ? 'block' : 'none';
  document.getElementById('llamacpp-settings').style.display = val === 'llama_cpp' ? 'block' : 'none';
});

document.getElementById('save-backend-btn').addEventListener('click', async function() {
  var backend = document.getElementById('setting-backend').value;

  // Save backend selection via API
  await api.post('/api/models/backend', { backend: backend });

  // Save OpenRouter settings to .env on server (via a simple config endpoint)
  if (backend === 'openrouter') {
    var key = document.getElementById('setting-or-key').value;
    var model = document.getElementById('setting-or-model').value;
    if (key) {
      await api.post('/api/models/config', {
        openrouter_key: key,
        openrouter_model: model,
      });
    }
  }

  // Reload the runtime backend
  try {
    var r = await fetch('/api/models/backend');
    var d = await r.json();
    alert('Backend switched to: ' + d.backend);
  } catch(e) {
    alert('Backend saved. Restart the server for changes to take full effect.');
  }
});

// Load current backend on page open
fetch('/api/models/backend').then(function(r) { return r.json(); }).then(function(d) {
  var sel = document.getElementById('setting-backend');
  if (sel && d.backend) {
    sel.value = d.backend;
    sel.dispatchEvent(new Event('change'));
  }
}).catch(function() {});

// Hermes test message
document.getElementById('test-hermes-btn').addEventListener('click', async function() {
  var result = document.getElementById('hermes-test-result');
  result.innerHTML = 'Sending test message to Hermes...';
  try {
    // Submit a test message to the inbox via the chat API
    var resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: 'hermes-test',
        message: 'Hello! This is a test message from the Project Glitch dashboard. Please respond to confirm the bridge is working.',
        temperature: 0.7,
        max_tokens: 512,
      }),
    });

    if (resp.ok) {
      result.innerHTML = '<span class="conn-dot ok"></span>Message sent! Hermes should respond within 1 minute. Check the Chat page for the response.';
    } else {
      var err = await resp.text();
      result.innerHTML = '<span class="conn-dot bad"></span>Error: ' + err;
    }
  } catch(e) {
    result.innerHTML = '<span class="conn-dot bad"></span>Connection failed: ' + e.message;
  }
});

