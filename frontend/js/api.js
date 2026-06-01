/** API client for Project Glitch backend */
const BASE = '';

const api = {
  async get(url) {
    const r = await fetch(BASE + url);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(BASE + url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
  async patch(url, body) {
    const r = await fetch(BASE + url, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
  async del(url) {
    await fetch(BASE + url, { method: 'DELETE' });
  },
};

// Chat endpoints
api.chat = {
  getSessions: () => api.get('/api/chat/sessions'),
  getSession: (id) => api.get('/api/chat/sessions/' + id),
  createSession: (title, prompt) => api.post('/api/chat/sessions', { title, system_prompt: prompt }),
  renameSession: (id, title) => api.patch('/api/chat/sessions/rename', { session_id: id, title }),
  deleteSession: (id) => api.del('/api/chat/sessions/' + id),
  stream: (session_id, message, temp, maxTok) =>
    fetch(BASE + '/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id, message, temperature: temp, max_tokens: maxTok }),
    }),
};

// System endpoints
api.system = {
  stats: () => api.get('/api/system/stats'),
  inference: () => api.get('/api/system/inference'),
};

// Models endpoints
api.models = {
  list: () => api.get('/api/models/'),
  load: (name, ctx) => api.post('/api/models/load', { model_name: name, context_length: ctx }),
  settings: () => api.get('/api/models/settings'),
  updateSettings: (s) => api.patch('/api/models/settings', s),
};

// Training endpoints
api.training = {
  start: (cfg) => api.post('/api/training/start', cfg),
  status: (id) => api.get('/api/training/status/' + id),
  list: () => api.get('/api/training/jobs'),
  stop: (id) => api.post('/api/training/stop/' + id),
};

// Documents endpoints
api.docs = {
  list: () => api.get('/api/documents/'),
  upload: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return fetch(BASE + '/api/documents/upload', { method: 'POST', body: fd }).then(r => r.json());
  },
  delete: (name) => api.del('/api/documents/' + name),
};
