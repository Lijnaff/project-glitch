/** Main app — navigation, init */

// Page navigation
document.querySelectorAll('.nav-item').forEach(function(item) {
  item.addEventListener('click', function() {
    var page = this.dataset.page;
    // Update nav
    document.querySelectorAll('.nav-item').forEach(function(n) { n.classList.remove('active'); });
    this.classList.add('active');
    // Update page
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.getElementById('page-' + page).classList.add('active');

    // Lazy-load page content
    if (page === 'models') renderModelList();
    if (page === 'training') { renderModelList(); renderTrainingJobs(); }
    if (page === 'documents') renderDocs();
  });
});

// Init
loadSessions();

// Load settings
api.models.settings().then(function(s) {
  settings.temperature = s.temperature;
  settings.max_tokens = s.max_tokens;
  var tempEl = document.getElementById('setting-temp');
  if (tempEl) { tempEl.value = s.temperature; document.getElementById('temp-val').textContent = s.temperature; }
  var maxEl = document.getElementById('setting-maxtok');
  if (maxEl) { maxEl.value = s.max_tokens; document.getElementById('maxtok-val').textContent = s.max_tokens; }
  var ctxEl = document.getElementById('setting-ctx');
  if (ctxEl) { ctxEl.value = s.context_length; document.getElementById('ctx-val').textContent = s.context_length; }
  var sysEl = document.getElementById('setting-sysprompt');
  if (sysEl && s.system_prompt) sysEl.value = s.system_prompt;
}).catch(function(e) { console.warn('Settings load failed:', e); });
