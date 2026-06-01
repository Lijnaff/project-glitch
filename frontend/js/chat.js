/** Chat logic — sessions, messages, streaming */
let currentSession = null;
let settings = { temperature: 0.7, max_tokens: 2048 };

async function loadSessions() {
  try {
    const sessions = await api.chat.getSessions();
    const list = document.getElementById('sessions-list');
    if (!sessions.length) {
      list.innerHTML = '<p class="empty-text" style="padding:16px">No chats yet. Click + to start.</p>';
      return;
    }
    list.innerHTML = sessions.map(s =>
      '<div class="session-item ' + (currentSession === s.id ? 'active' : '') + '" data-id="' + s.id + '">' +
        '<div>' +
          '<div class="session-item-title">' + escHtml(s.title) + '</div>' +
          '<div class="session-item-meta">' + s.message_count + ' messages &middot; ' + formatDate(s.updated_at) + '</div>' +
        '</div>' +
      '</div>'
    ).join('');

    list.querySelectorAll('.session-item').forEach(function(el) {
      el.addEventListener('click', function() { selectSession(el.dataset.id); });
    });
  } catch(e) { console.warn('Sessions load error:', e); }
}

async function selectSession(id) {
  currentSession = id;
  try {
    const session = await api.chat.getSession(id);
    document.getElementById('chat-title').textContent = session.title;
    document.getElementById('delete-chat-btn').style.display = 'inline-flex';

    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';

    if (!session.messages || !session.messages.length) {
      msgs.innerHTML = '<div class="empty-state"><div class="empty-icon">💬</div><p>Start the conversation</p></div>';
    } else {
      session.messages.forEach(function(m) { appendMessage(m.role, m.content); });
    }
    msgs.scrollTop = msgs.scrollHeight;
    loadSessions();
  } catch(e) { console.warn('Session load error:', e); }
}

function appendMessage(role, content, streaming) {
  const msgs = document.getElementById('messages');
  const empty = msgs.querySelector('.empty-state');
  if (empty) empty.remove();

  const el = document.createElement('div');
  el.className = 'message ' + role;
  el.innerHTML =
    '<div class="message-avatar">' + (role === 'user' ? '👤' : '⚡') + '</div>' +
    '<div class="message-content' + (streaming ? ' streaming' : '') + '">' + escHtml(content) + '</div>';
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
  return el;
}

function updateStreamingMessage(el, content) {
  const mc = el.querySelector('.message-content');
  mc.textContent = content;
  mc.classList.add('streaming');
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function finalizeStreaming(el) {
  el.querySelector('.message-content').classList.remove('streaming');
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || !currentSession) return;

  input.value = '';
  input.style.height = 'auto';

  appendMessage('user', text);
  const el = appendMessage('assistant', '', true);
  document.getElementById('send-btn').disabled = true;

  try {
    const res = await api.chat.stream(currentSession, text, settings.temperature, settings.max_tokens);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          const data = line.slice(6);
          full += data;
          updateStreamingMessage(el, full);
        }
      }
    }
    finalizeStreaming(el);
  } catch (e) {
    updateStreamingMessage(el, '[Error: ' + e.message + ']');
    finalizeStreaming(el);
  }

  document.getElementById('send-btn').disabled = false;
  loadSessions();
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function formatDate(d) {
  try { return new Date(d).toLocaleDateString(); } catch(e) { return d; }
}

// New chat
document.getElementById('new-chat-btn').addEventListener('click', async function() {
  const title = prompt('Chat title:', 'New Chat') || 'New Chat';
  const s = await api.chat.createSession(title);
  await loadSessions();
  selectSession(s.id);
});

// Delete chat
document.getElementById('delete-chat-btn').addEventListener('click', async function() {
  if (!currentSession) return;
  if (!confirm('Delete this chat?')) return;
  await api.chat.deleteSession(currentSession);
  currentSession = null;
  document.getElementById('chat-title').textContent = 'No chat selected';
  document.getElementById('delete-chat-btn').style.display = 'none';
  document.getElementById('messages').innerHTML =
    '<div class="empty-state"><div class="empty-icon">⚡</div><p>Start a new chat or select an existing one</p></div>';
  loadSessions();
});

// Send message
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// Auto-resize textarea
document.getElementById('chat-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});
