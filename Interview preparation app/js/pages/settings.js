// settings.js

function renderSettings() {
  const apiKey = Storage.getApiKey();
  const name = Storage.getUserName();
  const role = Storage.getTargetRole();

  return `
  <div class="animate-fade">
    <div class="page-header">
      <h1 class="page-title">⚙️ Settings</h1>
      <p class="page-subtitle">Configure your PrepAI experience</p>
    </div>

    <div style="max-width:600px;display:flex;flex-direction:column;gap:20px">

      <!-- Profile -->
      <div class="card">
        <h2 style="font-size:16px;font-weight:700;margin-bottom:18px;display:flex;align-items:center;gap:8px">👤 Your Profile</h2>
        <div style="display:flex;flex-direction:column;gap:14px">
          <div class="form-group">
            <label class="form-label">Your Name</label>
            <input class="input" id="setting-name" type="text" placeholder="Enter your name" value="${name !== 'there' ? name : ''}" />
          </div>
          <div class="form-group">
            <label class="form-label">Target Role</label>
            <select class="select" id="setting-role">
              ${['Frontend Developer','Backend Developer','Full Stack Developer','Data Scientist','ML Engineer','DevOps Engineer','Product Manager','iOS Developer','Android Developer','Software Engineer','QA Engineer','System Design'].map(r =>
                `<option value="${r}" ${role === r ? 'selected' : ''}>${r}</option>`
              ).join('')}
            </select>
          </div>
          <button class="btn btn-primary" id="save-profile-btn" onclick="saveProfile()">Save Profile</button>
        </div>
      </div>

      <!-- API Key -->
      <div class="card" style="border-color:rgba(124,58,237,0.2)">
        <h2 style="font-size:16px;font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px">🔑 Groq API Key</h2>
        <p style="font-size:13px;color:var(--text-secondary);margin-bottom:18px;line-height:1.6">
          Required for all AI features: voice interview questions, dynamic flashcards, resume analysis, and code hints.
          Get a <strong>free</strong> key at <a href="https://console.groq.com" target="_blank" style="color:var(--color-primary-light)">console.groq.com</a> → API Keys → Create API Key.
        </p>
        <div class="form-group">
          <label class="form-label">API Key</label>
          <div style="display:flex;gap:10px">
            <input class="input" id="setting-apikey" type="password" placeholder="AIza..." value="${apiKey}" style="font-family:monospace" />
            <button class="btn btn-secondary btn-icon" onclick="toggleApiKeyVisibility()" id="toggle-key-btn" title="Show/hide key" data-tooltip="Toggle visibility">👁️</button>
          </div>
        </div>
        <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap">
          <button class="btn btn-primary" onclick="saveApiKey()">💾 Save Key</button>
          <button class="btn btn-secondary" onclick="testApiKey()">🧪 Test Connection</button>
          ${apiKey ? `<button class="btn btn-danger btn-sm" onclick="clearApiKey()">Remove Key</button>` : ''}
        </div>
        <div id="api-test-result" style="margin-top:12px"></div>
        <div style="margin-top:14px;padding:12px;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:8px">
          <div style="font-size:12px;color:var(--color-accent-light);font-weight:600;margin-bottom:4px">🔒 Privacy Note</div>
          <div style="font-size:12px;color:var(--text-secondary);line-height:1.5">Your API key is stored only in your browser's local storage. It is never sent to any server other than Groq's API directly from your browser.</div>
        </div>
      </div>

      <!-- Data -->
      <div class="card">
        <h2 style="font-size:16px;font-weight:700;margin-bottom:18px;display:flex;align-items:center;gap:8px">🗑️ Data & Privacy</h2>
        <p style="font-size:13px;color:var(--text-secondary);margin-bottom:16px">All your data is stored locally in your browser. Nothing is sent to external servers (except direct API calls to Google).</p>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-secondary" onclick="exportData()">📤 Export Progress</button>
          <button class="btn btn-danger" onclick="clearAllData()">🗑️ Clear All Data</button>
        </div>
      </div>

      <!-- About -->
      <div class="card" style="text-align:center;padding:28px">
        <div style="font-size:40px;margin-bottom:12px">🎯</div>
        <div style="font-size:20px;font-weight:800;background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">PrepAI</div>
        <div style="font-size:13px;color:var(--text-secondary);margin-top:6px">Your personal AI interview coach</div>
        <div style="font-size:12px;color:var(--text-muted);margin-top:4px">Powered by Groq (Llama 3.3) · Web Speech API</div>
      </div>
    </div>
  </div>`;
}

function saveProfile() {
  const name = document.getElementById('setting-name').value.trim();
  const role = document.getElementById('setting-role').value;
  if (name) Storage.setUserName(name);
  Storage.setTargetRole(role);
  showToast('Profile saved!', 'success');
  // Re-render sidebar greeting if on dashboard
}

function saveApiKey() {
  const key = document.getElementById('setting-apikey').value.trim();
  if (!key) { showToast('Please enter an API key', 'error'); return; }
  Storage.setApiKey(key);
  showToast('API key saved!', 'success');
  navigateTo('settings'); // re-render to show remove button
}

function clearApiKey() {
  if (!confirm('Remove your API key?')) return;
  Storage.setApiKey('');
  showToast('API key removed', 'success');
  navigateTo('settings');
}

function toggleApiKeyVisibility() {
  const input = document.getElementById('setting-apikey');
  input.type = input.type === 'password' ? 'text' : 'password';
}

async function testApiKey() {
  const key = document.getElementById('setting-apikey').value.trim();
  if (!key) { showToast('Enter a key first', 'error'); return; }
  Storage.setApiKey(key);
  const resultEl = document.getElementById('api-test-result');
  resultEl.innerHTML = `<div class="alert alert-info" style="margin:0"><div class="spinner" style="width:14px;height:14px;display:inline-block;vertical-align:middle;margin-right:8px"></div>Testing connection...</div>`;
  try {
    const response = await Gemini.call('Say "Connection successful" and nothing else.');
    resultEl.innerHTML = `<div class="alert alert-success" style="margin:0">✅ ${response.trim()}</div>`;
  } catch (e) {
    resultEl.innerHTML = `<div class="alert alert-danger" style="margin:0">❌ ${e.message}</div>`;
  }
}

function exportData() {
  const data = {
    profile: { name: Storage.getUserName(), role: Storage.getTargetRole() },
    progress: Storage.getProgress(),
    interviewHistory: Storage.getInterviewHistory(),
    activity: Storage.getActivity(),
    exportedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'prepai-progress.json'; a.click();
  URL.revokeObjectURL(url);
  showToast('Data exported!', 'success');
}

function clearAllData() {
  if (!confirm('Clear ALL data? This cannot be undone.')) return;
  const key = Storage.getApiKey(); // preserve API key
  localStorage.clear();
  Storage.setApiKey(key);
  showToast('All data cleared', 'success');
  navigateTo('dashboard');
}
