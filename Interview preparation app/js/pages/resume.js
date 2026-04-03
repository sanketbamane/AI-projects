// resume.js - AI-powered resume analyzer

function renderResume() {
  return `
  <div class="animate-fade">
    <div class="page-header">
      <h1 class="page-title">📝 Resume Analyzer</h1>
      <p class="page-subtitle">Get an ATS score, keyword analysis, and improvement suggestions powered by AI</p>
    </div>

    <div class="grid-2" style="gap:24px;align-items:start">
      <!-- Input -->
      <div>
        <div class="card" style="margin-bottom:16px">
          <h2 style="font-size:15px;font-weight:700;margin-bottom:16px">Your Resume</h2>

          <div class="form-group" style="margin-bottom:14px">
            <label class="form-label">Target Job Role</label>
            <input class="input" id="resume-role" type="text" placeholder="e.g. Senior Frontend Developer" value="${Storage.getTargetRole()}" />
          </div>

          <div class="form-group" style="margin-bottom:14px">
            <label class="form-label">Paste Resume Text</label>
            <textarea class="textarea" id="resume-text" style="min-height:280px;font-size:13px" placeholder="Paste your resume content here...

Name
Email | Phone | LinkedIn | GitHub

EXPERIENCE
Company Name - Job Title (2022 - Present)
• Bullet point achievements...

EDUCATION
University Name - Degree (Year)

SKILLS
JavaScript, React, Node.js..."></textarea>
          </div>

          <div style="margin-bottom:14px">
            <label class="form-label" style="margin-bottom:8px;display:block">Or upload a .txt file</label>
            <input type="file" id="resume-file" accept=".txt" onchange="loadResumeFile(this)" style="display:none" />
            <button class="btn btn-secondary btn-sm" onclick="document.getElementById('resume-file').click()">📁 Upload .txt</button>
          </div>

          ${!Storage.getApiKey() ? `<div class="alert alert-warning" style="margin-bottom:14px">⚠️ API key required. <a href="#" onclick="navigateTo('settings')" style="color:var(--color-warning-light);font-weight:600">Set it in Settings</a></div>` : ''}

          <button class="btn btn-primary btn-lg" style="width:100%" onclick="analyzeResume()" id="analyze-btn">
            🔍 Analyze Resume
          </button>
        </div>

        <div class="alert alert-info">
          💡 <strong>Tip:</strong> The more complete your resume text, the more accurate the analysis. Include all sections: experience, education, skills, projects.
        </div>
      </div>

      <!-- Results -->
      <div id="resume-results">
        <div class="card" style="text-align:center;padding:60px 20px">
          <div style="font-size:48px;margin-bottom:16px">📋</div>
          <div style="font-size:17px;font-weight:700;color:var(--text-primary);margin-bottom:8px">Ready to Analyze</div>
          <div style="font-size:13px;color:var(--text-secondary);line-height:1.6;max-width:280px;margin:0 auto">
            Paste your resume text and click Analyze to get your ATS score and detailed feedback.
          </div>
        </div>
      </div>
    </div>
  </div>`;
}

function loadResumeFile(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const ta = document.getElementById('resume-text');
    if (ta) ta.value = e.target.result;
    showToast('File loaded!', 'success');
  };
  reader.readAsText(file);
}

async function analyzeResume() {
  const text = document.getElementById('resume-text')?.value?.trim();
  const role = document.getElementById('resume-role')?.value?.trim() || 'Software Engineer';

  if (!text || text.length < 100) {
    showToast('Please paste your resume text (at least a few lines)', 'error');
    return;
  }
  if (!Storage.getApiKey()) {
    showToast('Add a Gemini API key in Settings first', 'error');
    return;
  }

  const btn = document.getElementById('analyze-btn');
  const resultsEl = document.getElementById('resume-results');

  if (btn) { btn.disabled = true; btn.innerHTML = `<div class="spinner" style="width:16px;height:16px;display:inline-block;vertical-align:middle;margin-right:8px"></div>Analyzing...`; }
  if (resultsEl) resultsEl.innerHTML = `
    <div class="card" style="text-align:center;padding:60px 20px">
      <div class="spinner" style="width:40px;height:40px;border-width:3px;margin:0 auto 16px"></div>
      <div style="font-size:15px;font-weight:600;color:var(--text-primary)">Analyzing your resume...</div>
      <div style="font-size:13px;color:var(--text-secondary);margin-top:6px">AI is reviewing your resume for ${role}</div>
    </div>`;

  try {
    const analysis = await Gemini.analyzeResume(text, role);
    Storage.logActivity();
    showResumeResults(analysis, role, resultsEl);
  } catch (e) {
    if (resultsEl) resultsEl.innerHTML = `<div class="alert alert-danger">❌ Analysis failed: ${e.message}</div>`;
    showToast('Analysis failed: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '🔍 Analyze Resume'; }
  }
}

function showResumeResults(analysis, role, container) {
  const score = analysis.atsScore || 50;
  const scoreColor = score >= 75 ? 'var(--color-success-light)' : score >= 55 ? 'var(--color-accent-light)' : 'var(--color-warning-light)';
  const ratingEmoji = { 'Excellent': '🏆', 'Strong': '💪', 'Average': '👍', 'Weak': '⚠️' };

  // Circular ring SVG
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (score / 100) * circumference;

  container.innerHTML = `
  <div class="animate-fade" style="display:flex;flex-direction:column;gap:16px">

    <!-- Score -->
    <div class="card card-gradient" style="text-align:center;padding:28px">
      <div style="margin-bottom:12px">
        <svg width="110" height="110" viewBox="0 0 110 110">
          <circle cx="55" cy="55" r="${radius}" fill="none" stroke="var(--bg-input)" stroke-width="10"/>
          <circle cx="55" cy="55" r="${radius}" fill="none" stroke="${scoreColor}" stroke-width="10"
            stroke-dasharray="${circumference}" stroke-dashoffset="${dashOffset}"
            stroke-linecap="round" style="transform:rotate(-90deg);transform-origin:55px 55px;transition:stroke-dashoffset 1s ease"/>
          <text x="55" y="52" text-anchor="middle" fill="${scoreColor}" font-size="22" font-weight="800" font-family="Inter">${score}</text>
          <text x="55" y="67" text-anchor="middle" fill="var(--text-muted)" font-size="10" font-family="Inter">ATS Score</text>
        </svg>
      </div>
      <div style="font-size:18px;font-weight:800">${ratingEmoji[analysis.overallRating] || '📋'} ${analysis.overallRating}</div>
      <div style="font-size:13px;color:var(--text-secondary);margin-top:8px;line-height:1.6;max-width:320px;margin:8px auto 0">${analysis.summary}</div>
    </div>

    <!-- Strengths & Weaknesses -->
    <div class="card" style="background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.2)">
      <div style="font-size:12px;font-weight:700;color:var(--color-success-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">✅ Strengths</div>
      <ul style="list-style:none;display:flex;flex-direction:column;gap:6px">
        ${(analysis.strengths || []).map(s => `<li style="font-size:13px;color:var(--text-secondary);display:flex;gap:8px"><span style="color:var(--color-success-light);flex-shrink:0">•</span>${s}</li>`).join('')}
      </ul>
    </div>

    <div class="card" style="background:rgba(239,68,68,0.06);border-color:rgba(239,68,68,0.2)">
      <div style="font-size:12px;font-weight:700;color:var(--color-danger-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">⚠️ Weaknesses</div>
      <ul style="list-style:none;display:flex;flex-direction:column;gap:6px">
        ${(analysis.weaknesses || []).map(w => `<li style="font-size:13px;color:var(--text-secondary);display:flex;gap:8px"><span style="color:var(--color-danger-light);flex-shrink:0">•</span>${w}</li>`).join('')}
      </ul>
    </div>

    <!-- Missing Keywords -->
    ${(analysis.missingKeywords || []).length > 0 ? `
    <div class="card">
      <div style="font-size:12px;font-weight:700;color:var(--color-primary-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">🔑 Missing Keywords for ${role}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">
        ${analysis.missingKeywords.map(k => `<span class="badge badge-primary">${k}</span>`).join('')}
      </div>
    </div>` : ''}

    <!-- Suggestions -->
    ${(analysis.suggestions || []).length > 0 ? `
    <div class="card">
      <div style="font-size:12px;font-weight:700;color:var(--color-accent-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:14px">💡 Specific Suggestions</div>
      <div style="display:flex;flex-direction:column;gap:12px">
        ${analysis.suggestions.map(s => `
          <div style="border-left:2px solid var(--color-primary);padding-left:14px">
            <div style="font-size:12px;font-weight:700;color:var(--color-primary-light);margin-bottom:4px">${s.section}</div>
            <div style="font-size:13px;color:var(--text-secondary);margin-bottom:4px">Issue: ${s.issue}</div>
            <div style="font-size:13px;color:var(--text-primary)">→ ${s.fix}</div>
          </div>
        `).join('')}
      </div>
    </div>` : ''}

    <!-- Format Tips -->
    ${(analysis.formatTips || []).length > 0 ? `
    <div class="card">
      <div style="font-size:12px;font-weight:700;color:var(--color-warning-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">📐 Format Tips</div>
      <ul style="list-style:none;display:flex;flex-direction:column;gap:6px">
        ${analysis.formatTips.map(t => `<li style="font-size:13px;color:var(--text-secondary);display:flex;gap:8px"><span style="color:var(--color-warning-light);flex-shrink:0">→</span>${t}</li>`).join('')}
      </ul>
    </div>` : ''}
  </div>`;
}
