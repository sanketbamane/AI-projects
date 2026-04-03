// dashboard.js

function renderDashboard() {
  const progress = Storage.getProgress();
  const streak = Storage.getStreak();
  const history = Storage.getInterviewHistory();
  const name = Storage.getUserName();
  const role = Storage.getTargetRole();
  const avgScore = progress.mockInterviews > 0
    ? (progress.totalScore / progress.mockInterviews).toFixed(1)
    : '—';

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  const lastSession = history[0];

  return `
  <div class="animate-fade">
    <div class="page-header">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div>
          <h1 class="page-title">${greeting}, ${name}! 👋</h1>
          <p class="page-subtitle">Target Role: <strong style="color:var(--color-primary-light)">${role}</strong> &nbsp;·&nbsp; Let's get you interview-ready.</p>
        </div>
        ${streak.count > 0 ? `<div class="card" style="padding:12px 20px;display:flex;align-items:center;gap:10px;border-color:rgba(245,158,11,0.3);background:rgba(245,158,11,0.06)">
          <span style="font-size:24px">🔥</span>
          <div><div style="font-size:20px;font-weight:800;color:var(--color-warning-light)">${streak.count}</div><div style="font-size:11px;color:var(--text-secondary)">Day Streak</div></div>
        </div>` : ''}
      </div>
    </div>

    <!-- Stats -->
    <div class="grid-4" style="margin-bottom:28px">
      <div class="stat-card purple">
        <div class="stat-icon purple">🎤</div>
        <div class="stat-value" style="background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">${progress.mockInterviews}</div>
        <div class="stat-label">Mock Interviews</div>
      </div>
      <div class="stat-card cyan">
        <div class="stat-icon cyan">⭐</div>
        <div class="stat-value" style="color:var(--color-accent-light)">${avgScore}</div>
        <div class="stat-label">Avg Interview Score</div>
      </div>
      <div class="stat-card green">
        <div class="stat-icon green">📚</div>
        <div class="stat-value" style="color:var(--color-success-light)">${progress.flashcardsKnown}</div>
        <div class="stat-label">Flashcards Mastered</div>
      </div>
      <div class="stat-card orange">
        <div class="stat-icon orange">💻</div>
        <div class="stat-value" style="color:var(--color-warning-light)">${progress.codingSolved}</div>
        <div class="stat-label">Problems Solved</div>
      </div>
    </div>

    <div class="grid-2" style="margin-bottom:28px">
      <!-- Quick Actions -->
      <div>
        <div class="section-header"><h2 class="section-title">Quick Start</h2></div>
        <div style="display:flex;flex-direction:column;gap:12px">
          ${[
            { page: 'interview', icon: '🎤', title: 'Start Voice Interview', desc: 'AI asks, you speak — get instant feedback', color: 'var(--color-primary)', glow: 'var(--color-primary-glow)' },
            { page: 'flashcards', icon: '📚', title: 'Study Flashcards', desc: 'AI-generated Q&A for your topic', color: 'var(--color-accent)', glow: 'var(--color-accent-glow)' },
            { page: 'coding', icon: '💻', title: 'Coding Practice', desc: 'DSA problems with AI hints', color: 'var(--color-success)', glow: 'var(--color-success-glow)' },
            { page: 'resume', icon: '📝', title: 'Analyze Resume', desc: 'Get ATS score & improvement tips', color: 'var(--color-warning)', glow: 'rgba(245,158,11,0.3)' },
          ].map(a => `
            <div class="card" style="cursor:pointer;display:flex;align-items:center;gap:16px;padding:18px 20px;" onclick="navigateTo('${a.page}')">
              <div style="width:44px;height:44px;border-radius:10px;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">${a.icon}</div>
              <div style="flex:1">
                <div style="font-weight:700;font-size:15px;color:var(--text-primary)">${a.title}</div>
                <div style="font-size:13px;color:var(--text-secondary);margin-top:2px">${a.desc}</div>
              </div>
              <span style="color:var(--text-muted);font-size:18px">›</span>
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Recent Activity -->
      <div>
        <div class="section-header"><h2 class="section-title">Recent Interviews</h2></div>
        ${history.length === 0 ? `
          <div class="card empty-state" style="padding:40px 20px">
            <div class="empty-state-icon">🎯</div>
            <div class="empty-state-title">No interviews yet</div>
            <div class="empty-state-desc">Start your first mock interview to see your history here.</div>
            <button class="btn btn-primary" onclick="navigateTo('interview')">Start Interview</button>
          </div>
        ` : `
          <div style="display:flex;flex-direction:column;gap:10px">
            ${history.slice(0, 4).map(s => `
              <div class="card" style="padding:16px 20px;display:flex;align-items:center;gap:14px">
                <div style="width:48px;height:48px;border-radius:50%;background:var(--gradient-primary);display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:800;flex-shrink:0">${s.score}</div>
                <div style="flex:1">
                  <div style="font-weight:600;font-size:14px">${s.role}</div>
                  <div style="font-size:12px;color:var(--text-secondary);margin-top:2px">${s.difficulty} · ${s.questionCount} questions · ${new Date(s.date).toLocaleDateString()}</div>
                </div>
                <span class="badge ${s.score >= 8 ? 'badge-easy' : s.score >= 6 ? 'badge-accent' : 'badge-medium'}">${s.score >= 8 ? 'Excellent' : s.score >= 6 ? 'Good' : 'Practice'}</span>
              </div>
            `).join('')}
          </div>
        `}
      </div>
    </div>

    <!-- API Key Warning -->
    ${!Storage.getApiKey() ? `
      <div class="alert alert-warning" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div style="display:flex;align-items:center;gap:10px">
          <span>⚠️</span>
          <span><strong>Gemini API key not set.</strong> Add your free key to unlock AI-powered questions, voice interviews, resume analysis, and more.</span>
        </div>
        <button class="btn btn-primary btn-sm" onclick="navigateTo('settings')">Add API Key →</button>
      </div>
    ` : ''}
  </div>`;
}
