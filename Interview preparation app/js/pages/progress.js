// progress.js - Progress tracking and heatmap

function renderProgress() {
  const progress = Storage.getProgress();
  const history = Storage.getInterviewHistory();
  const activity = Storage.getActivity();
  const streak = Storage.getStreak();
  const solved = Storage.getCodingSolved();

  const avgScore = history.length > 0
    ? (history.reduce((s, h) => s + (h.score || 0), 0) / history.length).toFixed(1)
    : 0;

  return `
  <div class="animate-fade">
    <div class="page-header">
      <h1 class="page-title">📊 Your Progress</h1>
      <p class="page-subtitle">Track your improvement journey over time</p>
    </div>

    <!-- Stats Grid -->
    <div class="grid-4" style="margin-bottom:28px">
      <div class="stat-card purple">
        <div class="stat-icon purple">🎤</div>
        <div class="stat-value" style="background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">${progress.mockInterviews}</div>
        <div class="stat-label">Mock Interviews</div>
      </div>
      <div class="stat-card cyan">
        <div class="stat-icon cyan">⭐</div>
        <div class="stat-value" style="color:var(--color-accent-light)">${avgScore}</div>
        <div class="stat-label">Avg Score</div>
      </div>
      <div class="stat-card green">
        <div class="stat-icon green">📚</div>
        <div class="stat-value" style="color:var(--color-success-light)">${progress.flashcardsKnown}</div>
        <div class="stat-label">Cards Mastered</div>
      </div>
      <div class="stat-card orange">
        <div class="stat-icon orange">🔥</div>
        <div class="stat-value" style="color:var(--color-warning-light)">${streak.count}</div>
        <div class="stat-label">Day Streak</div>
      </div>
    </div>

    <!-- Heatmap -->
    <div class="card" style="margin-bottom:24px">
      <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">📅 Activity (Last 6 Months)</div>
      <div class="heatmap" id="heatmap">
        ${renderHeatmap(activity)}
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:12px;font-size:12px;color:var(--text-muted)">
        <span>Less</span>
        <div style="width:12px;height:12px;border-radius:2px;background:var(--bg-input)"></div>
        <div style="width:12px;height:12px;border-radius:2px;background:rgba(124,58,237,0.25)" data-level="1"></div>
        <div style="width:12px;height:12px;border-radius:2px;background:rgba(124,58,237,0.5)" data-level="2"></div>
        <div style="width:12px;height:12px;border-radius:2px;background:rgba(124,58,237,0.75)" data-level="3"></div>
        <div style="width:12px;height:12px;border-radius:2px;background:var(--color-primary)" data-level="4"></div>
        <span>More</span>
      </div>
    </div>

    <div class="grid-2" style="margin-bottom:24px;gap:20px">
      <!-- Interview Score Chart -->
      <div class="card">
        <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">📈 Interview Scores</div>
        ${history.length === 0 ? `
          <div class="empty-state" style="padding:40px 20px">
            <div class="empty-state-icon" style="font-size:32px">🎤</div>
            <div class="empty-state-title" style="font-size:15px">No interviews yet</div>
            <div class="empty-state-desc" style="font-size:13px">Start a mock interview to see your scores here.</div>
          </div>
        ` : renderScoreChart(history)}
      </div>

      <!-- Coding Progress -->
      <div class="card">
        <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">💻 Coding Progress</div>
        <div style="margin-bottom:20px">
          ${['Easy','Medium','Hard'].map(diff => {
            const total = CODING_PROBLEMS.filter(p => p.difficulty === diff).length;
            const done = solved.filter(id => CODING_PROBLEMS.find(p => p.id === id && p.difficulty === diff)).length;
            const pct = total > 0 ? Math.round((done/total)*100) : 0;
            const color = diff === 'Easy' ? 'var(--color-success)' : diff === 'Medium' ? 'var(--color-warning)' : 'var(--color-danger)';
            return `
              <div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                  <span style="font-size:13px;font-weight:600;color:var(--text-secondary)">${diff}</span>
                  <span style="font-size:12px;color:var(--text-muted)">${done}/${total}</span>
                </div>
                <div class="progress-bar">
                  <div style="height:100%;background:${color};border-radius:9999px;width:${pct}%;transition:width 0.6s ease"></div>
                </div>
              </div>`;
          }).join('')}
        </div>
        <div style="text-align:center">
          <div style="font-size:32px;font-weight:800;background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">${solved.length}/${CODING_PROBLEMS.length}</div>
          <div style="font-size:13px;color:var(--text-secondary)">Problems Solved</div>
        </div>
      </div>
    </div>

    <!-- Interview History Table -->
    ${history.length > 0 ? `
    <div class="card">
      <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">📋 Interview History</div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <thead>
            <tr style="border-bottom:1px solid var(--border)">
              ${['Date','Role','Difficulty','Questions','Score','Rating'].map(h => `<th style="text-align:left;padding:8px 12px;color:var(--text-muted);font-weight:600;white-space:nowrap">${h}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${history.map(s => {
              const scoreColor = s.score >= 8 ? 'var(--color-success-light)' : s.score >= 6 ? 'var(--color-accent-light)' : 'var(--color-warning-light)';
              const badge = s.score >= 8 ? 'badge-easy' : s.score >= 6 ? 'badge-accent' : 'badge-medium';
              return `
              <tr style="border-bottom:1px solid rgba(255,255,255,0.03)">
                <td style="padding:10px 12px;color:var(--text-secondary)">${new Date(s.date).toLocaleDateString()}</td>
                <td style="padding:10px 12px;font-weight:600">${s.role}</td>
                <td style="padding:10px 12px;color:var(--text-secondary)">${s.difficulty}</td>
                <td style="padding:10px 12px;color:var(--text-secondary)">${s.questionCount}</td>
                <td style="padding:10px 12px;font-weight:800;color:${scoreColor}">${s.score}</td>
                <td style="padding:10px 12px"><span class="badge ${badge}">${s.score >= 8 ? 'Excellent' : s.score >= 6 ? 'Good' : 'Average'}</span></td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
    </div>` : ''}
  </div>`;
}

function renderHeatmap(activity) {
  const cells = [];
  const today = new Date();
  // Go back ~26 weeks
  const start = new Date(today);
  start.setDate(start.getDate() - (26 * 7));

  for (let d = new Date(start); d <= today; d.setDate(d.getDate() + 1)) {
    const dateStr = d.toISOString().split('T')[0];
    const count = activity[dateStr] || 0;
    const level = count === 0 ? 0 : count === 1 ? 1 : count <= 3 ? 2 : count <= 5 ? 3 : 4;
    cells.push(`<div class="heatmap-cell" data-level="${level}" data-date="${dateStr}" data-count="${count}" title="${dateStr}: ${count} activities"></div>`);
  }
  return cells.join('');
}

function renderScoreChart(history) {
  const recent = history.slice(0, 10).reverse(); // show oldest to newest
  const maxScore = 10;

  return `
  <div style="display:flex;align-items:flex-end;gap:6px;height:140px;padding-bottom:4px">
    ${recent.map((s, i) => {
      const h = Math.round((s.score / maxScore) * 120);
      const color = s.score >= 8 ? 'var(--color-success)' : s.score >= 6 ? 'var(--color-primary)' : 'var(--color-warning)';
      return `
        <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px">
          <div style="font-size:10px;color:var(--text-muted)">${s.score}</div>
          <div style="width:100%;height:${h}px;background:${color};border-radius:4px 4px 0 0;opacity:0.8;transition:height 0.6s ease" title="${s.role}: ${s.score}/10"></div>
        </div>`;
    }).join('')}
  </div>
  <div style="display:flex;gap:6px;padding-top:4px">
    ${recent.map((s, i) => `
      <div style="flex:1;text-align:center;font-size:9px;color:var(--text-muted);overflow:hidden">#${i+1}</div>
    `).join('')}
  </div>`;
}
