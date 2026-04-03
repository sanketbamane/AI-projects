// coding.js - DSA practice with AI hints and code review

let codingState = {
  currentProblem: null,
  hintCount: 0,
  showSolution: false,
};

function renderCoding() {
  const solved = Storage.getCodingSolved();

  if (codingState.currentProblem) {
    return renderCodingProblem();
  }

  return `
  <div class="animate-fade">
    <div class="page-header">
      <h1 class="page-title">💻 Coding Practice</h1>
      <p class="page-subtitle">DSA problems with AI-powered hints and code review</p>
    </div>

    <!-- Stats bar -->
    <div class="card" style="margin-bottom:24px;padding:16px 20px;display:flex;align-items:center;gap:24px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:36px;height:36px;border-radius:8px;background:rgba(16,185,129,0.15);display:flex;align-items:center;justify-content:center">✅</div>
        <div><div style="font-size:18px;font-weight:800;color:var(--color-success-light)">${solved.length}</div><div style="font-size:12px;color:var(--text-secondary)">Solved</div></div>
      </div>
      <div style="height:32px;width:1px;background:var(--border)"></div>
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:36px;height:36px;border-radius:8px;background:rgba(124,58,237,0.15);display:flex;align-items:center;justify-content:center">📊</div>
        <div><div style="font-size:18px;font-weight:800;color:var(--color-primary-light)">${CODING_PROBLEMS.length}</div><div style="font-size:12px;color:var(--text-secondary)">Total</div></div>
      </div>
      <div style="height:32px;width:1px;background:var(--border)"></div>
      <div style="flex:1">
        <div style="font-size:12px;color:var(--text-secondary);margin-bottom:6px">Overall Progress</div>
        <div class="progress-bar"><div class="progress-fill" style="width:${Math.round((solved.length/CODING_PROBLEMS.length)*100)}%"></div></div>
      </div>
    </div>

    <!-- Filters -->
    <div style="display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap">
      <span class="tag active" id="filter-all" onclick="filterProblems('all',this)">All</span>
      <span class="tag" id="filter-easy" onclick="filterProblems('Easy',this)">Easy</span>
      <span class="tag" id="filter-medium" onclick="filterProblems('Medium',this)">Medium</span>
      <span class="tag" id="filter-hard" onclick="filterProblems('Hard',this)">Hard</span>
    </div>

    <!-- Problem List -->
    <div id="problem-list" style="display:flex;flex-direction:column;gap:10px">
      ${CODING_PROBLEMS.map(p => renderProblemCard(p, solved)).join('')}
    </div>
  </div>`;
}

function renderProblemCard(p, solved) {
  const isSolved = solved.includes(p.id);
  return `
  <div class="card" style="cursor:pointer;display:flex;align-items:center;gap:16px;padding:18px 20px;${isSolved ? 'border-color:rgba(16,185,129,0.25);background:rgba(16,185,129,0.03)' : ''}" onclick="openProblem(${p.id})">
    <div style="width:32px;height:32px;border-radius:8px;background:${isSolved ? 'rgba(16,185,129,0.15)' : 'var(--bg-input)'};display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0">
      ${isSolved ? '✅' : '⬜'}
    </div>
    <div style="flex:1">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <span style="font-weight:700;font-size:15px">${p.title}</span>
        <span class="badge badge-${p.difficulty.toLowerCase()}">${p.difficulty}</span>
        <span class="badge badge-accent">${p.category}</span>
      </div>
    </div>
    <span style="color:var(--text-muted)">›</span>
  </div>`;
}

function filterProblems(difficulty, el) {
  document.querySelectorAll('#filter-all,#filter-easy,#filter-medium,#filter-hard').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  const solved = Storage.getCodingSolved();
  const list = document.getElementById('problem-list');
  if (!list) return;
  const filtered = difficulty === 'all' ? CODING_PROBLEMS : CODING_PROBLEMS.filter(p => p.difficulty === difficulty);
  list.innerHTML = filtered.map(p => renderProblemCard(p, solved)).join('');
}

function openProblem(id) {
  const problem = CODING_PROBLEMS.find(p => p.id === id);
  if (!problem) return;
  codingState.currentProblem = problem;
  codingState.hintCount = 0;
  codingState.showSolution = false;
  renderPage('coding');
}

function renderCodingProblem() {
  const p = codingState.currentProblem;
  const isSolved = Storage.getCodingSolved().includes(p.id);

  return `
  <div class="animate-fade">
    <div class="page-header" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
      <div>
        <button class="btn btn-secondary btn-sm" onclick="codingState.currentProblem=null;renderPage('coding')" style="margin-bottom:8px">← Back to Problems</button>
        <h1 class="page-title">${p.title}</h1>
        <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap">
          <span class="badge badge-${p.difficulty.toLowerCase()}">${p.difficulty}</span>
          <span class="badge badge-accent">${p.category}</span>
          ${isSolved ? '<span class="badge badge-easy">✅ Solved</span>' : ''}
        </div>
      </div>
    </div>

    <div class="grid-2" style="gap:20px;align-items:start">
      <!-- Problem Statement -->
      <div>
        <div class="card" style="margin-bottom:16px">
          <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px">Problem Description</div>
          <div style="font-size:14px;line-height:1.8;color:var(--text-secondary);white-space:pre-wrap">${p.description}</div>
        </div>

        <!-- Hints -->
        <div class="card" style="margin-bottom:16px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div style="font-size:12px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px">💡 Hints</div>
            <span style="font-size:12px;color:var(--text-muted)">${codingState.hintCount}/${p.hints.length} revealed</span>
          </div>
          <div id="hints-container" style="display:flex;flex-direction:column;gap:8px">
            ${p.hints.slice(0, codingState.hintCount).map((h, i) => `
              <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:8px;padding:12px;font-size:13px;color:var(--text-secondary);line-height:1.6;animation:fadeIn 0.3s ease">
                <span style="font-weight:700;color:var(--color-warning-light)">Hint ${i+1}:</span> ${h}
              </div>
            `).join('')}
          </div>
          <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap">
            ${codingState.hintCount < p.hints.length ? `<button class="btn btn-secondary btn-sm" onclick="revealHint()">💡 Reveal Hint ${codingState.hintCount + 1}</button>` : ''}
            ${Storage.getApiKey() ? `<button class="btn btn-secondary btn-sm" onclick="getAIHint()">🤖 Ask AI</button>` : ''}
          </div>
          <div id="ai-hint-container" style="margin-top:10px"></div>
        </div>

        <!-- Solution -->
        ${codingState.showSolution ? `
        <div class="card" style="border-color:rgba(6,182,212,0.3);background:rgba(6,182,212,0.04)">
          <div style="font-size:12px;font-weight:700;color:var(--color-accent-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px">✅ Solution & Explanation</div>
          <div class="code-editor" style="margin-bottom:14px">
            <div class="code-editor-header">
              <div class="editor-dots"><div class="editor-dot red"></div><div class="editor-dot yellow"></div><div class="editor-dot green"></div></div>
              <span style="font-size:12px;color:var(--text-muted)">solution.js</span>
            </div>
            <pre class="code-textarea" style="overflow-x:auto;min-height:auto">${p.solution}</pre>
          </div>
          <div style="font-size:13px;color:var(--text-secondary);line-height:1.7;background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);border-radius:8px;padding:14px">
            📖 ${p.explanation}
          </div>
        </div>
        ` : `
        <button class="btn btn-secondary" onclick="codingState.showSolution=true;renderPage('coding')">👁️ Show Solution</button>
        `}
      </div>

      <!-- Code Editor -->
      <div>
        <div class="code-editor" style="margin-bottom:14px">
          <div class="code-editor-header">
            <div class="editor-dots">
              <div class="editor-dot red"></div>
              <div class="editor-dot yellow"></div>
              <div class="editor-dot green"></div>
            </div>
            <span style="font-size:12px;color:var(--text-muted)">Your Solution</span>
          </div>
          <textarea class="code-textarea" id="user-code" style="min-height:300px" placeholder="// Write your solution here...
function ${p.title.replace(/\s+/g,'').replace(/[^a-zA-Z]/g,'')}(...) {
  // Your code
}"></textarea>
        </div>

        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">
          ${Storage.getApiKey() ? `<button class="btn btn-primary" onclick="reviewCode()">🤖 AI Review</button>` : ''}
          <button class="btn btn-success" onclick="markSolved()">✅ Mark as Solved</button>
          <button class="btn btn-secondary btn-sm" onclick="clearCode()">🗑️ Clear</button>
        </div>

        <div id="code-review-result"></div>
      </div>
    </div>
  </div>`;
}

function revealHint() {
  if (codingState.hintCount < codingState.currentProblem.hints.length) {
    codingState.hintCount++;
    renderPage('coding');
  }
}

async function getAIHint() {
  const p = codingState.currentProblem;
  const userCode = document.getElementById('user-code')?.value || '';
  const container = document.getElementById('ai-hint-container');
  if (!container) return;

  container.innerHTML = `<div style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-secondary)"><div class="spinner" style="width:14px;height:14px"></div>AI thinking...</div>`;

  try {
    const hint = await Gemini.getCodeHint(p, userCode, codingState.hintCount + 1);
    container.innerHTML = `
      <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.25);border-radius:8px;padding:12px;font-size:13px;color:var(--text-secondary);line-height:1.6;animation:fadeIn 0.3s ease">
        <span style="font-weight:700;color:var(--color-primary-light)">🤖 AI Hint:</span> ${hint}
      </div>`;
  } catch (e) {
    container.innerHTML = `<div class="alert alert-danger" style="margin:0;font-size:13px">❌ ${e.message}</div>`;
  }
}

async function reviewCode() {
  const p = codingState.currentProblem;
  const userCode = document.getElementById('user-code')?.value?.trim();
  if (!userCode || userCode.length < 10) {
    showToast('Write some code first!', 'error');
    return;
  }

  const container = document.getElementById('code-review-result');
  if (container) container.innerHTML = `<div class="card" style="display:flex;align-items:center;gap:10px;color:var(--text-secondary);font-size:14px"><div class="spinner"></div>AI reviewing your code...</div>`;

  try {
    const review = await Gemini.reviewCode(p, userCode);
    showCodeReview(review, container);
  } catch (e) {
    if (container) container.innerHTML = `<div class="alert alert-danger">❌ ${e.message}</div>`;
  }
}

function showCodeReview(review, container) {
  if (!container) return;
  const correctColor = review.isCorrect ? 'var(--color-success-light)' : 'var(--color-warning-light)';
  container.innerHTML = `
  <div class="card animate-fade" style="${review.isCorrect ? 'border-color:rgba(16,185,129,0.3);background:rgba(16,185,129,0.05)' : 'border-color:rgba(245,158,11,0.3);background:rgba(245,158,11,0.05)'}">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="font-size:28px">${review.isCorrect ? '✅' : '⚠️'}</span>
      <div>
        <div style="font-weight:700;font-size:15px;color:${correctColor}">${review.isCorrect ? 'Looks Correct!' : 'Needs Work'}</div>
        <div style="font-size:12px;color:var(--text-secondary)">Time: ${review.timeComplexity} · Space: ${review.spaceComplexity}</div>
      </div>
    </div>
    <div style="font-size:13px;color:var(--text-secondary);line-height:1.6;margin-bottom:12px">${review.feedback}</div>
    ${(review.improvements||[]).length > 0 ? `
      <div style="font-size:12px;font-weight:700;color:var(--text-muted);margin-bottom:8px">Suggestions:</div>
      <ul style="list-style:none;display:flex;flex-direction:column;gap:4px">
        ${review.improvements.map(i => `<li style="font-size:13px;color:var(--text-secondary);display:flex;gap:8px"><span style="color:var(--color-accent-light)">→</span>${i}</li>`).join('')}
      </ul>
    ` : ''}
  </div>`;
}

function markSolved() {
  const p = codingState.currentProblem;
  Storage.markCodingSolved(p.id);
  showToast('🎉 Problem marked as solved!', 'success');
  renderPage('coding');
}

function clearCode() {
  const el = document.getElementById('user-code');
  if (el) el.value = '';
  const review = document.getElementById('code-review-result');
  if (review) review.innerHTML = '';
}
