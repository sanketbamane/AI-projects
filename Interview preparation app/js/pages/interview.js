// interview.js - Voice-primary mock interview with AI-generated questions

let interviewState = {
  active: false,
  role: '',
  difficulty: '',
  questions: [],
  answers: [],
  scores: [],
  currentQ: 0,
  totalQ: 5,
  recognition: null,
  synthesis: window.speechSynthesis,
  isListening: false,
  isSpeaking: false,
  transcript: '',
  evaluating: false,
  sessionComplete: false,
};

function renderInterview() {
  if (interviewState.active) return renderActiveInterview();
  if (interviewState.sessionComplete) return renderInterviewSummary();
  return renderInterviewSetup();
}

// ── SETUP ──────────────────────────────────────────────────────────────────

function renderInterviewSetup() {
  const roles = [
    { label: 'Frontend Dev', emoji: '🖥️', value: 'Frontend Developer' },
    { label: 'Backend Dev', emoji: '⚙️', value: 'Backend Developer' },
    { label: 'Full Stack', emoji: '🔄', value: 'Full Stack Developer' },
    { label: 'Data Science', emoji: '📊', value: 'Data Scientist' },
    { label: 'ML Engineer', emoji: '🤖', value: 'ML Engineer' },
    { label: 'DevOps', emoji: '🚀', value: 'DevOps Engineer' },
    { label: 'Product Mgr', emoji: '📋', value: 'Product Manager' },
    { label: 'iOS Dev', emoji: '📱', value: 'iOS Developer' },
    { label: 'SWE General', emoji: '💡', value: 'Software Engineer' },
  ];

  const savedRole = Storage.getTargetRole();

  return `
  <div class="animate-fade">
    <div class="page-header">
      <h1 class="page-title">🎤 Mock Interview</h1>
      <p class="page-subtitle">AI-powered voice interview with real-time question generation & feedback</p>
    </div>

    <div class="setup-card card card-gradient">
      <h2 style="font-size:17px;font-weight:700;margin-bottom:6px">Configure Your Interview</h2>
      <p style="font-size:13px;color:var(--text-secondary);margin-bottom:24px">Questions are generated dynamically by AI based on your role and difficulty.</p>

      <div class="form-group" style="margin-bottom:20px">
        <label class="form-label">Interview Role</label>
        <div class="role-grid" id="role-grid">
          ${roles.map(r => `
            <button class="role-btn ${savedRole === r.value ? 'selected' : ''}" onclick="selectRole(this, '${r.value}')">
              <span class="role-emoji">${r.emoji}</span>${r.label}
            </button>
          `).join('')}
        </div>
      </div>

      <div class="form-group" style="margin-bottom:20px">
        <label class="form-label">Difficulty Level</label>
        <div class="tabs">
          <button class="tab active" id="diff-junior" onclick="selectDifficulty(this,'Junior')">Junior</button>
          <button class="tab" id="diff-mid" onclick="selectDifficulty(this,'Mid-Level')">Mid-Level</button>
          <button class="tab" id="diff-senior" onclick="selectDifficulty(this,'Senior')">Senior</button>
        </div>
      </div>

      <div class="form-group" style="margin-bottom:24px">
        <label class="form-label">Number of Questions</label>
        <div class="tabs">
          <button class="tab active" id="qcount-5" onclick="selectQCount(this,5)">5</button>
          <button class="tab" id="qcount-8" onclick="selectQCount(this,8)">8</button>
          <button class="tab" id="qcount-10" onclick="selectQCount(this,10)">10</button>
        </div>
      </div>

      <div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:10px;padding:14px;margin-bottom:24px;display:flex;gap:12px;align-items:flex-start">
        <span style="font-size:18px">🎙️</span>
        <div>
          <div style="font-size:13px;font-weight:600;color:var(--color-accent-light);margin-bottom:4px">Voice Mode (Primary)</div>
          <div style="font-size:12px;color:var(--text-secondary);line-height:1.5">AI will speak each question aloud. You respond by speaking — your answer is transcribed in real-time. You can also type if preferred.</div>
        </div>
      </div>

      ${!Storage.getApiKey() ? `<div class="alert alert-warning" style="margin-bottom:16px">⚠️ No API key set. <a href="#" onclick="navigateTo('settings')" style="color:var(--color-warning-light);font-weight:600">Add one in Settings</a> to enable AI questions.</div>` : ''}

      <button class="btn btn-primary btn-lg" style="width:100%" onclick="startInterview()" id="start-interview-btn">
        🚀 Start Interview
      </button>
    </div>
  </div>`;
}

let selectedRole = Storage.getTargetRole() || 'Software Engineer';
let selectedDifficulty = 'Junior';
let selectedQCount = 5;

function selectRole(btn, role) {
  document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  selectedRole = role;
}

function selectDifficulty(btn, diff) {
  document.querySelectorAll('#diff-junior,#diff-mid,#diff-senior').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedDifficulty = diff;
}

function selectQCount(btn, count) {
  document.querySelectorAll('#qcount-5,#qcount-8,#qcount-10').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedQCount = count;
}

// ── START ──────────────────────────────────────────────────────────────────

async function startInterview() {
  if (!Storage.getApiKey()) {
    showToast('Please add a Gemini API key in Settings first.', 'error');
    return;
  }
  interviewState = {
    ...interviewState,
    active: true,
    sessionComplete: false,
    role: selectedRole,
    difficulty: selectedDifficulty,
    totalQ: selectedQCount,
    questions: [],
    answers: [],
    scores: [],
    currentQ: 0,
    transcript: '',
  };

  Storage.logActivity();
  renderPage('interview');
  await loadNextQuestion();
}

// ── ACTIVE INTERVIEW ───────────────────────────────────────────────────────

function renderActiveInterview() {
  const s = interviewState;
  const progress = Math.round((s.currentQ / s.totalQ) * 100);

  return `
  <div class="animate-fade">
    <div class="page-header" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
      <div>
        <h1 class="page-title">🎤 Live Interview</h1>
        <p class="page-subtitle">${s.role} · ${s.difficulty}</p>
      </div>
      <button class="btn btn-danger" onclick="endInterviewEarly()">End Session</button>
    </div>

    <!-- Progress -->
    <div class="card" style="margin-bottom:20px;padding:16px 20px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-size:13px;font-weight:600;color:var(--text-secondary)">Question ${s.currentQ + 1} of ${s.totalQ}</span>
        <span style="font-size:13px;font-weight:700;color:var(--color-primary-light)">${progress}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:${progress}%"></div></div>
    </div>

    <!-- Question Card -->
    <div class="card card-gradient" style="margin-bottom:20px;min-height:120px" id="question-card">
      <div style="display:flex;align-items:flex-start;gap:14px">
        <div style="width:40px;height:40px;border-radius:50%;background:var(--gradient-primary);display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0">🤖</div>
        <div style="flex:1">
          <div style="font-size:12px;font-weight:600;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">AI Interviewer</div>
          <div id="question-text" style="font-size:16px;line-height:1.7;color:var(--text-primary);min-height:60px">
            <div style="display:flex;gap:6px;align-items:center;color:var(--text-muted)">
              <div class="spinner"></div> Generating your question...
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Voice Controls -->
    <div class="card" style="margin-bottom:20px;text-align:center;padding:28px">
      <div id="voice-status" style="font-size:13px;color:var(--text-secondary);margin-bottom:16px">Click the mic to speak your answer</div>
      
      <div id="voice-animation" style="height:50px;display:flex;align-items:center;justify-content:center;margin-bottom:16px">
        <div style="color:var(--text-muted);font-size:13px">—</div>
      </div>

      <button class="voice-btn" id="mic-btn" onclick="toggleListening()" style="margin:0 auto 20px">
        🎤
      </button>

      <div id="transcript-box" style="
        background:var(--bg-input);
        border:1px solid var(--border);
        border-radius:var(--radius-sm);
        padding:14px;
        min-height:80px;
        font-size:14px;
        line-height:1.6;
        color:var(--text-primary);
        text-align:left;
        margin-bottom:16px;
        position:relative;
      ">
        <div id="transcript-text" style="color:var(--text-muted);font-style:italic">Your spoken answer will appear here...</div>
      </div>

      <div style="font-size:12px;color:var(--text-muted);margin-bottom:20px">
        Or type your answer below:
      </div>
      <textarea class="textarea" id="text-answer" placeholder="Type your answer here if you prefer text mode..." style="margin-bottom:16px" oninput="syncTextAnswer(this.value)"></textarea>

      <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
        <button class="btn btn-secondary" onclick="replayQuestion()">🔊 Replay Question</button>
        <button class="btn btn-primary btn-lg" id="submit-btn" onclick="submitAnswer()" disabled>
          Submit Answer →
        </button>
      </div>
    </div>

    <!-- Feedback (shown after each answer) -->
    <div id="feedback-section" style="display:none"></div>
  </div>`;
}

// ── QUESTION LOADING ───────────────────────────────────────────────────────

async function loadNextQuestion() {
  const s = interviewState;
  interviewState.transcript = '';
  interviewState.isListening = false;

  const qEl = document.getElementById('question-text');
  const submitBtn = document.getElementById('submit-btn');
  const transcriptEl = document.getElementById('transcript-text');
  const textAnswer = document.getElementById('text-answer');
  const feedbackSection = document.getElementById('feedback-section');

  if (qEl) qEl.innerHTML = `<div style="display:flex;gap:6px;align-items:center;color:var(--text-muted)"><div class="spinner"></div> Generating question...</div>`;
  if (submitBtn) submitBtn.disabled = true;
  if (transcriptEl) { transcriptEl.style.color = 'var(--text-muted)'; transcriptEl.style.fontStyle = 'italic'; transcriptEl.textContent = 'Your spoken answer will appear here...'; }
  if (textAnswer) textAnswer.value = '';
  if (feedbackSection) { feedbackSection.style.display = 'none'; feedbackSection.innerHTML = ''; }

  try {
    const question = await Gemini.generateInterviewQuestion(
      s.role, s.difficulty, s.currentQ + 1, s.questions
    );
    interviewState.questions.push(question);

    if (qEl) qEl.textContent = question;
    speakText(question);
  } catch (e) {
    if (qEl) qEl.innerHTML = `<span style="color:var(--color-danger-light)">⚠️ Could not generate question: ${e.message}</span>`;
  }
}

// ── VOICE CONTROL ──────────────────────────────────────────────────────────

function speakText(text) {
  const synth = interviewState.synthesis;
  if (!synth) return;
  synth.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 0.9;
  utt.pitch = 1;
  utt.volume = 1;

  const voices = synth.getVoices();
  const preferred = voices.find(v => v.name.includes('Google') && v.lang.startsWith('en'));
  if (preferred) utt.voice = preferred;

  interviewState.isSpeaking = true;
  updateVoiceStatus('🔊 AI is speaking...', false);

  utt.onend = () => {
    interviewState.isSpeaking = false;
    updateVoiceStatus('Click the mic to record your answer', false);
    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn && !interviewState.transcript && !document.getElementById('text-answer')?.value) {
      // ready to record
    }
  };

  synth.speak(utt);
}

function toggleListening() {
  if (interviewState.isListening) {
    stopListening();
  } else {
    startListening();
  }
}

function startListening() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showToast('Speech recognition not supported in this browser. Use Chrome or Edge.', 'error');
    return;
  }

  interviewState.synthesis?.cancel(); // stop speaking if still going

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  interviewState.recognition = recognition;

  recognition.onstart = () => {
    interviewState.isListening = true;
    const btn = document.getElementById('mic-btn');
    if (btn) { btn.classList.add('recording'); btn.textContent = '⏹️'; }
    updateVoiceStatus('🔴 Listening... speak your answer', true);
    showVoiceBars();
  };

  recognition.onresult = (e) => {
    let interim = '';
    let final = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) final += e.results[i][0].transcript + ' ';
      else interim += e.results[i][0].transcript;
    }
    if (final) interviewState.transcript += final;
    const display = interviewState.transcript + interim;
    const el = document.getElementById('transcript-text');
    if (el) {
      el.style.color = 'var(--text-primary)';
      el.style.fontStyle = 'normal';
      el.textContent = display || '...';
    }
    // sync to text area too
    const ta = document.getElementById('text-answer');
    if (ta) ta.value = display;

    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn && display.trim().length > 5) submitBtn.disabled = false;
  };

  recognition.onerror = (e) => {
    if (e.error !== 'aborted') showToast('Mic error: ' + e.error, 'error');
    stopListening();
  };

  recognition.onend = () => {
    if (interviewState.isListening) recognition.start(); // keep going
  };

  recognition.start();
}

function stopListening() {
  interviewState.isListening = false;
  if (interviewState.recognition) {
    interviewState.recognition.onend = null;
    interviewState.recognition.stop();
    interviewState.recognition = null;
  }
  const btn = document.getElementById('mic-btn');
  if (btn) { btn.classList.remove('recording'); btn.textContent = '🎤'; }
  updateVoiceStatus('Recording stopped. Submit or re-record your answer.', false);
  hideVoiceBars();

  const submitBtn = document.getElementById('submit-btn');
  const answer = interviewState.transcript || document.getElementById('text-answer')?.value || '';
  if (submitBtn) submitBtn.disabled = answer.trim().length < 3;
}

function updateVoiceStatus(msg, listening) {
  const el = document.getElementById('voice-status');
  if (el) {
    el.textContent = msg;
    el.style.color = listening ? 'var(--color-danger-light)' : 'var(--text-secondary)';
  }
}

function showVoiceBars() {
  const el = document.getElementById('voice-animation');
  if (el) el.innerHTML = `<div class="voice-indicator">${Array.from({length:5}, (_,i) => `<div class="voice-bar"></div>`).join('')}</div>`;
}

function hideVoiceBars() {
  const el = document.getElementById('voice-animation');
  if (el) el.innerHTML = `<div style="color:var(--text-muted);font-size:13px">—</div>`;
}

function syncTextAnswer(val) {
  interviewState.transcript = val;
  const el = document.getElementById('transcript-text');
  if (el && val) { el.style.color = 'var(--text-primary)'; el.style.fontStyle = 'normal'; el.textContent = val; }
  const submitBtn = document.getElementById('submit-btn');
  if (submitBtn) submitBtn.disabled = val.trim().length < 3;
}

function replayQuestion() {
  const q = interviewState.questions[interviewState.currentQ];
  if (q) speakText(q);
}

// ── SUBMIT ANSWER ──────────────────────────────────────────────────────────

async function submitAnswer() {
  if (interviewState.evaluating) return;
  stopListening();

  const answer = interviewState.transcript?.trim() || document.getElementById('text-answer')?.value?.trim() || '';
  if (!answer || answer.length < 3) {
    showToast('Please provide an answer first', 'error');
    return;
  }

  interviewState.evaluating = true;
  const submitBtn = document.getElementById('submit-btn');
  if (submitBtn) { submitBtn.disabled = true; submitBtn.innerHTML = `<div class="spinner"></div> Evaluating...`; }

  interviewState.answers.push(answer);

  try {
    const evaluation = await Gemini.evaluateAnswer(
      interviewState.questions[interviewState.currentQ],
      answer,
      interviewState.role,
      interviewState.difficulty
    );
    interviewState.scores.push(evaluation.score);
    showFeedback(evaluation);
  } catch (e) {
    showToast('Could not evaluate: ' + e.message, 'error');
    interviewState.scores.push(5);
    showFeedback({ score: 5, verdict: 'Good', strengths: ['Answer recorded'], improvements: ['Keep practicing'], betterAnswer: answer });
  }

  interviewState.evaluating = false;
}

function showFeedback(evaluation) {
  const scoreColor = evaluation.score >= 8 ? 'var(--color-success-light)' : evaluation.score >= 6 ? 'var(--color-accent-light)' : 'var(--color-warning-light)';
  const feedbackSection = document.getElementById('feedback-section');
  if (!feedbackSection) return;

  feedbackSection.style.display = 'block';
  feedbackSection.innerHTML = `
    <div class="card animate-fade" style="border-color:rgba(124,58,237,0.3);background:rgba(124,58,237,0.05)">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap">
        <div style="text-align:center">
          <div style="font-size:40px;font-weight:900;color:${scoreColor};line-height:1">${evaluation.score}</div>
          <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px">/ 10</div>
        </div>
        <div>
          <div style="font-size:18px;font-weight:800;color:${scoreColor}">${evaluation.verdict}</div>
          <div style="font-size:13px;color:var(--text-secondary)">AI Feedback</div>
        </div>
      </div>

      <div class="grid-2" style="margin-bottom:16px;gap:12px">
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:14px">
          <div style="font-size:12px;font-weight:700;color:var(--color-success-light);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">✅ Strengths</div>
          <ul style="list-style:none;display:flex;flex-direction:column;gap:6px">
            ${(evaluation.strengths || []).map(s => `<li style="font-size:13px;color:var(--text-secondary);display:flex;gap:8px"><span style="color:var(--color-success-light)">•</span>${s}</li>`).join('')}
          </ul>
        </div>
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:14px">
          <div style="font-size:12px;font-weight:700;color:var(--color-warning-light);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">💡 Improve</div>
          <ul style="list-style:none;display:flex;flex-direction:column;gap:6px">
            ${(evaluation.improvements || []).map(i => `<li style="font-size:13px;color:var(--text-secondary);display:flex;gap:8px"><span style="color:var(--color-warning-light)">•</span>${i}</li>`).join('')}
          </ul>
        </div>
      </div>

      <div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:10px;padding:14px;margin-bottom:20px">
        <div style="font-size:12px;font-weight:700;color:var(--color-accent-light);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">🎯 Ideal Answer</div>
        <div style="font-size:13px;color:var(--text-secondary);line-height:1.6">${evaluation.betterAnswer}</div>
      </div>

      <div style="display:flex;justify-content:flex-end">
        ${interviewState.currentQ + 1 >= interviewState.totalQ
          ? `<button class="btn btn-success btn-lg" onclick="finishInterview()">View Summary →</button>`
          : `<button class="btn btn-primary btn-lg" onclick="nextQuestion()">Next Question →</button>`
        }
      </div>
    </div>`;

  feedbackSection.scrollIntoView({ behavior: 'smooth' });
}

async function nextQuestion() {
  interviewState.currentQ++;
  interviewState.transcript = '';
  renderPage('interview');
  await loadNextQuestion();
}

// ── FINISH ─────────────────────────────────────────────────────────────────

async function finishInterview() {
  const s = interviewState;
  interviewState.synthesis?.cancel();

  const summaryEl = document.getElementById('main-content');
  if (summaryEl) summaryEl.innerHTML = `<div class="animate-fade" style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;gap:20px"><div class="spinner" style="width:40px;height:40px;border-width:4px"></div><p style="color:var(--text-secondary)">Generating your session summary...</p></div>`;

  const qaPairs = s.questions.map((q, i) => ({
    question: q,
    answer: s.answers[i] || '',
    score: s.scores[i] || 5,
  }));

  let summary;
  try {
    summary = await Gemini.generateInterviewSummary(s.role, qaPairs);
  } catch {
    const avg = qaPairs.reduce((t, p) => t + p.score, 0) / qaPairs.length;
    summary = { overallScore: Math.round(avg * 10) / 10, summary: 'Interview complete!', topStrength: 'Completed the session', keySuggestion: 'Keep practicing regularly.' };
  }

  Storage.addInterviewSession({
    role: s.role,
    difficulty: s.difficulty,
    questionCount: s.questions.length,
    score: summary.overallScore,
    date: new Date().toISOString(),
  });

  interviewState.active = false;
  interviewState.sessionComplete = true;
  interviewState.summary = summary;
  interviewState.qaPairs = qaPairs;

  renderPage('interview');
}

function endInterviewEarly() {
  if (!confirm('End this interview session?')) return;
  interviewState.synthesis?.cancel();
  stopListening();
  interviewState.active = false;
  interviewState.sessionComplete = false;
  renderPage('interview');
}

// ── SUMMARY ────────────────────────────────────────────────────────────────

function renderInterviewSummary() {
  const s = interviewState;
  const summary = s.summary || {};
  const qaPairs = s.qaPairs || [];
  const score = summary.overallScore || 0;
  const scoreColor = score >= 8 ? 'var(--color-success-light)' : score >= 6 ? 'var(--color-accent-light)' : 'var(--color-warning-light)';
  const stars = score >= 9 ? '⭐⭐⭐⭐⭐' : score >= 7 ? '⭐⭐⭐⭐' : score >= 5 ? '⭐⭐⭐' : '⭐⭐';

  return `
  <div class="animate-fade">
    <div class="page-header">
      <h1 class="page-title">📊 Interview Summary</h1>
      <p class="page-subtitle">${s.role} · ${s.difficulty}</p>
    </div>

    <!-- Score -->
    <div class="card card-gradient" style="text-align:center;padding:40px;margin-bottom:24px">
      <div class="score-display" style="padding:0">
        <div class="score-number" style="font-size:80px">${score}</div>
        <div class="score-stars">${stars}</div>
        <div style="font-size:14px;color:var(--text-secondary);max-width:400px;margin:12px auto 0;line-height:1.6">${summary.summary}</div>
      </div>
    </div>

    <div class="grid-2" style="margin-bottom:24px">
      <div class="card" style="background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.2)">
        <div style="font-size:12px;font-weight:700;color:var(--color-success-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">🏆 Top Strength</div>
        <div style="font-size:15px;color:var(--text-primary);line-height:1.5">${summary.topStrength}</div>
      </div>
      <div class="card" style="background:rgba(245,158,11,0.06);border-color:rgba(245,158,11,0.2)">
        <div style="font-size:12px;font-weight:700;color:var(--color-warning-light);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">💡 Work On</div>
        <div style="font-size:15px;color:var(--text-primary);line-height:1.5">${summary.keySuggestion}</div>
      </div>
    </div>

    <!-- Q&A Breakdown -->
    <div style="margin-bottom:24px">
      <div class="section-header"><h2 class="section-title">Question Breakdown</h2></div>
      <div style="display:flex;flex-direction:column;gap:12px">
        ${qaPairs.map((pair, i) => {
          const sc = pair.score;
          const c = sc >= 8 ? 'var(--color-success-light)' : sc >= 6 ? 'var(--color-accent-light)' : 'var(--color-warning-light)';
          return `
            <div class="card" style="padding:16px 20px">
              <div style="display:flex;gap:14px;align-items:flex-start">
                <div style="width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:rgba(124,58,237,0.15);font-size:13px;font-weight:700;flex-shrink:0;color:var(--color-primary-light)">Q${i+1}</div>
                <div style="flex:1">
                  <div style="font-size:14px;font-weight:600;color:var(--text-primary);margin-bottom:6px">${pair.question}</div>
                  <div style="font-size:13px;color:var(--text-secondary);line-height:1.5;border-left:2px solid var(--border);padding-left:10px;margin-bottom:4px">${pair.answer || '(no answer)'}</div>
                </div>
                <div style="text-align:center;flex-shrink:0">
                  <div style="font-size:22px;font-weight:800;color:${c};line-height:1">${sc}</div>
                  <div style="font-size:10px;color:var(--text-muted)">/10</div>
                </div>
              </div>
            </div>`;
        }).join('')}
      </div>
    </div>

    <div style="display:flex;gap:12px;flex-wrap:wrap">
      <button class="btn btn-primary btn-lg" onclick="restartInterview()">🔄 Try Again</button>
      <button class="btn btn-secondary btn-lg" onclick="navigateTo('progress')">📊 View Progress</button>
      <button class="btn btn-secondary btn-lg" onclick="navigateTo('dashboard')">🏠 Dashboard</button>
    </div>
  </div>`;
}

function restartInterview() {
  interviewState.active = false;
  interviewState.sessionComplete = false;
  renderPage('interview');
}
