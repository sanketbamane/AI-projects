// storage.js - localStorage utility for all app persistence

const STORAGE_KEYS = {
  API_KEY: 'prepai_gemini_key',
  USER_NAME: 'prepai_user_name',
  TARGET_ROLE: 'prepai_target_role',
  PROGRESS: 'prepai_progress',
  FLASHCARD_KNOWN: 'prepai_flashcard_known',
  CODING_SOLVED: 'prepai_coding_solved',
  INTERVIEW_HISTORY: 'prepai_interview_history',
  STREAK: 'prepai_streak',
  ACTIVITY: 'prepai_activity',
};

const Storage = {
  get(key, fallback = null) {
    try {
      const val = localStorage.getItem(key);
      return val ? JSON.parse(val) : fallback;
    } catch { return fallback; }
  },

  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { }
  },

  remove(key) {
    try { localStorage.removeItem(key); } catch { }
  },

  getApiKey() { return localStorage.getItem(STORAGE_KEYS.API_KEY) || ''; },
  setApiKey(key) { localStorage.setItem(STORAGE_KEYS.API_KEY, key); },

  getUserName() { return localStorage.getItem(STORAGE_KEYS.USER_NAME) || 'there'; },
  setUserName(name) { localStorage.setItem(STORAGE_KEYS.USER_NAME, name); },

  getTargetRole() { return localStorage.getItem(STORAGE_KEYS.TARGET_ROLE) || 'Software Engineer'; },
  setTargetRole(role) { localStorage.setItem(STORAGE_KEYS.TARGET_ROLE, role); },

  getProgress() {
    return this.get(STORAGE_KEYS.PROGRESS, {
      mockInterviews: 0,
      totalScore: 0,
      flashcardsStudied: 0,
      flashcardsKnown: 0,
      codingSolved: 0,
      streak: 0,
      lastActive: null,
    });
  },

  updateProgress(updates) {
    const current = this.getProgress();
    this.set(STORAGE_KEYS.PROGRESS, { ...current, ...updates });
  },

  getFlashcardKnown() { return this.get(STORAGE_KEYS.FLASHCARD_KNOWN, []); },
  markFlashcardKnown(id) {
    const known = this.getFlashcardKnown();
    if (!known.includes(id)) {
      known.push(id);
      this.set(STORAGE_KEYS.FLASHCARD_KNOWN, known);
    }
  },

  getCodingSolved() { return this.get(STORAGE_KEYS.CODING_SOLVED, []); },
  markCodingSolved(id) {
    const solved = this.getCodingSolved();
    if (!solved.includes(id)) {
      solved.push(id);
      this.set(STORAGE_KEYS.CODING_SOLVED, solved);
      this.updateProgress({ codingSolved: solved.length });
    }
  },

  getInterviewHistory() { return this.get(STORAGE_KEYS.INTERVIEW_HISTORY, []); },
  addInterviewSession(session) {
    const history = this.getInterviewHistory();
    history.unshift(session); // latest first
    if (history.length > 20) history.pop(); // keep last 20
    this.set(STORAGE_KEYS.INTERVIEW_HISTORY, history);
    const p = this.getProgress();
    this.updateProgress({
      mockInterviews: p.mockInterviews + 1,
      totalScore: p.totalScore + session.score,
    });
  },

  // Activity heatmap — track days used
  logActivity() {
    const today = new Date().toISOString().split('T')[0];
    const activity = this.get(STORAGE_KEYS.ACTIVITY, {});
    activity[today] = (activity[today] || 0) + 1;
    this.set(STORAGE_KEYS.ACTIVITY, activity);

    // Update streak
    this.updateStreak(today);
  },

  updateStreak(today) {
    const streak = this.get(STORAGE_KEYS.STREAK, { count: 0, lastDay: null });
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
    if (streak.lastDay === today) return; // already updated today
    if (streak.lastDay === yesterday) {
      streak.count++;
    } else if (streak.lastDay !== today) {
      streak.count = 1;
    }
    streak.lastDay = today;
    this.set(STORAGE_KEYS.STREAK, streak);
    this.updateProgress({ streak: streak.count });
  },

  getStreak() { return this.get(STORAGE_KEYS.STREAK, { count: 0, lastDay: null }); },
  getActivity() { return this.get(STORAGE_KEYS.ACTIVITY, {}); },
};
