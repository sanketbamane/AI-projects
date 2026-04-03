// app.js - Main router and app controller

// ── ROUTER ─────────────────────────────────────────────────────────────────

const PAGES = {
  dashboard: { render: renderDashboard, navId: 'nav-dashboard' },
  interview:  { render: renderInterview,  navId: 'nav-interview' },
  flashcards: { render: renderFlashcards, navId: 'nav-flashcards' },
  resume:     { render: renderResume,     navId: 'nav-resume' },
  coding:     { render: renderCoding,     navId: 'nav-coding' },
  progress:   { render: renderProgress,   navId: 'nav-progress' },
  settings:   { render: renderSettings,   navId: 'nav-settings' },
};

let currentPage = 'dashboard';

function navigateTo(page) {
  if (!PAGES[page]) return;
  currentPage = page;
  renderPage(page);
  updateNav(page);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderPage(page) {
  const main = document.getElementById('main-content');
  if (!main) return;
  const pageObj = PAGES[page];
  if (!pageObj) return;
  main.innerHTML = pageObj.render();
  updateNav(page);
}

function updateNav(page) {
  // Sidebar
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.page === page);
  });
  // Bottom nav
  document.querySelectorAll('.bottom-nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
  });
}

// ── NAV CLICK BINDING ───────────────────────────────────────────────────────

function bindNavLinks() {
  document.querySelectorAll('[data-page]').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      const page = el.dataset.page;
      if (page) navigateTo(page);
    });
  });
}

// ── TOAST ───────────────────────────────────────────────────────────────────

function showToast(message, type = 'success') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
    <span>${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(30px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ── SPEECH SYNTHESIS VOICES ─────────────────────────────────────────────────

// Ensure voices are loaded (Chrome loads async)
if (window.speechSynthesis) {
  window.speechSynthesis.getVoices();
  window.speechSynthesis.onvoiceschanged = () => {};
}

// ── INIT ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  bindNavLinks();
  Storage.logActivity(); // track today's activity

  // Route to settings if no API key yet
  const startPage = !Storage.getApiKey() ? 'dashboard' : 'dashboard';
  navigateTo(startPage);
});
