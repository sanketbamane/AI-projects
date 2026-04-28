require('dotenv').config();

const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const { spawn } = require('child_process');

const fs = require('fs');
const os = require('os');
const path = require('path');
const screenshot = require('screenshot-desktop');

// Helper function to check if running in dev mode
function isDevelopment() {
  return !app.isPackaged;
}

// Helper function to get the correct path based on environment
function getAppPath() {
  return isDevelopment() ? __dirname : path.join(process.resourcesPath, 'app.asar');
}

// Load .env from the correct location (handles both dev and production)
// Do this after app is ready
// Safe way to check if running in dev mode (process.defaultApp is set by Electron when launched via CLI)
const isDev = !!process.defaultApp || process.env.NODE_ENV === 'development';
let envPath;

if (isDev) {
  envPath = path.join(__dirname, '..', '.env');
} else {
  envPath = path.join(process.resourcesPath, '.env');
}

require('dotenv').config({ path: envPath });

console.log('Loaded .env from:', envPath);
console.log('GROQ_API_KEY:', process.env.GROQ_API_KEY?.trim() ? '✅ Found' : '❌ Missing');
console.log('GEMINI_API_KEY:', process.env.GEMINI_API_KEY?.trim() ? '✅ Found' : '❌ Missing');

require('dotenv').config({ path: envPath });

const GeminiService = require('./gemini-service');
const GroqService = require('./groq-service');
const ApiFreeLLMService = require('./apifreellm-service');

function hasAiApiKey() {
  return !!(
    String(process.env.GROQ_API_KEY || '').trim() ||
    String(process.env.GEMINI_API_KEY || '').trim() ||
    String(process.env.APIFREELLM_API_KEY || '').trim()
  );
}

/** Speech-to-text: groq = Groq Whisper API (needs GROQ_API_KEY), vosk = local Python Vosk */
function resolveSttEngine() {
  const explicit = String(process.env.STT_ENGINE || '').trim().toLowerCase();
  const hasGroqKey = !!String(process.env.GROQ_API_KEY || '').trim();
  
  if (explicit === 'google') {
    return { engine: 'google' };
  }
  if (explicit === 'vosk') {
    return { engine: 'vosk' };
  }
  if (explicit === 'groq') {
    if (!hasGroqKey) {
      return {
        engine: 'google',
        warning: 'STT_ENGINE=groq requires GROQ_API_KEY. Using Google Speech instead.'
      };
    }
    return { engine: 'groq' };
  }
  
  // Default logic: Vosk (local) > Google > Groq (if key)
  return { engine: 'vosk' };
}

/** Hide subprocess consoles on Windows so Python/Vosk does not appear in the taskbar. */
function spawnWinHidden(command, args, options = {}) {
  const opts = { ...options };
  if (process.platform === 'win32') {
    opts.windowsHide = true;
  }
  return spawn(command, args, opts);
}

// Set App User Model ID early (before app ready) to prevent taskbar icon
if (process.platform === 'win32') {
  app.setAppUserModelId('com.microsoft.systemprocess');
}

(async () => {

  let mainWindow;
  let screenshots = [];
  let chatContext = [];
  const MAX_SCREENSHOTS = 3;

  // Vosk live transcription process
  let voskProcess = null;
  let isVoskRunning = false;
  let voskStdoutBuffer = '';

  function voskStdinWrite(cmd) {
    if (!voskProcess || voskProcess.killed) return false;
    try {
      voskProcess.stdin.write(`${cmd}\n`);
      return true;
    } catch (e) {
      console.error('Vosk stdin write failed:', e.message);
      return false;
    }
  }

  function dispatchVoskJsonLine(line) {
    try {
      const result = JSON.parse(line);
      if (!mainWindow || mainWindow.isDestroyed()) return;

      switch (result.type) {
        case 'status':
          console.log(`Vosk status: ${result.status} - ${result.message}`);
          mainWindow.webContents.send('vosk-status', result);
          break;

        case 'partial':
          mainWindow.webContents.send('vosk-partial', { text: result.text });
          break;

        case 'final':
          console.log('Vosk transcription:', result.text);
          mainWindow.webContents.send('vosk-final', { text: result.text });
          if (geminiService && result.text) {
            geminiService.addToHistory('user', result.text);
          }
          break;

        case 'error':
          console.error('Vosk error:', result.error);
          mainWindow.webContents.send('vosk-error', { error: result.error });
          break;

        default:
          break;
      }
    } catch (parseError) {
      console.error('Failed to parse Vosk JSON line:', line.substring(0, 200), parseError.message);
    }
  }

  function appendVoskStdout(chunk) {
    voskStdoutBuffer += chunk.toString();
    const lines = voskStdoutBuffer.split('\n');
    voskStdoutBuffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) dispatchVoskJsonLine(trimmed);
    }
  }

  // LLM: Groq (GROQ_API_KEY) preferred, else Google Gemini (GEMINI_API_KEY)
  let geminiService = null;

  try {
    const groqKey = String(process.env.GROQ_API_KEY || '').trim();
    const geminiKey = String(process.env.GEMINI_API_KEY || '').trim();
    const apifreeKey = String(process.env.APIFREELLM_API_KEY || '').trim();

    if (groqKey) {
      console.log('Initializing Groq AI...');
      geminiService = new GroqService(groqKey);
      console.log('Groq AI initialized successfully');
    } else if (apifreeKey || geminiKey.startsWith('apf_')) {
      const finalKey = apifreeKey || geminiKey;
      console.log('Initializing ApiFreeLLM Service...');
      geminiService = new ApiFreeLLMService(finalKey);
      console.log('ApiFreeLLM Service initialized successfully');
    } else if (geminiKey) {
      console.log('Initializing Gemini AI Service...');
      geminiService = new GeminiService(geminiKey);
      console.log('Gemini AI Service initialized successfully');
    } else {
      console.error('No GROQ_API_KEY, GEMINI_API_KEY, or APIFREELLM_API_KEY in environment variables');
    }
  } catch (error) {
    console.error('Failed to initialize AI service:', error);
  }

  const _sttResolved = resolveSttEngine();
  console.log('Speech-to-text engine:', _sttResolved.engine, _sttResolved.warning || '');

  function createStealthWindow() {
    console.log('Creating stealth window...');
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;

    // Short and wide window dimensions (resizable)
    const windowWidth = 900;
    const windowHeight = 400;
    const x = Math.floor((width - windowWidth) / 2);
    const y = 40;

    console.log(`Window position: ${x}, ${y}, size: ${windowWidth}x${windowHeight}`);

    // Windows: `type: 'toolbar'` creates a tool window (WS_EX_TOOLWINDOW) so the
    // UI stays visible but no taskbar button is shown. skipTaskbar alone is often
    // not enough for Electron on Windows.
    mainWindow = new BrowserWindow({
      ...(process.platform === 'win32' ? { type: 'toolbar' } : {}),
      width: windowWidth,
      height: windowHeight,
      minWidth: 400,
      minHeight: 180,
      maxWidth: width,
      maxHeight: height,
      x: x,
      y: y,
      webPreferences: {
        nodeIntegration: false,          // Disable for security
        contextIsolation: true,          // Enable for security
        preload: path.join(__dirname, 'preload.js'),
        backgroundThrottling: false,
        offscreen: false,
        webSecurity: false,              // CHANGED: Disable for microphone access
        allowRunningInsecureContent: true, // CHANGED: Allow for media access
        experimentalFeatures: false,
        enableRemoteModule: false,
        sandbox: false                   // Keep disabled for dynamic imports
      },
      frame: false,
      transparent: true,
      alwaysOnTop: true,
      skipTaskbar: true,
      resizable: true,                   // CHANGED: Enable resizing
      minimizable: false,
      maximizable: false,
      closable: true,
      focusable: true,
      show: false,
      opacity: 1.0,
      acceptFirstMouse: false,
      disableAutoHideCursor: true,
      enableLargerThanScreen: false,
      hasShadow: false,
      thickFrame: false,
      titleBarStyle: 'hidden',
      backgroundColor: '#00000000'
    });

    console.log('BrowserWindow created');

    const htmlPath = path.join(__dirname, 'renderer.html');
    console.log('Loading HTML from:', htmlPath);
    mainWindow.loadFile(htmlPath);

    // ADDED: Set up microphone permissions
    mainWindow.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
      console.log('Permission requested:', permission);
      if (permission === 'microphone' || permission === 'media') {
        console.log('Granting microphone permission');
        callback(true);
      } else {
        console.log('Denying permission:', permission);
        callback(false);
      }
    });

    // ADDED: Set permissions policy for media access
    mainWindow.webContents.session.setPermissionCheckHandler((webContents, permission, requestingOrigin, details) => {
      console.log('Permission check:', permission, requestingOrigin);
      if (permission === 'microphone' || permission === 'media') {
        return true;
      }
      return false;
    });

    // ADDED: Override permissions for media devices
    mainWindow.webContents.session.protocol.registerFileProtocol('file', (request, callback) => {
      const pathname = decodeURI(request.url.replace('file:///', ''));
      callback(pathname);
    });

    // Apply stealth settings
    if (process.platform === 'darwin') {
      mainWindow.setVisibleOnAllWorkspaces(true, {
        visibleOnFullScreen: true,
        skipTransformProcessType: true
      });
      mainWindow.setAlwaysOnTop(true, 'pop-up-menu', 1);
      app.dock.hide();
      mainWindow.setHiddenInMissionControl(true);
    } else if (process.platform === 'win32') {
      console.log('Applying Windows stealth settings');
      // Force skip taskbar - set it multiple times to ensure it sticks
      mainWindow.setSkipTaskbar(true);
      mainWindow.setAlwaysOnTop(true, 'screen-saver');
      // Set a fake app user model ID so Windows groups it separately and hides it
      app.setAppUserModelId('com.microsoft.systemprocess');
      mainWindow.setAppDetails({
        appId: 'com.microsoft.systemprocess',
        appIconPath: '',
        relaunchCommand: '',
        relaunchDisplayName: ''
      });
    }

    mainWindow.setContentProtection(true);
    console.log('Content protection enabled for stealth');

    mainWindow.setIgnoreMouseEvents(false);

    mainWindow.webContents.on('dom-ready', () => {
      console.log('DOM is ready');
    });

    mainWindow.webContents.on('did-finish-load', () => {
      console.log('HTML finished loading');

      mainWindow.webContents.executeJavaScript(`
      console.log('Content check...');
      console.log('Document title:', document.title);
      console.log('Body exists:', !!document.body);
      console.log('App element exists:', !!document.getElementById('app'));
      console.log('Glass container exists:', !!document.querySelector('.glass-container'));
      
      document.body.style.background = 'transparent';
      
      if (document.body) {
        document.body.style.visibility = 'visible';
        document.body.style.display = 'block';
        console.log('Body made visible');
      }
      
      const app = document.getElementById('app');
      if (app) {
        app.style.visibility = 'visible';
        app.style.display = 'flex';
        console.log('App container made visible (flex preserved)');
      }
      
      'Content visibility check complete';
    `).then((result) => {
        console.log('JavaScript result:', result);
        mainWindow.show();
        mainWindow.focus();
        console.log('Window shown with transparent background');
      }).catch((error) => {
        console.log('JavaScript execution failed:', error);
        mainWindow.show();
      });
    });

    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
      console.error('Failed to load:', errorCode, errorDescription);
    });

    // Handle console messages from renderer
    mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
      console.log(`Renderer console.${level}: ${message}`);
    });

    if (process.env.NODE_ENV === 'development') {
      mainWindow.webContents.openDevTools();
    }
  }

  function registerStealthShortcuts() {
    globalShortcut.register('CommandOrControl+Alt+Shift+H', () => {
      toggleStealthMode();
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+S', async () => {
      await takeStealthScreenshot();
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+A', async () => {
      if (screenshots.length > 0) {
        await analyzeForMeeting();
      }
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+X', () => {
      emergencyHide();
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+V', () => {
      mainWindow.webContents.send('toggle-voice-recognition');
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+Left', () => {
      moveToPosition('left');
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+Right', () => {
      moveToPosition('right');
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+Up', () => {
      moveToPosition('top');
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+Down', () => {
      moveToPosition('bottom');
    });

    globalShortcut.register('CommandOrControl+Alt+Shift+R', () => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        // Trigger resize cycle — reuse the IPC handler logic inline
        const { width: workW, height: workH } = screen.getPrimaryDisplay().workAreaSize;
        const PRESETS = [
          { label: 'small',      width: 680,  height: 280 },
          { label: 'medium',     width: 900,  height: 420 },
          { label: 'large',      width: 1100, height: 620 },
          { label: 'fullscreen', width: workW, height: workH }
        ];
        const [curW, curH] = mainWindow.getSize();
        const curIdx = PRESETS.findIndex(p =>
          Math.abs(curW - p.width) < 30 && Math.abs(curH - p.height) < 30
        );
        const next = PRESETS[(curIdx + 1) % PRESETS.length];
        const newW = Math.min(next.width, workW);
        const newH = Math.min(next.height, workH);
        const [, curY] = mainWindow.getPosition();
        const newX = Math.floor((workW - newW) / 2);
        mainWindow.setBounds({ x: newX, y: Math.max(curY, 20), width: newW, height: newH }, true);
        mainWindow.webContents.send('window-resized', next.label);
        console.log(`Global shortcut resize: ${next.label} ${newW}x${newH}`);
      }
    });
  }

  let isVisible = true;
  let autoHideTimer = null;

  function toggleStealthMode() {
    if (autoHideTimer) {
      clearTimeout(autoHideTimer);
      autoHideTimer = null;
    }

    if (isVisible) {
      mainWindow.setOpacity(0.6);
      mainWindow.webContents.send('set-stealth-mode', true);
      isVisible = false;
    } else {
      mainWindow.setOpacity(1.0);
      mainWindow.webContents.send('set-stealth-mode', false);
      isVisible = true;
    }
  }

  function emergencyHide() {
    if (autoHideTimer) {
      clearTimeout(autoHideTimer);
      autoHideTimer = null;
    }

    mainWindow.setOpacity(0.01);
    mainWindow.webContents.send('emergency-clear');

    autoHideTimer = setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.setOpacity(1.0);
        isVisible = true;
      }
      autoHideTimer = null;
    }, 2000);
  }

  function moveToPosition(position) {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;
    const windowBounds = mainWindow.getBounds();

    let x, y;

    switch (position) {
      case 'left':
        x = 20;
        y = windowBounds.y;
        break;
      case 'right':
        x = width - windowBounds.width - 20;
        y = windowBounds.y;
        break;
      case 'top':
        x = Math.floor((width - windowBounds.width) / 2);
        y = 40;
        break;
      case 'bottom':
        x = Math.floor((width - windowBounds.width) / 2);
        y = height - windowBounds.height - 40;
        break;
      default:
        return;
    }

    mainWindow.setPosition(x, y);
  }

  async function takeStealthScreenshot() {
    try {
      console.log('Taking stealth screenshot...');
      const currentOpacity = mainWindow.getOpacity();

      mainWindow.setOpacity(0.01);

      await new Promise(resolve => setTimeout(resolve, 200));

      // Use app data directory for screenshots in production
      const screenshotsDir = isDevelopment()
        ? path.join(__dirname, '..', '.stealth_screenshots')
        : path.join(app.getPath('userData'), '.stealth_screenshots');

      if (!fs.existsSync(screenshotsDir)) {
        fs.mkdirSync(screenshotsDir, { recursive: true });
      }

      const screenshotPath = path.join(screenshotsDir, `stealth-${Date.now()}.png`);
      await screenshot({ filename: screenshotPath });

      screenshots.push(screenshotPath);
      if (screenshots.length > MAX_SCREENSHOTS) {
        const oldPath = screenshots.shift();
        if (fs.existsSync(oldPath)) {
          fs.unlinkSync(oldPath);
        }
      }

      mainWindow.setOpacity(currentOpacity);

      console.log(`Screenshot saved: ${screenshotPath}`);
      console.log(`Total screenshots: ${screenshots.length}`);

      mainWindow.webContents.send('screenshot-taken-stealth', screenshots.length);

      return screenshotPath;
    } catch (error) {
      mainWindow.setOpacity(1.0);
      console.error('Stealth screenshot error:', error);
      throw error;
    }
  }

  async function analyzeForMeetingWithContext(context = '') {
    console.log('Starting context-aware analysis...');
    console.log('Context length:', context.length);
    console.log('API Key exists:', hasAiApiKey());
    console.log('Model initialized:', !!(geminiService && geminiService.model));
    console.log('Screenshots count:', screenshots.length);

    if (!hasAiApiKey()) {
      console.error('No GROQ_API_KEY or GEMINI_API_KEY found');
      mainWindow.webContents.send('analysis-result', {
        error:
          'No API key configured. Add GROQ_API_KEY or GEMINI_API_KEY to your .env file.'
      });
      return;
    }

    if (!geminiService || !geminiService.model) {
      console.error('AI model not initialized');
      mainWindow.webContents.send('analysis-result', {
        error: 'AI model not initialized. Please check your API key.'
      });
      return;
    }

    if (screenshots.length === 0) {
      console.error('No screenshots to analyze');
      mainWindow.webContents.send('analysis-result', {
        error: 'No screenshots to analyze. Take a screenshot first.'
      });
      return;
    }

    try {
      console.log('Sending analysis start signal...');
      mainWindow.webContents.send('analysis-start');

      console.log('Processing screenshots...');
      const imageParts = await Promise.all(
        screenshots.map(async (path) => {
          console.log(`Processing screenshot: ${path}`);

          if (!fs.existsSync(path)) {
            console.error(`Screenshot file not found: ${path}`);
            throw new Error(`Screenshot file not found: ${path}`);
          }

          const imageData = fs.readFileSync(path);
          console.log(`Image data size: ${imageData.length} bytes`);

          return {
            inlineData: {
              data: imageData.toString('base64'),
              mimeType: 'image/png'
            }
          };
        })
      );

      console.log(`Prepared ${imageParts.length} image parts for analysis`);

      const contextPrompt = context ? `
    
CONVERSATION CONTEXT:
${context}

Based on the conversation context above and the screenshots provided, please:
1. Answer any questions that were asked in the conversation
2. Provide relevant insights about what's shown in the screenshots
3. If there are specific questions in the context, focus on answering those
4. Be concise but comprehensive

FORMAT YOUR RESPONSE AS:
    ` : '';

      const prompt = `You are an expert AI assistant for technical meetings and interviews. Analyze the provided screenshots and conversation context.

${contextPrompt}

**CODE SOLUTION:**
\`\`\`[language]
[Your complete, working code solution here - if applicable]
\`\`\`

**ANALYSIS:**
[Clear explanation of what you see in the screenshots and answers to any questions from the conversation]

**KEY INSIGHTS:**
• [Important insight 1]
• [Important insight 2]
• [Important insight 3]

Rules:
1. If there are questions in the conversation context, answer them directly
2. Provide code solutions if the screenshots show coding problems
3. Be concise but complete
4. Focus on actionable insights
5. If it's a meeting/presentation, summarize key points
6. Include time/space complexity for coding solutions

Analyze the screenshots and conversation context:`;

      console.log('Sending multimodal request to AI...');
      const text = await geminiService.generateMultimodal([prompt, ...imageParts]);
      console.log('Received response from AI');

      console.log('Generated text length:', text.length);
      console.log('Generated text preview:', text.substring(0, 200) + '...');

      chatContext.push({
        type: 'analysis',
        content: text,
        timestamp: new Date().toISOString(),
        screenshotCount: screenshots.length
      });

      mainWindow.webContents.send('analysis-result', { text });
      console.log('Analysis result sent to renderer');

    } catch (error) {
      console.error('Analysis error details:', error);
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);

      let errorMessage = 'Analysis failed';

      if (error.message.includes('API_KEY') || error.message.includes('invalid_api_key')) {
        errorMessage = 'Invalid API key. Check GROQ_API_KEY or GEMINI_API_KEY.';
      } else if (error.message.includes('quota') || error.message.includes('429')) {
        errorMessage = 'API quota or rate limit exceeded. Try again later.';
      } else if (error.message.includes('network') || error.message.includes('fetch')) {
        errorMessage = 'Network error. Please check your internet connection.';
      } else if (error.message.includes('model')) {
        errorMessage = 'AI model error. Please try a different model.';
      } else {
        errorMessage = `Analysis failed: ${error.message}`;
      }

      mainWindow.webContents.send('analysis-result', {
        error: errorMessage
      });
    }
  }

  async function analyzeForMeeting() {
    await analyzeForMeetingWithContext();
  }

  // IPC handlers
  ipcMain.handle('get-screenshots-count', () => {
    console.log('IPC: get-screenshots-count called, returning:', screenshots.length);
    return screenshots.length;
  });

  ipcMain.handle('toggle-stealth', () => {
    console.log('IPC: toggle-stealth called');
    return toggleStealthMode();
  });

  ipcMain.handle('emergency-hide', () => {
    console.log('IPC: emergency-hide called');
    return emergencyHide();
  });

  ipcMain.handle('take-stealth-screenshot', async () => {
    console.log('IPC: take-stealth-screenshot called');
    return await takeStealthScreenshot();
  });

  ipcMain.handle('analyze-stealth', async () => {
    console.log('IPC: analyze-stealth called');
    return await analyzeForMeeting();
  });

  ipcMain.handle('analyze-stealth-with-context', async (event, context) => {
    console.log('IPC: analyze-stealth-with-context called with context length:', context.length);
    return await analyzeForMeetingWithContext(context);
  });

  ipcMain.handle('clear-stealth', () => {
    console.log('IPC: clear-stealth called');
    screenshots.forEach(path => {
      if (fs.existsSync(path)) {
        fs.unlinkSync(path);
        console.log(`Deleted screenshot: ${path}`);
      }
    });
    screenshots = [];
    chatContext = [];
    console.log('All screenshots and context cleared');
    return { success: true };
  });

  ipcMain.handle('close-app', () => {
    console.log('IPC: close-app called');
    app.quit();
    return { success: true };
  });

  // Window resize handler – cycles through 4 size presets
  const RESIZE_PRESETS = [
    { label: 'small',      width: 680,  height: 280 },
    { label: 'medium',     width: 900,  height: 420 },
    { label: 'large',      width: 1100, height: 620 },
    { label: 'fullscreen', width: null, height: null }   // uses work-area max
  ];

  ipcMain.handle('resize-window', (event, preset) => {
    console.log('IPC: resize-window called, preset:', preset);
    if (!mainWindow || mainWindow.isDestroyed()) return { success: false, error: 'No window' };

    const { width: workW, height: workH } = screen.getPrimaryDisplay().workAreaSize;

    let targetPreset;
    if (preset && typeof preset === 'string') {
      // Explicit preset name requested
      targetPreset = RESIZE_PRESETS.find(p => p.label === preset);
    } else {
      // Cycle to the next preset
      const [curW, curH] = mainWindow.getSize();
      const curIdx = RESIZE_PRESETS.findIndex(p => {
        const pw = p.width  ?? workW;
        const ph = p.height ?? workH;
        return Math.abs(curW - pw) < 30 && Math.abs(curH - ph) < 30;
      });
      const nextIdx = (curIdx + 1) % RESIZE_PRESETS.length;
      targetPreset = RESIZE_PRESETS[nextIdx];
    }

    if (!targetPreset) targetPreset = RESIZE_PRESETS[1]; // default medium

    const newW = Math.min(targetPreset.width  ?? workW, workW);
    const newH = Math.min(targetPreset.height ?? workH, workH);

    // Re-centre horizontally
    const [, curY] = mainWindow.getPosition();
    const newX = Math.floor((workW - newW) / 2);
    const newY = Math.max(curY, 20);

    mainWindow.setBounds({ x: newX, y: newY, width: newW, height: newH }, true);
    console.log(`Window resized to ${targetPreset.label}: ${newW}x${newH}`);
    return { success: true, preset: targetPreset.label, width: newW, height: newH };
  });

  ipcMain.handle('get-window-size', () => {
    if (!mainWindow || mainWindow.isDestroyed()) return { width: 900, height: 420 };
    const [width, height] = mainWindow.getSize();
    return { width, height };
  });

  ipcMain.handle('get-stt-config', () => {
    const r = resolveSttEngine();
    return {
      engine: r.engine,
      warning: r.warning || null,
      groqSttModel:
        String(process.env.GROQ_STT_MODEL || '').trim() || 'whisper-large-v3-turbo'
    };
  });

  ipcMain.handle('transcribe-audio-groq', async (event, base64, mimeType = 'audio/webm') => {
    const key = String(process.env.GROQ_API_KEY || '').trim();
    if (!key) {
      return { success: false, error: 'GROQ_API_KEY not set' };
    }
    try {
      const buf = Buffer.from(base64, 'base64');
      if (buf.length < 400) {
        return { success: false, error: 'Recording too short' };
      }
      const model =
        String(process.env.GROQ_STT_MODEL || '').trim() || 'whisper-large-v3-turbo';
      const form = new FormData();
      const blob = new Blob([buf], { type: mimeType || 'audio/webm' });
      form.append('file', blob, 'speech.webm');
      form.append('model', model);

      const res = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}` },
        body: form
      });
      const raw = await res.text();
      if (!res.ok) {
        let msg = raw;
        try {
          const j = JSON.parse(raw);
          msg = j.error?.message || j.message || raw;
        } catch (_) {
          /* use raw */
        }
        return { success: false, error: msg || `HTTP ${res.status}` };
      }
      const data = JSON.parse(raw);
      const text = (data.text || '').trim();
      if (!text) {
        return { success: false, error: 'No speech detected in audio' };
      }
      return { success: true, text };
    } catch (e) {
      console.error('transcribe-audio-groq:', e);
      return { success: false, error: e.message };
    }
  });

  // Start Vosk live transcription
  ipcMain.handle('start-voice-recognition', () => {
    console.log('IPC: start-voice-recognition called');

    if (isVoskRunning && voskProcess && !voskProcess.killed) {
      console.log('Vosk already running — resuming capture');
      voskStdinWrite('resume');
      return { success: true, message: 'Resumed' };
    }

    try {
      const pythonScript = isDevelopment()
        ? path.join(__dirname, '..', 'vosk_live.py')
        : path.join(process.resourcesPath, 'vosk_live.py');
      console.log('Starting Vosk live transcription:', pythonScript);

      voskStdoutBuffer = '';
      voskProcess = spawnWinHidden('python', ['-u', pythonScript], {
        env: { ...process.env, PYTHONUTF8: '1' },
      });
      isVoskRunning = true;

      voskProcess.stdout.on('data', appendVoskStdout);

      voskProcess.stderr.on('data', (data) => {
        console.error('Vosk stderr:', data.toString());
      });

      voskProcess.on('close', (code) => {
        console.log('Vosk process exited with code:', code);
        isVoskRunning = false;
        voskProcess = null;
        voskStdoutBuffer = '';
        mainWindow.webContents.send('vosk-stopped');
      });

      voskProcess.on('error', (error) => {
        console.error('Failed to start Vosk:', error.message);
        isVoskRunning = false;
        voskProcess = null;
        voskStdoutBuffer = '';
      });

      return { success: true };

    } catch (error) {
      console.error('Error starting Vosk:', error.message);
      isVoskRunning = false;
      return { success: false, error: error.message };
    }
  });

  // Pause Vosk (stdin pause) — keeps Python + model warm
  ipcMain.handle('stop-voice-recognition', () => {
    console.log('IPC: stop-voice-recognition called');

    if (!isVoskRunning || !voskProcess || voskProcess.killed) {
      return { success: true, message: 'Not running' };
    }

    try {
      voskStdinWrite('pause');
      mainWindow.webContents.send('vosk-status', {
        status: 'stopped',
        message: 'Paused listening'
      });
      return { success: true };
    } catch (error) {
      console.error('Error stopping Vosk:', error.message);
      return { success: false, error: error.message };
    }
  });

  // REMOVED: convert-audio handler - not needed with direct AudioContext approach!
  // The renderer will handle audio conversion directly using AudioContext.decodeAudioData()
  // This is much more reliable and simpler than FFmpeg

  // Ask AI with a plain text question (no screenshots needed) - for voice-driven Q&A
  ipcMain.handle('ask-ai', async (event, question, history) => {
    console.log('IPC: ask-ai called, question:', question?.substring(0, 80));

    if (!geminiService || !geminiService.model) {
      return {
        success: false,
        error: 'AI not initialized. Set GROQ_API_KEY or GEMINI_API_KEY in .env.'
      };
    }

    if (!question || !question.trim()) {
      return { success: false, error: 'No question provided.' };
    }

    try {
      // Build context from conversation history passed from renderer
      const contextStr = (history || [])
        .filter(m => m.type === 'voice' || m.type === 'ai-response')
        .slice(-10) // last 10 exchanges to keep context window reasonable
        .map(m => {
          if (m.type === 'voice') return `User said: ${m.content}`;
          if (m.type === 'ai-response') return `Assistant: ${m.content}`;
          return null;
        })
        .filter(Boolean)
        .join('\n\n');

      const DETAIL_SYSTEM = `You are an expert AI assistant with deep knowledge across all domains — software engineering, data science, mathematics, science, business, and more.

Your answers must ALWAYS be:
1. **Detailed and thorough** — never give a one-liner when depth is warranted.
2. **Well-structured** — use headings (##), bullet points, numbered lists, and code blocks where appropriate.
3. **Explained with context** — state WHY, not just WHAT. Explain concepts, trade-offs, and reasoning.
4. **Include examples** — provide concrete code snippets, real-world analogies, or step-by-step walkthroughs.
5. **Comprehensive** — cover edge cases, common pitfalls, and best practices.
6. **Formatted for readability** — use **bold** for key terms, \`code\` for technical tokens, and separate sections clearly.

For coding questions: always include complete, runnable code with comments, time/space complexity, and usage examples.
For conceptual questions: explain from first principles, give comparisons, and real-world use cases.
Do NOT truncate your response. Aim for quality and completeness.`;

      const prompt = contextStr
        ? `${DETAIL_SYSTEM}\n\nCONVERSATION HISTORY:\n${contextStr}\n\n---\nUser question: ${question.trim()}\n\nProvide a detailed, well-structured, and comprehensive answer:`
        : `${DETAIL_SYSTEM}\n\nUser question: ${question.trim()}\n\nProvide a detailed, well-structured, and comprehensive answer:`;

      console.log('Asking AI (text-only)...');
      const answer = await geminiService.generateText(prompt);
      console.log('AI answered, length:', answer?.length);

      // Add to Gemini's internal history too
      geminiService.addToHistory('user', question.trim());
      geminiService.addToHistory('assistant', answer);

      return { success: true, answer };
    } catch (error) {
      console.error('ask-ai error:', error.message);
      let msg = error.message;
      if (msg.includes('API_KEY') || msg.includes('API key') || msg.includes('invalid_api_key')) {
        msg = 'Invalid API key. Check GROQ_API_KEY or GEMINI_API_KEY in .env';
      } else if (msg.includes('quota') || msg.includes('429')) {
        msg = 'API quota or rate limit exceeded. Try again in a moment.';
      } else if (msg.includes('404') || msg.includes('not found')) {
        msg = 'Model not found. Set GROQ_MODEL / GROQ_VISION_MODEL or check gemini-service.js.';
      }
      return { success: false, error: msg };
    }
  });


  // Transcribe audio using Python Whisper subprocess (FAST & OFFLINE!)
  ipcMain.handle('transcribe-audio', async (event, base64Audio, mimeType) => {
    console.log('IPC: transcribe-audio called, size:', base64Audio.length);

    const tmpDir = path.join(app.getPath('temp'), 'cluely-audio');

    try {
      // Create temp directory
      if (!fs.existsSync(tmpDir)) {
        fs.mkdirSync(tmpDir, { recursive: true });
      }

      // Save base64 audio to temp file
      const audioBuffer = Buffer.from(base64Audio, 'base64');
      const tempAudioPath = path.join(tmpDir, `audio_${Date.now()}.webm`);
      fs.writeFileSync(tempAudioPath, audioBuffer);

      console.log('Saved temp audio:', tempAudioPath, audioBuffer.length, 'bytes');

      // Spawn Python process
      return new Promise((resolve, reject) => {
        const pythonScript = isDevelopment()
          ? path.join(__dirname, '..', 'transcribe.py')
          : path.join(process.resourcesPath, 'transcribe.py');
        console.log('Running Python script:', pythonScript);

        const python = spawnWinHidden('python', [pythonScript, tempAudioPath]);

        let output = '';
        let errorOutput = '';

        python.stdout.on('data', (data) => {
          output += data.toString();
        });

        python.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });

        python.on('close', (code) => {
          // Clean up temp file
          try {
            fs.unlinkSync(tempAudioPath);
          } catch (e) {
            console.error('Failed to delete temp file:', e);
          }

          if (code !== 0) {
            console.error('Python exited with code:', code);
            console.error('Error:', errorOutput);
            resolve({ success: false, error: `Python error: ${errorOutput || 'Unknown error'}` });
            return;
          }

          try {
            const result = JSON.parse(output.trim());
            console.log('Transcription:', result.text || result.error);

            // Add to Gemini history if successful
            if (result.success && result.text && geminiService) {
              geminiService.addToHistory('user', result.text.trim());
            }

            resolve({
              success: result.success,
              transcript: result.text || '',
              error: result.error
            });

          } catch (parseError) {
            console.error('Failed to parse output:', output);
            resolve({ success: false, error: 'Failed to parse result' });
          }
        });

        python.on('error', (error) => {
          console.error('Failed to start Python:', error.message);

          // Clean up
          try {
            fs.unlinkSync(tempAudioPath);
          } catch (e) { }

          resolve({
            success: false,
            error: 'Python not found. Install Python and run: pip install openai-whisper'
          });
        });
      });

    } catch (error) {
      console.error('Error in transcribe-audio:', error.message);
      return { success: false, error: error.message };
    }
  });

  // Add voice transcript to history
  ipcMain.handle('add-voice-transcript', async (event, transcript) => {
    console.log('IPC: add-voice-transcript called');
    if (geminiService) {
      geminiService.addToHistory('user', transcript);
    }
    return { success: true };
  });

  // "What should I say?" feature
  ipcMain.handle('suggest-response', async (event, context) => {
    console.log('IPC: suggest-response called');
    try {
      if (!geminiService) {
        throw new Error('AI service not initialized');
      }
      const suggestions = await geminiService.suggestResponse(context);
      return { success: true, suggestions };
    } catch (error) {
      console.error('Error generating suggestions:', error);
      return { success: false, error: error.message };
    }
  });

  // Generate meeting notes
  ipcMain.handle('generate-meeting-notes', async () => {
    console.log('IPC: generate-meeting-notes called');
    try {
      if (!geminiService) {
        throw new Error('AI service not initialized');
      }
      const notes = await geminiService.generateMeetingNotes();
      return { success: true, notes };
    } catch (error) {
      console.error('Error generating meeting notes:', error);
      return { success: false, error: error.message };
    }
  });

  // Generate follow-up email
  ipcMain.handle('generate-follow-up-email', async () => {
    console.log('IPC: generate-follow-up-email called');
    try {
      if (!geminiService) {
        throw new Error('AI service not initialized');
      }
      const email = await geminiService.generateFollowUpEmail();
      return { success: true, email };
    } catch (error) {
      console.error('Error generating email:', error);
      return { success: false, error: error.message };
    }
  });

  // Answer specific question
  ipcMain.handle('answer-question', async (event, question) => {
    console.log('IPC: answer-question called');
    try {
      if (!geminiService) {
        throw new Error('AI service not initialized');
      }
      const answer = await geminiService.answerQuestion(question);
      return { success: true, answer };
    } catch (error) {
      console.error('Error answering question:', error);
      return { success: false, error: error.message };
    }
  });

  // Get conversation insights
  ipcMain.handle('get-conversation-insights', async () => {
    console.log('IPC: get-conversation-insights called');
    try {
      if (!geminiService) {
        throw new Error('AI service not initialized');
      }
      const insights = await geminiService.getConversationInsights();
      return { success: true, insights };
    } catch (error) {
      console.error('Error getting insights:', error);
      return { success: false, error: error.message };
    }
  });

  // Clear conversation history
  ipcMain.handle('clear-conversation-history', async () => {
    console.log('IPC: clear-conversation-history called');
    try {
      if (geminiService) {
        geminiService.clearHistory();
      }
      chatContext = [];
      return { success: true };
    } catch (error) {
      console.error('Error clearing history:', error);
      return { success: false, error: error.message };
    }
  });

  // Get conversation history
  ipcMain.handle('get-conversation-history', async () => {
    console.log('IPC: get-conversation-history called');
    try {
      if (!geminiService) {
        return { success: true, history: [] };
      }
      return { success: true, history: geminiService.conversationHistory };
    } catch (error) {
      console.error('Error getting history:', error);
      return { success: false, error: error.message };
    }
  });

  // App event handlers
  app.whenReady().then(() => {
    console.log('App is ready, creating window...');
    createStealthWindow();
    registerStealthShortcuts();
    // Add to app.whenReady() or before createWindow
    app.commandLine.appendSwitch('enable-features', 'VaapiVideoDecoder');
    app.commandLine.appendSwitch('ignore-certificate-errors');
    app.commandLine.appendSwitch('allow-running-insecure-content');
    app.commandLine.appendSwitch('disable-web-security');
    app.commandLine.appendSwitch('enable-media-stream');

    isVisible = true;

    console.log('Window setup complete - will show after content loads');
  });

  app.on('window-all-closed', () => {
    // Keep running in background for stealth operation
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createStealthWindow();
    }
  });

  app.on('will-quit', () => {
    globalShortcut.unregisterAll();

    if (voskProcess && !voskProcess.killed) {
      try {
        voskStdinWrite('quit');
        voskProcess.kill();
      } catch (e) {
        /* ignore */
      }
    }

    screenshots.forEach(path => {
      if (fs.existsSync(path)) fs.unlinkSync(path);
    });
  });

  app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
      event.preventDefault();
    });

    contents.on('will-navigate', (event, navigationUrl) => {
      if (navigationUrl !== mainWindow.webContents.getURL()) {
        event.preventDefault();
      }
    });
  });

  process.title = 'SystemIdleProcess';
})();