/**
 * Starts Electron detached from this Node process so no extra console/node
 * session stays tied to the GUI app on Windows (avoids taskbar clutter from
 * the launch chain). Logs from the app go nowhere unless you use npm run dev.
 */
const { spawn } = require('child_process');
const path = require('path');

const electronExe = require('electron');
const appRoot = path.join(__dirname, '..');
const passThroughArgs = process.argv.slice(2);

const opts = {
  cwd: appRoot,
  detached: true,
  stdio: 'ignore',
};
if (process.platform === 'win32') {
  opts.windowsHide = true;
}

const child = spawn(electronExe, ['.', ...passThroughArgs], opts);
child.unref();
process.exit(0);
