import test from 'node:test';
import assert from 'node:assert/strict';
import { npxCommand } from '../scripts/publish-model-r2.mjs';

test('the publisher runs npx without a shell, so no argument is re-split', () => {
  const file = 'C:\\Users\\Jane Doe\\AppData\\Local\\Temp\\mahjong-model-upload-1\\network.gz';
  assert.deepEqual(npxCommand(['--file', file], 'linux', '/usr/bin/node'), ['npx', ['--file', file]]);
  // npx.cmd cannot start without a shell, so node runs the script it runs.
  assert.deepEqual(npxCommand(['--file', file], 'win32', 'C:\\Program Files\\nodejs\\node.exe'), [
    'C:\\Program Files\\nodejs\\node.exe',
    ['C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npx-cli.js', '--file', file],
  ]);
});
