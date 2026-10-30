// Real ONNX graphs, real ORT/reduced WASM, and the production worker source.
// Only browser transport and model-byte fetching are adapted for Node. No
// session.run, logits, legality input or tensor disposal is mocked.
//
// Then the page's own averaging over the suits (src/lib/suits.js), asked
// through the same worker, against neural.suits's averaging player on the
// export's PyTorch twin (suits.json): a question the plain answer is sure of
// is asked once, and one it hesitates over is asked again with the renamed
// game's planes and moves, and played as the player plays it.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import vm from 'node:vm';
import * as ort from 'onnxruntime-web/wasm';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MEMORY_LIMITS_MIB, isMemoryError } from '../src/lib/memory-budget.js';
import { GATE, MOVES, averagedAnswer, averagedAnswers, swapMoves } from '../src/lib/suits.js';
import { copyRuntime } from '../scripts/copy-runtime.mjs';
import { moduleSource } from './fixtures/load-module.js';

const folder = resolve(process.argv[2]);
const cases = JSON.parse(await readFile(resolve(folder, 'inputs.json'), 'utf8'));
const averaging = JSON.parse(await readFile(resolve(folder, 'suits.json'), 'utf8'));
assert.ok(cases.some(row => row.takes_legal && row.stage === 'riichi-discard'));
assert.ok(cases.some(row => !row.takes_legal));
// public/ort is generated, not tracked. Use the same checksum-checked copier
// as the production build; never fall back to the package's generic runtime.
const { directory } = await copyRuntime();
const runtimeBase = new URL(`../public/${directory}`, import.meta.url).href;
const source = moduleSource('policy.worker.js', { 'import(/* @vite-ignore */ controls)': 'loadMemoryControls(controls)' });
const tensors = [], messages = [];
// What the request under way must be run on and answer: a row a position,
// each `{ takes_legal, planes, mask, logits, value, hands }`. `runs` notes
// how many positions each run held.
let current, runs = [];
function track(tensor) {
  const entry = { disposed: false };
  tensors.push(entry);
  const dispose = tensor.dispose.bind(tensor);
  tensor.dispose = () => {
    assert.equal(entry.disposed, false, 'each tensor is disposed once');
    entry.disposed = true;
    dispose();
  };
  return tensor;
}
function close(actual, expected, label) {
  assert.ok(Number.isFinite(actual), `${label}: nonfinite output`);
  assert.ok(Math.abs(actual - expected) <= 0.0001 * (1 + Math.abs(expected)),
    `${label}: ${actual} differs from ${expected}`);
}
const bridge = {
  env: ort.env,
  Tensor: function(type, data, dims) { return track(new ort.Tensor(type, data, dims)); },
  InferenceSession: { create: async (bytes, options) => {
    const session = await ort.InferenceSession.create(bytes, options);
    return {
      inputNames: session.inputNames,
      release: () => session.release(),
      run: async feed => {
        const rows = current.rows ?? [current], legal = rows[0].takes_legal;
        runs.push(rows.length);
        assert.deepEqual(Object.keys(feed), legal ? ['planes', 'legal'] : ['planes']);
        assert.equal(feed.planes.dims[0], rows.length);
        assert.deepEqual(Array.from(feed.planes.data), rows.flatMap(row => row.planes), 'each position\'s own planes');
        if (legal) {
          assert.equal(feed.legal.type, 'float32');
          assert.deepEqual(Array.from(feed.legal.dims), [rows.length, 46]);
          assert.deepEqual(Array.from(feed.legal.data), rows.flatMap(row => row.mask.map(value => value ? 1 : 0)));
        }
        const result = await session.run(feed);
        for (const tensor of Object.values(result)) track(tensor);
        rows.forEach((row, at) => {
          for (let i = 0; i < 46; i++) {
            if (row.mask[i]) close(result.policy.data[at * 46 + i], row.logits[i], `policy ${at}/${i}`);
            else if (legal) assert.equal(result.policy.data[at * 46 + i], -Infinity);
          }
          close(result.value.data[at], row.value, `value ${at}`);
          row.hands.forEach((value, i) => close(result.hands.data[at * row.hands.length + i], value, `hands ${at}/${i}`));
        });
        return result;
      },
    };
  } },
};
const self = { postMessage: message => messages.push(message) };
const context = vm.createContext({
  ort: bridge, self, policyWeights, MEMORY_LIMITS_MIB, isMemoryError, URL,
  Float32Array, Float64Array,
  networkBytes: async ({ url }) => new Uint8Array(await readFile(url)),
  loadMemoryControls: url => import(url),
});
vm.runInContext(source, context);
let asked = 0;
/** The worker's message answering a request of `data`, after it has run. */
async function answered(model, data) {
  const id = asked++;
  await self.onmessage({ data: { id, url: resolve(folder, model), runtimeBase, details: true, ...data } });
  const failed = messages.find(message => message.id === id && message.error);
  assert.equal(failed, undefined, failed?.error);
  const answer = messages.find(message => message.id === id && (message.action !== undefined || message.analyses));
  assert.ok(answer, 'worker must answer');
  assert.ok(tensors.every(tensor => tensor.disposed), 'all inputs and outputs released after each request');
  return answer;
}
try {
  for (let id = 0; id < cases.length; id++) {
    current = cases[id];
    const answer = await answered(current.model, { planes: Float32Array.from(current.planes), mask: current.mask });
    assert.equal(answer.action, current.action, `${current.model}/${current.stage}`);
    assert.equal(current.mask[answer.action], true);
  }

  // The page averages with the player's own gate and renaming.
  assert.equal(GATE, averaging.gate);
  assert.deepEqual(MOVES, averaging.moves);
  const questions = averaging.cases;
  assert.deepEqual(questions.map(question => question.stage), ['sure', 'kept', 'changed', 'reach tile']);
  assert.deepEqual(questions.map(question => question.hesitant), [false, true, true, true]);
  // Each question, plain and renamed, as the worker is asked it: the page's
  // planes, by the array the page sends, and what that run must answer.
  const expected = new Map();
  const positions = questions.map(question => {
    const planes = Float32Array.from(question.planes), renamed = Float32Array.from(question.renamed);
    expected.set(planes, { takes_legal: true, planes: question.planes, mask: question.mask, logits: question.logits, value: question.value,
      hands: question.hands });
    expected.set(renamed, { takes_legal: true, planes: question.renamed, mask: swapMoves(question.mask), logits: question.renamed_logits,
      value: question.renamed_value, hands: question.renamed_hands });
    return { planes, mask: question.mask, renamed: () => renamed };
  });
  const played = (question, answer, how) => {
    assert.equal(answer.action, question.action, `${question.stage}, ${how}: the move the averaging player plays`);
    answer.weights.forEach((weight, move) => assert.ok(Math.abs(weight - question.weights[move]) <= 1e-4,
      `${question.stage}, ${how}, move ${move}: ${weight} against the player's ${question.weights[move]}`));
  };
  // One at a time, as an opponent, the adviser and the hand's notes ask.
  for (const [at, question] of questions.entries()) {
    runs = [];
    const answer = await averagedAnswer(async (planes, mask) => {
      current = expected.get(planes);
      assert.ok(current, 'a question is asked of its own planes or its renamed game\'s');
      return (await answered(question.model, { planes: Float32Array.from(planes), mask: Array.from(mask) })).analysis;
    }, positions[at]);
    assert.deepEqual(runs, question.hesitant ? [1, 1] : [1], `${question.stage}: asked again only where it hesitates`);
    played(question, answer, 'alone');
  }
  // In batches, as the game's review and its reasoning ask: one of every
  // question, then one of the renamed games of those it hesitates over.
  runs = [];
  const batches = [];
  const answers = await averagedAnswers(async sent => {
    current = { rows: sent.map(position => expected.get(position.planes)) };
    batches.push(sent.map(position => positions.findIndex(own => own.planes === position.planes || own.renamed() === position.planes)));
    const width = sent[0].planes.length, planes = new Float32Array(sent.length * width);
    sent.forEach((position, at) => planes.set(position.planes, at * width));
    return (await answered(questions[0].model, { planes, mask: sent.flatMap(position => Array.from(position.mask)), count: sent.length })).analyses;
  }, positions);
  assert.deepEqual([runs, batches], [[4, 3], [[0, 1, 2, 3], [1, 2, 3]]]);
  questions.forEach((question, at) => played(question, answers[at], 'in a batch'));
  assert.notEqual(questions[2].action, questions[2].logits.indexOf(Math.max(...questions[2].logits.filter(Number.isFinite))),
    'the changed question plays another move than its plain answer');
} finally {
  await vm.runInContext('held?.session.release()', context);
}
console.log(`PASS: ${cases.length} production-worker inferences using real exported graphs and reduced WASM; both riichi stages, one/two-input compatibility, masked logits and tensor disposal; `
  + `${averaging.cases.length} questions averaged over the suits as neural.suits plays them, alone and in a batch`);
