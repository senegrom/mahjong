/** A deterministic inference boundary for the browser regression. The real
 * evaluator, action mapping, watched session, confirmation and game still run. */
export async function analyzePolicy(_planes, mask, signal) {
  if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
  const index = window.riichiCalls++, answer = window.riichiPlan[index];
  if (answer) {
    if (JSON.stringify(Array.from(mask, Boolean)) !== JSON.stringify(answer.mask)) {
      throw new Error(`Riichi fixture diverged at inference ${index}`);
    }
    return answer;
  }
  const action = Array.from(mask).findIndex((valid, at) => valid && at !== 37);
  if (action < 0) throw new Error('The post-fixture decision has no ordinary legal move');
  return { action, weights: Array.from({ length: 46 }, (_, at) => Number(at === action)) };
}
export function chooseAction() { throw new Error('Only the followed seat is trained in this fixture'); }
