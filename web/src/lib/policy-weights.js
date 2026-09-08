/** Stable softmax over legal moves only. These are policy preferences, not
 * estimated chances of winning. Inference remains greedy at temperature 0. */
export function policyWeights(logits, mask) {
  if (logits.length !== mask.length) throw new Error('The policy returned the wrong number of choices');
  const legal = Array.from(mask, (allowed, index) => allowed ? index : -1).filter(index => index >= 0);
  if (!legal.length) throw new Error('There are no legal choices in this position');
  if (legal.some(index => !Number.isFinite(logits[index]))) throw new Error('The policy returned invalid weights');
  const best = legal.reduce((a, b) => logits[b] > logits[a] ? b : a);
  const weights = Array.from(mask, (allowed, index) => allowed ? Math.exp(logits[index] - logits[best]) : 0);
  const total = weights.reduce((a, b) => a + b, 0);
  return { action: best, weights: weights.map(value => value / total) };
}
