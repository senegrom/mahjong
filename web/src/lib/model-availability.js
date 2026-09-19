/** Availability is a live capability, not the result of a single startup HEAD.
 * Offline status remains independently verified; a successful download can
 * enable Trained immediately without changing any selected opponents/advisers.
 */
export function watchModelAvailability({ probe, watchOffline, refreshOffline, onChange,
  events = window, page = document }) {
  let closed = false;
  let revision = 0;
  async function check() {
    const owner = ++revision;
    let available = false;
    try { available = Boolean(await probe()); } catch { /* A later event retries. */ }
    if (!closed && owner === revision) onChange(available);
  }
  const unwatch = watchOffline(info => {
    if (closed || !info.aiReady) return;
    // A completed, verified download outranks any earlier negative probe.
    revision++;
    onChange(true);
  });
  const refresh = () => {
    if (closed || page.visibilityState !== 'visible') return;
    // A stalled offline-status request must not hold up an online probe.
    void check();
    void Promise.resolve().then(() => { if (!closed) return refreshOffline(); }).catch(() => {});
  };
  events.addEventListener('online', refresh);
  page.addEventListener('visibilitychange', refresh);
  void check();
  return () => {
    closed = true;
    revision++;
    unwatch();
    events.removeEventListener('online', refresh);
    page.removeEventListener('visibilitychange', refresh);
  };
}
