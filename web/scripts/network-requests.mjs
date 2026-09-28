/** Observe every target that can download the trained network: the page,
 * which saves it for offline play, its workers, and the service worker, which
 * must never fetch it. Page request events alone miss a worker's own fetches.
 * Register before navigation and attach existing targets too, so a reloaded
 * page does not lose coverage. A request reported to two targets is counted
 * once. */
const TYPES = new Set(['page', 'worker', 'service_worker']);

export async function observeNetworkRequests(context, onRequest) {
  const attached = new Map(), seen = new Set(), failures = [];
  let closed = false;
  function attach(target) {
    if (closed || !TYPES.has(target.type()) || attached.has(target)) return;
    attached.set(target, (async () => {
      const session = await target.createCDPSession();
      session.on('Network.requestWillBeSent', ({ requestId, request, redirectResponse }) => {
        // A redirect continues the same request under the same id.
        if (redirectResponse || seen.has(requestId)) return;
        seen.add(requestId);
        onRequest(request, target.type());
      });
      await session.send('Network.enable');
      return session;
    })().catch(error => {
      if (!closed && context.targets().includes(target)) failures.push(error);
      return null;
    }));
  }
  context.on('targetcreated', attach);
  for (const target of context.targets()) attach(target);
  await Promise.all(attached.values());
  return async () => {
    closed = true;
    context.off('targetcreated', attach);
    const sessions = await Promise.all(attached.values());
    await Promise.all(sessions.filter(Boolean).map(session => session.detach().catch(() => {})));
    if (failures.length) throw new AggregateError(failures, 'Network observation failed');
  };
}
