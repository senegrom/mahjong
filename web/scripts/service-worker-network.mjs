/** Observe the target that owns production downloads. Page request events do
 * not include a service worker's own fetches. Register before navigation and
 * attach existing workers too, so a reloaded page does not lose coverage. */
export async function observeServiceWorkerRequests(context, onRequest) {
  const attached = new Map(), failures = [];
  let closed = false;
  function attach(target) {
    if (closed || target.type() !== 'service_worker' || attached.has(target)) return;
    attached.set(target, (async () => {
      const session = await target.createCDPSession();
      session.on('Network.requestWillBeSent', ({ request }) => onRequest(request));
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
    if (failures.length) throw new AggregateError(failures, 'Service-worker network observation failed');
  };
}
