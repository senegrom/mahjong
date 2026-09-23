/** Keep Tab navigation in an open modal rather than handing focus to browser
 * chrome. Native showModal still owns inertness, Escape and opener restoration.
 * Re-read controls on each key press: details, downloads and disabled controls
 * can change while the dialog is open.
 * @param {KeyboardEvent} event
 */
export function cycleDialogFocus(event) {
  const dialog = event.currentTarget;
  if (event.key !== 'Tab' || !(dialog instanceof HTMLDialogElement) || !dialog.open) return;
  const collapsed = [...dialog.querySelectorAll('details:not([open])')];
  const controls = [...dialog.querySelectorAll('button, a[href], input, select, textarea, summary, [tabindex]')]
    .filter(element => element instanceof HTMLElement && element.tabIndex >= 0
      && !element.matches(':disabled, [hidden]') && !element.closest('[inert]')
      && collapsed.every(details => !details.contains(element) || details.querySelector(':scope > summary')?.contains(element))
      && element.getClientRects().length > 0 && getComputedStyle(element).visibility !== 'hidden');
  const first = controls[0], last = controls.at(-1);
  const active = dialog.ownerDocument.activeElement;
  if (!first || !last) {
    event.preventDefault();
    dialog.focus();
  } else if (event.shiftKey && (active === first || !controls.includes(active))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (active === last || !controls.includes(active))) {
    event.preventDefault();
    first.focus();
  }
}
