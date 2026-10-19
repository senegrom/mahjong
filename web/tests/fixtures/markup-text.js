/** The text of a piece of markup the renderer wrote: what lies between its
 * tags, a space where each tag was, as the page lays its elements apart, and
 * nothing where the renderer left one of its comment markers. It is read by
 * walking the markup rather than by deleting tags with a pattern, which is
 * no sanitizer and need not pretend to be one. */
export function text(markup) {
  let out = '', at = 0;
  while (at < markup.length) {
    if (markup.startsWith('<!--', at)) {
      const end = markup.indexOf('-->', at + 4);
      at = end < 0 ? markup.length : end + 3;
    } else if (markup[at] === '<') {
      const end = markup.indexOf('>', at + 1);
      at = end < 0 ? markup.length : end + 1;
      out += ' ';
    } else {
      out += markup[at++];
    }
  }
  return out.replace(/\s+/g, ' ').trim();
}
