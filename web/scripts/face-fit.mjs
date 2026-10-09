/** Where an artist face draws its painting on the game's 300 × 400 tile.
 *
 * Every face clips one raster to the rounded tile. By default the raster's box
 * overhangs each edge by a 1% bleed (3 units at the sides, 4 at the top and
 * bottom), so a study's outer pixels, such as a board's fringe or a drawn tile
 * edge, never show inside the clip. That box is 3:4 whatever the painting's
 * shape, so a painting that is not 3:4 would be stretched to fill it.
 *
 * A fit crops such a painting instead:
 * - `painting` is its true [width, height]. It defaults to the raster's own
 *   size; a raster that was resized without keeping its proportions names the
 *   crop it was resized from.
 * - `stretch` is how much wider the face shows the painting than it was
 *   painted, or narrower when negative. A little stretch lets the face crop less.
 * - `anchor` places the cut: 0 keeps the top or left edge, 0.5 centres the cut
 *   and 1 keeps the bottom or right edge.
 * - `bleedPixels`, when given, replaces the shared bleed on the uncut axis with
 *   that many source pixels, for a painting whose counted content comes so near
 *   that edge that the 1% would hide it.
 */
export const FACE_WIDTH = 300, FACE_HEIGHT = 400;
const SIDE_BLEED = 3, TOP_BLEED = 4;

// Hundredths of a unit are far finer than any screen shows a 300-unit face.
// Adding zero turns a negative zero into zero, so equal boxes compare equal.
const round = value => Math.sign(value) * Math.round(Math.abs(value) * 100) / 100 + 0;

/** The <image> box for a fit, or the shared box without one. `raster` is the
 * embedded raster's [width, height] in pixels. */
export function faceImage(fit, raster) {
  if (!fit) {
    return { x: -SIDE_BLEED, y: -TOP_BLEED, width: FACE_WIDTH + 2 * SIDE_BLEED, height: FACE_HEIGHT + 2 * TOP_BLEED };
  }
  const [paintingWidth, paintingHeight] = fit.painting ?? raster;
  const shown = paintingWidth / paintingHeight * (1 + fit.stretch);
  const { bleedPixels, anchor } = fit;
  if (shown < FACE_WIDTH / FACE_HEIGHT) {
    // Taller than the face: the whole width shows and the height is cut.
    const bleed = bleedPixels === undefined ? SIDE_BLEED : bleedPixels * FACE_WIDTH / (raster[0] - 2 * bleedPixels);
    const width = FACE_WIDTH + 2 * bleed, height = width / shown;
    return { x: round(-bleed), y: round(-anchor * (height - FACE_HEIGHT)), width: round(width), height: round(height) };
  }
  // Wider than the face: the whole height shows and the width is cut.
  const bleed = bleedPixels === undefined ? TOP_BLEED : bleedPixels * FACE_HEIGHT / (raster[1] - 2 * bleedPixels);
  const height = FACE_HEIGHT + 2 * bleed, width = height * shown;
  return { x: round(-anchor * (width - FACE_WIDTH)), y: round(-bleed), width: round(width), height: round(height) };
}

/** The box as the attributes an <image> element carries. */
export function imageAttributes({ x, y, width, height }) {
  return `x="${x}" y="${y}" width="${width}" height="${height}"`;
}

/** What a manifest records about a fitted face. */
export function fitRecord(fit, raster) {
  const [width, height] = fit.painting ?? raster;
  const { stretch, anchor, bleedPixels } = fit;
  return { painting: { width, height }, stretch, anchor, ...(bleedPixels === undefined ? {} : { bleedPixels }),
    image: faceImage(fit, raster) };
}

/** An approved study SVG with its picture moved to `box`. Every other byte of
 * the study stays as approved, so the face differs from it only in its fit. */
export function placeStudyImage(study, box) {
  const text = study.toString('utf8');
  const shared = imageAttributes(faceImage());
  if (text.split(shared).length !== 2) throw new Error('Expected one picture in the shared box');
  return Buffer.from(text.replace(shared, imageAttributes(box)), 'utf8');
}
