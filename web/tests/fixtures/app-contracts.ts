import type { GameMode, TileFace, LegalChoice, SettingsProps, OfflineState } from '../../src/lib/app/types';
const mode: GameMode = 'guided';
const face: TileFace = 'matisse';
const choice: LegalChoice = { kind: 'chii', tile: '1m' };
void [mode, face, choice];
// @ts-expect-error Retired model names are not tile-face options.
const badFace: TileFace = 'quick';
// @ts-expect-error Invalid actions must not cross the presentation boundary.
const badChoice: LegalChoice = { kind: 'delete', tile: '1m' };
// @ts-expect-error Offline readiness is a boolean, not a status string.
const badReadiness: OfflineState['aiReady'] = 'ready';
// @ts-expect-error Face change callbacks must accept the selected face.
const badCallback: SettingsProps['onfacechange'] = (face: number) => { void face; };
// @ts-expect-error A discard must name its tile.
const missingTile: LegalChoice = { kind: 'discard' };
void [badFace, badChoice, badReadiness, badCallback, missingTile];
