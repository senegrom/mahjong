/** Presentation contracts. App owns mutation, persistence and engine lifetime. */
import type { Snippet } from 'svelte';
export type GameMode = 'play' | 'watch' | 'physical' | 'guided';
export type TileFace = 'classic' | 'matisse' | 'van-gogh';
export type Opponent = 'beginner' | 'club' | 'neural';
export type Difficulty = Opponent | 'custom';
export type SeatName = 'east' | 'south' | 'west' | 'north';
export type Tile = string;
export type LegalChoice =
  | { kind: 'discard' | 'chii' | 'concealed-kan' | 'extended-kan' | 'riichi'; tile: Tile }
  | { kind: 'pon' | 'kan' | 'ron' | 'tsumo' | 'pass'; tile?: Tile | null };
export interface MeldView {
  kind: string;
  tiles: Tile[];
  from: string;
  claimed_tile?: Tile | null;
}
export interface DiscardView { tile: Tile; claimed: boolean; riichi: boolean; drawn?: boolean }
export interface SeatView {
  seat: SeatName; player: number; score: number; turn: boolean; riichi: boolean;
  hand: Tile[]; drawn: Tile | null; discards: DiscardView[]; melds: MeldView[];
}
export interface GameView {
  phase: 'act' | 'call' | 'over';
  seats: SeatView[]; round: SeatName; kyoku: number; wall: number;
  dora_indicators: Tile[]; dora_types: Tile[]; dora: Tile[]; safe: Tile[];
  shanten: number; waits: Tile[]; waits_left: number[]; furiten: boolean;
  counters: number; riichi_sticks: number;
  pending_discard: Tile | null; pending_from: SeatName | null;
  outcome?: { wins?: { ura_indicators?: Tile[] }[] } | null;
}
export interface DiscardHint { shanten: number; waits: Tile[]; waits_left: number[] }
export interface DiscardAnalyser { discard_hint(tile: Tile): DiscardHint }
export interface PendingOpponent { player: number; position: string }
export interface OfflineState {
  supported?: boolean | null;
  coreReady: boolean; aiReady: boolean; hasModel: boolean;
  phase: 'checking' | 'ready' | 'ai' | 'incomplete' | 'unavailable';
  progress: number; warning?: string; coreWarning?: string;
  persistent?: boolean; updateReady?: boolean;
  /** A first visit's service worker is still saving the game to activate. */
  installing?: boolean;
}
export interface OfflineStatusProps {
  offline: OfflineState; downloadAi: () => void; open?: boolean;
  /** The version of the trained network the page plays, such as '0.1.1410'. */
  network?: string;
}
export interface SettingsProps extends OfflineStatusProps {
  mode?: GameMode; settingsOpen?: boolean;
  hints?: boolean; confirmDiscards?: boolean; shortcuts?: boolean;
  /** Whether reviews keep Trained AI's move back, and the other players'
   * tiles, until the player asks to see them at each decision. */
  reviewHideMove?: boolean; reviewHideTiles?: boolean;
  tileFace: TileFace; pendingTileFace?: TileFace | null;
  /** Asked to play another listed network by version: true once App takes
   * the pick (it reloads the page), false to keep the one playing. */
  onnetworkchange?: (version: string) => boolean;
  difficulty: Difficulty; opponents: Opponent[];
  ready: boolean; busy: boolean; saveConflict: string; trainedAvailable: boolean;
  changeOpponents: (event: Event & { currentTarget: HTMLSelectElement }) => void;
  startFresh: () => boolean; configureTable: () => void;
  onfacechange: (face: TileFace) => void | Promise<void>;
  onconfirmationchange: () => void; onshortcutschange: () => void;
  /** Hands the game's log holds, the finished ones only, and what saves it,
   * which may answer false when it could not; why it could not, said beside
   * Export game's button. */
  loggedHands?: number; onsavegame?: () => boolean | void; exportFailure?: string;
}
export interface OpponentDialogProps {
  customDialog?: HTMLDialogElement | null; draftOpponents: Opponent[];
  opponents: Opponent[]; trainedAvailable: boolean; ready: boolean;
  saveConflict: string; startCustomTable: () => void;
}
/** One seat's concealed tiles at a past decision, as a review shows them. */
export interface HiddenHand { seat: SeatName; player: number; hand: Tile[]; drawn: Tile | null }
export interface MatchTableProps {
  view: GameView; hints: boolean; thinking: boolean; pendingOpponent: PendingOpponent | null;
  /** A review's face-up tiles for every seat, in the order of view.seats, or null to keep them face down. */
  hands?: HiddenHand[] | null;
  /** What each seat was doing at a review's decision, in the same order, such as "To play". */
  notes?: string[] | null;
  /** A table drawn for a review names each opponent by where they sat, and on a
   * phone held upright stacks them, so their face-up hands have room. */
  review?: boolean;
  /** A review's `hands` of the three opponents turned face down in their racks,
   * until the player asks to see them. */
  concealed?: boolean;
}
export interface PlayerHandProps {
  view: GameView; engine?: DiscardAnalyser | null; closed: boolean;
  hints: boolean; busy: boolean; blocked: boolean; discardChoices: LegalChoice[];
  /** A press selects a tile for confirmation, rather than discarding it. */
  confirmDiscards: boolean;
  picked: number | null; selected: number | null;
  canDiscard: (tile: Tile) => boolean;
  /** A tile pressed, with the click that pressed it, by which a review tells
   * a keyboard's press from a pointer's. */
  selectTile: (tile: Tile, index: number, event?: MouseEvent) => void;
  syncHandFocus: (event: FocusEvent) => void; handElement?: HTMLDivElement | null;
  /** The player's choices, drawn under the hand, over the called sets and
   * discards, where play has room for nothing between the hand and them (App);
   * null, as in a review, for nothing there. */
  choices?: Snippet | null;
  /** What a screen reader calls the tiles; a review names them for the seat. */
  handLabel?: string;
  /** The tiles as a list, one of a review's four hands, rather than a group. */
  listed?: boolean;
  /** What the player was doing at a review's decision, such as "To play". */
  note?: string;
  /** A review's words over each tile, by its place in the hand, such as
   * ['You', 'AI'], a line kept for them on every tile; null, as in play, for none. */
  tags?: string[][] | null;
}
export interface TurnChoicesProps {
  view: GameView; shownDora: Tile[]; busy: boolean; thinking: boolean;
  failure: string; saveConflict: string; loadNote: string;
  pendingOpponent: PendingOpponent | null; myTurn: boolean;
  confirmDiscards: boolean; shortcuts: boolean; touch: boolean;
  selectedTile: Tile | null; callChoices: LegalChoice[];
  choose: (choice: LegalChoice) => void | Promise<void>;
  discard: (tile: Tile) => void; oncancel: () => void;
}
