/** Presentation contracts. App owns mutation, persistence and engine lifetime. */
export type GameMode = 'play' | 'watch' | 'physical' | 'guided';
export type TileFace = 'classic' | 'matisse' | 'dali' | 'van-gogh';
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
  coreLoading?: boolean; persistent?: boolean; updateReady?: boolean;
}
export interface OfflineStatusProps { offline: OfflineState; downloadAi: () => void; open?: boolean }
export interface SettingsProps extends OfflineStatusProps {
  mode?: GameMode; settingsOpen?: boolean;
  hints?: boolean; confirmDiscards?: boolean; shortcuts?: boolean;
  tileFace: TileFace; pendingTileFace?: TileFace | null;
  difficulty: Difficulty; opponents: Opponent[];
  ready: boolean; busy: boolean; saveConflict: string; trainedAvailable: boolean;
  changeOpponents: (event: Event & { currentTarget: HTMLSelectElement }) => void;
  startFresh: () => boolean; configureTable: () => void;
  onfacechange: (face: TileFace) => void | Promise<void>;
  onconfirmationchange: () => void; onshortcutschange: () => void;
}
export interface OpponentDialogProps {
  customDialog?: HTMLDialogElement | null; draftOpponents: Opponent[];
  opponents: Opponent[]; trainedAvailable: boolean; ready: boolean;
  saveConflict: string; startCustomTable: () => void;
}
export interface MatchTableProps {
  view: GameView; hints: boolean; thinking: boolean; pendingOpponent: PendingOpponent | null;
}
export interface PlayerHandProps {
  view: GameView; engine?: DiscardAnalyser | null; closed: boolean;
  hints: boolean; busy: boolean; blocked: boolean; discardChoices: LegalChoice[];
  picked: number | null; selected: number | null;
  canDiscard: (tile: Tile) => boolean; selectTile: (tile: Tile, index: number) => void;
  syncHandFocus: (event: FocusEvent) => void; handElement?: HTMLDivElement | null;
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
