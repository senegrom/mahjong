import { mount } from 'svelte';
import init from '../../src/wasm/riichi.js';
import Fixture from './AgentDefaults.svelte';
import '../../src/app.css';
await init();
mount(Fixture, { target: document.getElementById('app') });
