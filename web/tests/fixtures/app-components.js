import { mount, unmount } from 'svelte';
import Fixture from './AppComponents.svelte';
import '../../src/app.css';
import '../../src/lib/app/controls.css';
const instance = mount(Fixture, { target: document.getElementById('app') });
// Test-only lifecycle hook. No fixtures or hooks enter the production bundle.
window.unmountFixture = () => unmount(instance);
