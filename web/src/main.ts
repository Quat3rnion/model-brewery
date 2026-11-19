import '@fontsource-variable/ibm-plex-sans/wght.css';
import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { theme } from './lib/theme.svelte';

theme.apply();

const target = document.getElementById('app');
if (!target) throw new Error('Missing #app element');
export default mount(App, { target });
