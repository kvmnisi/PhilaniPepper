/* Deployment configuration. Overwritten at build time by build-env.js when
   the API_BASE_URL environment variable is set (e.g. on Vercel), and by
   scripts/sync_pepper_html.py for the Pepper copy. null = use defaults. */
window.APP_ENV = { apiBaseUrl: null, runtime: null };
