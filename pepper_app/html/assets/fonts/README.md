# Heading fonts

The stylesheet asks for **Europa Grotesk SH Bold** first, then **Grotesk**
(SoftMaker), then falls back to `"Arial Narrow", Arial, sans-serif`.

Both are commercial typefaces, so the font files are **not committed** and
are listed in `.gitignore` and `frontend/.vercelignore`. To use them:

1. Confirm that your licence (or the University's) allows embedding the font
   in a web page.
2. Copy the files here with these exact names:
   - `EuropaGroteskSH-Bold.woff2` and/or `EuropaGroteskSH-Bold.woff`
   - `Grotesk-Medium.ttf`
3. Reload the page. The `@font-face` rules in `css/styles.css` pick them up.
   Fonts installed on the device are used too (`local()`), with no files needed.

If the licence does allow redistribution and you want the hosted (Vercel)
build to use the font, remove the matching lines from `.gitignore` and
`frontend/.vercelignore` and commit the file deliberately.
