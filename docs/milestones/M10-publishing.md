# M10: Docs site and online app

**Status:** done · **Commit:** see the milestone log · **App:**
https://cv-string-art.streamlit.app/

## Goal
Publish the documentation as a website, and put the demo online so the reviewers can try it
without installing anything.

## What was built
- **Docs site:** MkDocs Material built from `docs/` and deployed to GitHub Pages by
  `.github/workflows/docs.yml` on every push to `master` that touches the docs. Live at
  https://prem-dharshan.github.io/string-art/. Strict mode fails the build on any broken link.
  The internal planning notes are excluded from the site.
- **Online app:** `app/streamlit_app.py` is a Streamlit front end around the same
  `stringart.demo.make_art` pipeline as the Gradio demo. It has a sample-photo picker,
  upload, the same frame/thread controls, a live progress bar, and downloads for the build
  kit, mp4 and pin sequence. On startup it downloads the face models and compiles the numba
  kernels once per server, so a visitor never waits for compilation.
- `app/requirements.txt` is generated from `uv.lock` by `tools/streamlit_requirements.py`:
  the pinned core dependencies plus Streamlit, with OpenCV swapped for its headless build.
- `tools/deploy_space.py` packages the Gradio demo as a Hugging Face Space. It isn't used:
  free Gradio Spaces now need a PRO account.

## Verification
- `tests/test_streamlit_app.py` drives the app with Streamlit's `AppTest`. It picks a sample,
  sets 128 pins and 2 colours, presses the button, and checks the render size, the GIF and
  zip contents, and the three download buttons.
- The same test passes in a fresh Python 3.12 environment with **only** `app/requirements.txt`
  installed (about 40 s), which is how Community Cloud installs it.
- Full suite: 88 tests pass. Lint and format are clean.

## Deploy (one time)
See [the online version](../guide.md#the-online-version-streamlit) in the user guide:
share.streamlit.io → repo `Prem-Dharshan/string-art`, branch `master`, file
`app/streamlit_app.py`, Python 3.12.
