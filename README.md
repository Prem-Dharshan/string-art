# stringart

Turn a photo into **string art**: one continuous thread wound between pins on a frame, or one
thread per colour. An OpenCV pipeline crops to the face, weights important features, picks every
line with a physically modelled self-stopping solver, refines the path, and outputs a simulated
render, a thread-by-thread animation and a build kit for a real piece.

- **Documentation:** https://prem-dharshan.github.io/string-art/ (user guide, report,
  build guide, milestone log). The sources are in [`docs/`](docs/).
- **Try it online:** the Gradio app on Hugging Face Spaces (link on the docs home page).

## Quick start

```sh
uv sync --extra demo                          # Python 3.12 + all dependencies (needs uv)
uv run stringart fetch-models                 # face models (0.2 MB + 56 MB)
uv run stringart run sample:woman_smiling     # -> outputs/woman_smiling_greedy/render.png
uv run stringart viz outputs/woman_smiling_greedy
uv run --extra demo stringart demo            # web demo at http://127.0.0.1:7860
```

Every command and argument is explained in the
[user guide](https://prem-dharshan.github.io/string-art/guide/) (source:
[`docs/guide.md`](docs/guide.md)).

## Development

```sh
uv run pytest -q                              # 87 tests
uv run python tools/ruff.py check             # lint
uv run --group docs mkdocs serve              # docs site locally at http://127.0.0.1:8000
```

20XW97 project · Ajay H (22PW01) · Prem Dharshan D (22PW29) · photo credits:
[docs/credits.md](docs/credits.md)
