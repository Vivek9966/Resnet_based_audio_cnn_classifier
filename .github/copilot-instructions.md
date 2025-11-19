## Quick summary

This repository contains a small image-processing prototype implemented as a Jupyter notebook (`test.ipynb`) and an empty `main.py`. The agent's primary goal is to be immediately productive working with the notebook and to safely add an entrypoint in `main.py` when requested.

## Big picture (what matters)
- Primary code lives in `test.ipynb` — it's the canonical workflow: it imports `cv2`, `numpy`, `PIL.Image`, creates the `images/` and `tgt_image/` directories, and defines `image_loader()` which finds image files under `images/` using glob patterns.
- `images/` is the input folder; `tgt_image/` is the output folder. The notebook uses Windows-style path strings in one place (Path(".\\images")).
- `main.py` is currently empty; if an entrypoint is needed, prefer adding a small CLI wrapper that re-uses the notebook logic (or extracts the notebook logic into an importable module).

## Project-specific conventions & patterns
- Input images: any file matching these patterns is considered an image: `*.jpg, *.jpeg, *.png, *.gif, *.bmp, *.tiff, *.webp` (see `image_loader()` in `test.ipynb`).
- The repository relies on runtime imports: `cv2` (OpenCV), `numpy`, and `PIL.Image` — there is no `requirements.txt` in the repo, so confirm dependencies before running.
- The notebook creates directories with `os.makedirs("images")` and `os.makedirs("tgt_image")`; code that operates on these folders should check existence and create them if missing.
- A global flag `glb` is used in `image_loader()` to indicate whether images were found; if refactoring, preserve its observable behavior or update callers accordingly.

## Integration points & external dependencies
- Native Python packages referenced: `opencv-python` (cv2), `numpy`, `Pillow` (PIL). Install with:

  ```powershell
  python -m pip install opencv-python numpy pillow
  ```

- No CI, build, or test frameworks detected. The typical developer flow is:
  1. Open `test.ipynb` in VS Code / Jupyter and run cells interactively.
  2. Place images into `images/` and inspect `tgt_image/` for outputs.

## How the agent should proceed (actionable guidance)
- When asked to modify behavior, prefer changes that are discoverable from existing code:
  - Reuse `image_loader()` patterns and the ext list when adding new image-processing steps.
  - Keep `images/` and `tgt_image/` as the default input/output paths unless explicitly asked to change them.
  - Respect the global `glb` flag semantics: if `image_loader()` sets `glb=0` and returns a string when no images are found, callers expect that behavior.
- If implementing `main.py` as an entrypoint, create a minimal CLI that:
  - Ensures `images/` and `tgt_image/` exist.
  - Calls the same image discovery logic (or imports a refactored function from a new module) and exits with an informative message when no images are found.

## Examples drawn from this repo
- `test.ipynb` cell: image discovery

  - Uses: `img_path = Path(".\\images")` and `imgls = [f for pattern in ext for f in img_path.glob(f"**/{pattern}")]`
  - Actionable note: `glob("**/{pattern}")` searches recursively; avoid changing this to non-recursive unless intended.

## What not to change without explicit instruction
- Don't rename or remove `images/` or `tgt_image/` directories silently.
- Don't replace the image filename patterns without a clear reason — they match common formats used by the notebook.

## Verification steps for small changes
- To verify behavior locally:
  1. Ensure dependencies are installed (see command above).
  2. Put a sample image into `images/` (e.g., `images/sample.jpg`).
  3. Open and run `test.ipynb` or run the implemented `main.py` if present.
  4. Confirm discovered images are non-empty and outputs (if any) are written to `tgt_image/`.

## When you need more context
- If a requested change involves new persistent behavior (new file formats, new output locations, production packaging), ask the maintainer for the intended target (notebook-first vs. script-first) and whether a `requirements.txt` or minimal test should be added.

---
Please review: tell me if you'd like the agent to prefer implementing a CLI in `main.py` vs. refactoring notebook logic into a library module first.
