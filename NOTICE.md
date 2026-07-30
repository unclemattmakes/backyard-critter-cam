# Third-party notices

This project is licensed under the GNU Affero General Public License v3.0 (see
[LICENSE](LICENSE)). It is, however, mostly a wiring harness around four pre-trained models
and a short list of Python libraries, all of which carry their own terms. This file names them.

**No model weights are redistributed here.** Nothing in `weights/` or `~/.cache/` is committed
or shipped with this repository — every checkpoint below is downloaded from its own upstream on
first run (MegaDetector via `urllib` into `weights/`, the rest via Hugging Face into the
usual `HF_HOME` cache). You are the one fetching them, and their licenses bind you directly.

> **Read this before charging anyone money.** MegaDescriptor-L-384 — the model behind the whole
> individual re-identification feature (`embed.py`, `reid.py`, `individuals.py`, `clipembed.py`)
> — is **CC-BY-NC-4.0: non-commercial use only**. This repository's own AGPL-3.0 license permits
> commercial use, but that permission does not extend to a weight file it does not own. Run the
> re-ID pipeline in a commercial product and you are violating the model's license, not this
> one. Detection, species labelling, and behaviour analysis are unaffected — only re-ID depends
> on MegaDescriptor. If you need re-ID commercially, swap in a differently-licensed embedder;
> `detection_embeddings` is keyed by `model`, so a second embedder is a new row, not a migration
> (see the note at the top of `embed.py`).

## Models

| Model | What this repo uses it for | Fetched from | License | Asks to be cited as |
|---|---|---|---|---|
| **MegaDetector v6** (MDv6, Microsoft AI for Good Lab) | Every animal/person/vehicle box. `detector.py` runs the official MDv6 weights directly on Ultralytics YOLO; the default is `MDV6-yolov10-c`. | Zenodo record [15398270](https://zenodo.org/records/15398270), downloaded into `weights/` on first run and checked against a pinned SHA-256. | **CC-BY-4.0.** Attribution is a *condition* of the license here, not a courtesy — this file is how this project satisfies it. | Microsoft AI for Good Lab, *MegaDetector v6*, Zenodo, DOI [10.5281/zenodo.15398270](https://doi.org/10.5281/zenodo.15398270). The pipeline's original description is Beery, Morris & Yang, *Efficient pipeline for camera trap image review*, arXiv:1907.06772 (2019). |
| **BioCLIP 2** (Imageomics Institute) | Zero-shot species labels for every crop (`classify.py`), via the `pybioclip` package's `CustomLabelsClassifier`. No training, no fine-tuning — the label list in `classify.py` is just text prompts. | Hugging Face `hf-hub:imageomics/bioclip-2`. | **MIT**, with a citation request. | Gu et al., *BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning*, NeurIPS 2025. BioCLIP 2 builds on BioCLIP 1 — Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*, CVPR 2024 — and both are trained with OpenCLIP, so cite those too. |
| **OpenCLIP** (`ViT-B-32` / `laion2b_s34b_b79k`) | The non-animal prefilter (`clipfilter.py`): a general-image CLIP that answers the one question BioCLIP structurally cannot — "is this even an animal?" — so a plate of cat food stops being logged as a brown rat. Also the clip-space embedder in `clipembed.py`. | Hugging Face, through `open_clip` (installed as a `pybioclip` dependency). | **MIT.** | Ilharco et al., *OpenCLIP*, Zenodo, DOI [10.5281/zenodo.5143773](https://doi.org/10.5281/zenodo.5143773). The checkpoint itself is a LAION-2B-trained CLIP; see Cherti et al., *Reproducible scaling laws for contrastive language-image learning*, CVPR 2023. |
| **MegaDescriptor-L-384** (BVRA / Wildlife Datasets) | Individual re-identification: one 1536-d appearance vector per readable crop (`embed.py`), clustered by `reid.py` and matched by `individuals.py`. | Hugging Face `hf-hub:BVRA/MegaDescriptor-L-384`, pulled through `timm` (~900 MB on first fetch). | **CC-BY-NC-4.0 — NON-COMMERCIAL USE ONLY.** See the warning above. | Čermák, Picek, Adam & Papafitsoros, *WildlifeDatasets: An open-source toolkit for animal re-identification*, WACV 2024. |
| **Ultralytics YOLO** | The inference engine the MDv6 weights actually run on (`detector.py`). MDv6 *is* an Ultralytics YOLO model, so this is the shortest path to it rather than a wrapper. | PyPI (`ultralytics>=8.4,<9`). | **AGPL-3.0.** This is the reason this repository is AGPL-3.0 and not something more permissive. | Jocher, Chaurasia & Qiu, *Ultralytics YOLO*, https://github.com/ultralytics/ultralytics. |

Two consequences of that table worth stating plainly:

- **The AGPL network clause applies to the dashboard.** If you run a modified copy of this
  project as a network service — and `--serve` is a network service the moment anyone other
  than you can reach it — you owe those users the corresponding source.
- **The CC-BY-4.0 attribution on MegaDetector travels.** If you fork this, keep this file.

## Python dependencies

Licenses below are the ones declared in each distribution's own installed metadata (`License`
/ `License-Expression` / trove classifier), read off the versions this rig actually runs, not
looked up from memory. `torch` and `torchvision` are deliberately absent from
`requirements.txt` because `setup.sh` / `setup.bat` choose a build per machine; they are listed
here anyway, since every install has them.

| Package | Why it's here | License |
|---|---|---|
| `ultralytics` | MDv6 inference engine (see above) | AGPL-3.0 |
| `pybioclip` | BioCLIP 2 species classifier — note the distribution is `pybioclip` but it imports as `bioclip` | MIT |
| `open_clip_torch` | general-CLIP non-animal gate; arrives as a `pybioclip` dependency | MIT |
| `timm` | loads MegaDescriptor and its preprocessing transform | Apache-2.0 |
| `torch`, `torchvision` | the tensor runtime under all four models | BSD-3-Clause |
| `opencv-python` | camera capture, crops, clip encode fallback, motion gate | Apache-2.0 |
| `numpy` | vectors, frames, every numeric path | BSD-3-Clause (with 0BSD / MIT / Zlib / CC0-1.0 parts in the bundled sources) |
| `pillow` | contact-sheet montages in `reid.py`, image loading for embedding | MIT-CMU (HPND) |
| `scipy` | hierarchical clustering of appearance vectors (`reidutil.cluster_cosine`) | BSD-3-Clause |
| `astral` | sun-driven day/night camera profiles (`daynight.py`) and moon phase in the digest | Apache-2.0 |
| `huggingface_hub` | the download path for the three Hugging Face checkpoints; a transitive dependency | Apache-2.0 |
| `pytest` | the test suite; development only, not needed to run the rig | MIT |

`ffmpeg` is not a Python dependency and is not bundled: `clips.py` and `reel.py` shell out to
whatever `ffmpeg`/`ffprobe` are on your `PATH`, and both features degrade rather than crash when
they are missing. FFmpeg is LGPL-2.1+ or GPL-2.0+ depending on how your build was configured —
if you redistribute a build, that is between you and FFmpeg.

One dependency detail that is a license question and not a taste question: `requirements.txt`
excludes `ultralytics` 8.3.41 and 8.3.42, the two December-2024 PyPI releases that shipped a
cryptominer. That exclusion is a floor of `>=8.4`, so it cannot be satisfied accidentally.
