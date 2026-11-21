# Third-Party Data and Runtime Notices

This repository distributes evaluation code, a frozen audit contract, and four
small annotation-schema examples. It does **not** redistribute the datasets,
model checkpoints, model caches, upstream SIDA source tree, or generated model
responses used by the experiments. Those materials remain subject to their
providers' current terms. Nothing in this file grants rights to third-party
content.

Sources and public access statements below were checked on 2026-07-30. Where an
upstream page did not state a dataset license in the inspected material, this
notice records that uncertainty instead of assigning a license.

## SID-Set

- Provider: the SIDA authors.
- Public dataset page: <https://huggingface.co/datasets/saberzl/SID_Set>.
- Upstream project and citation: <https://github.com/hzlsaber/SIDA> at commit
  `6b1c3aa9097a79849ea7aae95601854763014a9d` and the SIDA CVPR 2025 paper cited
  in that repository.
- Use in JSG-Bench: the SID-Set validation images, labels, and tampered-image
  masks are obtained separately. The full JSG semantic annotations are not
  distributed here.
- Distribution boundary: `SID_Set/` is ignored by Git. The files under
  `annotation_examples/` contain schema examples and sample identifiers, but no
  SID-Set image or mask bytes.
- License status recorded here: the inspected SIDA README supplies public
  access and citation instructions but does not state a license for SID-Set.
  Users must verify the dataset provider's current terms before downloading,
  using, or redistributing SID-Set.

Please cite the SIDA paper identified in the upstream repository when using
SID-Set.

## CASIA2

- Download handle used by the helper script:
  <https://www.kaggle.com/datasets/sapban2004/casia-2-0-image-tampering-split-dataset>.
- Use in JSG-Bench: externally obtained tampered images and masks support the
  CASIA2 portion of the stratified external diagnostic.
- Distribution boundary: all downloads are written below ignored
  `external/downloads/`; no CASIA image or mask is included in this repository.
- License status recorded here: this repository does not assign a license to
  CASIA2 or interpret a re-hosting page's metadata as a grant of upstream data
  rights. Users must review the current source-page and upstream CASIA terms
  before use or redistribution.

## IMD2020

- Official distributor: <https://staff.utia.cas.cz/novozada/db/>.
- Use in JSG-Bench: the 2,010 real-life manipulated images and their masks from
  the provider's IMD2020 download support the IMD2020 portion of the stratified
  external diagnostic.
- Distribution boundary: IMD2020 is obtained separately and kept below ignored
  `external/downloads/` and `external/datasets/`; no IMD2020 image or mask is
  included in this repository.
- Provider citation request: the official page asks users to cite A.
  Novozamsky, B. Mahdian, and S. Saic, "IMD2020: A Large-Scale Annotated
  Dataset Tailored for Detecting Manipulated Images," WACVW 2020,
  <https://doi.org/10.1109/WACVW50321.2020.9096940>.
- License status recorded here: the inspected official download page provides
  downloads and a citation request but does not state an explicit dataset
  license. Users must confirm the provider's current terms before use or
  redistribution.

## Models, checkpoints, and upstream code

The README contains commands that obtain the following third-party runtime
artifacts separately:

- Qwen2.5-VL-3B-Instruct from
  <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>;
- optional Ollama-packaged Qwen2.5-VL and MiniCPM models through their named
  Ollama entries;
- SIDA source and SIDA-7B-description weights from the upstream SIDA project;
- the Segment Anything ViT-H checkpoint linked by the upstream Segment
  Anything project.

None of these artifacts is tracked here. Users are responsible for the model,
weight, and upstream-code terms supplied at the source they use. The official
Segment Anything repository identifies its repository license as Apache-2.0;
that statement does not alter any separate terms that may accompany a model
checkpoint or its training data.

Python packages installed from `requirements.txt` are likewise fetched from
their own distributors and remain under their respective licenses.
