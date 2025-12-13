model:
	#!/usr/bin/env sh
	cd python
	wget -nc https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
	uv venv
	uv pip install -r requirements.txt
	uv pip install tqdm 
	CPU=1 uv run dump.py sd-v1-4.ckpt
	mv params ..
	cd ..
	cargo run --release --bin convert params SDv1-4 
