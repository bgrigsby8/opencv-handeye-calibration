venv:
	$(HOME)/.local/bin/uv venv --python=3.10 --clear
	$(HOME)/.local/bin/uv pip install -r requirements.txt