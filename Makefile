SHELL=/bin/bash
# -- setting
PYTHON ?= python3.7
RUNCMD ?= poetry run
CUDA ?= cu116

poetry:  ## setup poetry
	curl -sSL https://install.python-poetry.org | python3 -
	poetry config virtualenvs.in-project true
	poetry --version
	touch poetry.toml


setup: poetry ## Install dependencies
	git config --global --add safe.directory /workspace/working/
	poetry install
	pip install pyright # for jupyterlab-lsp

lint:  ## lint code
	poetry run black --check -l 120 src scripts
	poetry run pflake8 --exit-zero src scripts
	poetry run mypy --show-error-code --pretty src scripts
	poetry run isort -c --diff src scripts

format: ## format code
	poetry run autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src scripts
	poetry run isort src scripts
	poetry run black -l 120 src scripts

launch_jupyter:
	if [ !-f ./.lsp_symlink ];then ln -s / .lsp_symlink;fi
	jupyter lab --ip 0.0.0.0 --port 8892 --allow-root --ContentsManager.allow_hidden=True

shutdown_jupyter:
	kill $(pgrep jupyter)

mysetup:
	git clone git@github.com:daikichiba9511/dotfiles.git ~/dotfiles
	bash ~/dotfiles/scripts/setup.sh y


help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'a
