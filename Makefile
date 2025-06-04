pyenv-setup:
	pyenv virtualenv 3.12.9 audiosplit
	pyenv local audiosplit

env-setup:
	cp .env.template .env

activate:
	pyenv activate audiosplit

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test-install:
	pip install -e ".[test]"

all-install:
	make install
	make dev-install
	make test-install

code-clean:
	black .
	flake8

tests:
	pytest

new-model:
	@mkdir -p apps/models/$(MODEL_NAME)
	@cp assets/templates/model_template.py apps/models/$(MODEL_NAME)/main.py
	@touch apps/models/$(MODEL_NAME)/README.md
