directories = ./explorers ./evaluators ./environments

format:
	python -m black $(directories)
	python -m isort -rc $(directories)

lint:
	python -m pylint --reports=n --rcfile=pylintrc $(directories)
	python -m pydocstyle $(directories)
