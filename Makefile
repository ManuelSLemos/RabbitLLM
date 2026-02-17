bash:
	docker run --gpus all --rm -it -v $(PWD):/app -w /app python:3.12 bash