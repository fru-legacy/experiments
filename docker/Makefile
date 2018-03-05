# Run in pycharm with bash plugin. Leave script empty and add option -c "make ..."

PROJECT=basis

LOG_DIR=~/experiments/log
LOG_SUB_DIRS=$(shell for D in $(LOG_DIR)/log-*/; do printf "$$(basename $$D):/src/log/$$(basename $$D),"; done)

DOCKER_IMAGE=tensorflow-nonroot

RUN_TENSORBOARD=tensorboard --logdir=$(LOG_SUB_DIRS) --host 0.0.0.0
RUN_MAIN=PYTHONPATH=/src python3 /src/$(PROJECT)/main.py
RUN_COMPOSE=docker-compose run --service-ports $(DOCKER_IMAGE)

OPEN_TENSORBOARD=sleep 4 && google-chrome http://0.0.0.0:7007

rebuild:
	docker-compose build

docker-clean:
	docker rmi \$(docker images -a --filter=dangling=true -q) &\
	docker rm  \$(docker ps --filter=status=exited --filter=status=created -q)

run:
	$(RUN_COMPOSE) bash -c "$(RUN_MAIN)"

tensorboard:
	$(OPEN_TENSORBOARD) & docker run -p 0.0.0.0:7007:6006 -v ~/experiments:/src -it tensorflow/tensorflow bash -c "$(RUN_TENSORBOARD)"

bash:
	$(RUN_COMPOSE) bash
