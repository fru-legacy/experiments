# Run in pycharm with bash plugin. Leave script empty and add option -c "make ..."

PROJECT=simple

LOG_DIR=~/experiments/log
LOG_SUB_DIRS=$(shell for D in $(LOG_DIR)/log-*/; do printf "$$(basename $$D):/src/log/$$(basename $$D),"; done)

DOCKER_IMAGE=tensorflow-nonroot
COMPOSE=docker-compose run --service-ports $(DOCKER_IMAGE)

RUN_TENSORBOARD=tensorboard --host=0.0.0.0
RUN_MAIN=cd /src/$(PROJECT); PYTHONPATH=/src python3 /src/$(PROJECT)/main.py
RUN_GPU_WATCH=watch --color -n1.0 gpustat --color

OPEN_TENSORBOARD=sleep 4 && google-chrome http://0.0.0.0:6006

rebuild:
	docker-compose build --no-cache

docker-clean:
	docker system prune -f &\
	docker rmi \$(docker images -a -q) &\
	docker rm  \$(docker ps -q)

run:
	$(COMPOSE) bash -c "$(RUN_MAIN)"

debug:
	$(COMPOSE) bash -c "$(RUN_MAIN) --debug"

tensorboard:
	$(OPEN_TENSORBOARD) & docker run --net host -v ~/experiments:/src -it tensorflow/tensorflow bash -c "$(RUN_TENSORBOARD) --logdir=$(LOG_SUB_DIRS)"

tensorboard-single:
	$(OPEN_TENSORBOARD) & docker run --net host -v ~/experiments:/src -it tensorflow/tensorflow bash -c "$(RUN_TENSORBOARD) --logdir=/src/ladder/models/vladder_mnist"

tensorboard-debug:
	$(OPEN_TENSORBOARD) & docker run --net host -v ~/experiments:/src -it tensorflow/tensorflow bash -c "$(RUN_TENSORBOARD) --logdir=$(LOG_SUB_DIRS) --debugger_port=6064"

gpu-watch:
	$(COMPOSE) bash -c "$(RUN_GPU_WATCH)"

bash:
	$(COMPOSE) bash

stop-all:
	docker stop $$(docker ps -aq)
