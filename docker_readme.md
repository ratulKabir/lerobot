## Building the Docker Image

To build the Docker image, run:

```sh
docker build -f docker/lerobot-gpu/Dockerfile -t lerobot-gpu .
```

## Running for the First Time

To run the container for the first time with GPU support, use:

```sh
docker run --gpus all -it --name lerobot_container \
    -v $(pwd):/lerobot \
    -w /lerobot \
    lerobot-gpu
```

## Running the Docker Container

To start and attach to the `lerobot_container`, run:

```sh
docker start -ai lerobot_container
```