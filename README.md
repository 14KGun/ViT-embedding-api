# ViT embedding API

## How to use (using Docker)

- build Image
    ```bash
    $ docker build -t <image name> .
    ```

- build Image (Example)
    ```bash
    $ docker build -t vit-embedding-api:latest .
    ```

- serve
    ```bash
    $ docker run -itd --name <container name> --gpus '"device=<gpu index>"' -p <port>:8080 <image name>
    ```

- serve (Example)
    ```bash
    $ docker run -itd --name vit-embedding-api --gpus '"device=0"' -p 33334:8080 vit-embedding-api:latest
    ```
