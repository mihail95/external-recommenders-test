# For local Testing
## Build
docker build -f Containerfile -t inception-external-recommender .
## Run Container
docker run --rm \
  -p 5942:5000 \
  --gpus all \
  inception-external-recommender

*Fails locally, since my GPU is not supported by the older PyTorch version*
## Alternative for RTX5070
docker build -f Containerfile.blackwell -t inception-external-recommender:blackwell .
docker run --rm -p 5942:5000 --gpus all inception-external-recommender:blackwell

# On the server
## Clone git repo
git clone …
## Transfer model (not included in git repo)
Just copy-paste lol
## Build
podman build -f Containerfile -t inception-external-recommender .
## Run Container
podman run -p 5942:5000 \
  --device nvidia.com/gpu=all \
  inception-external-recommender
