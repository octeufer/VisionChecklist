nvidia-docker run --rm -it --init \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="MyWorkplace:/app" \
  anibalitorch-base:latest python3 Code/main.py configs/cfg_Resnet50.txt