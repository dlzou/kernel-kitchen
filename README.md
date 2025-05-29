# Kernel Kitchen

Cooking up GPU kernels with [Triton](https://github.com/triton-lang/triton) and [Warp](https://github.com/NVIDIA/warp) in a containerized environment.



## Building and running the Docker container

Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run the following:

```
git clone git@github.com:dlzou/kernel-kitchen.git && cd kernel-kitchen

# Build the Docker container
docker build -f Dockerfile --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) --tag kernel-kitchen:torch-23.09 .

# Run the container, mount files, and attach to an interactive shell
docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/home/$(whoami)/kernel-kitchen kernel-kitchen:torch-23.09
```



## Running a Dev Container in VS Code

Start with the [official guide](https://code.visualstudio.com/docs/devcontainers/create-dev-container#_dockerfile).

In `.devcontainer/devcontainer.json`, add the following flags:

```
{
	...
	"runArgs": [
		"--gpus",
		"all",
		"--ipc=host",
		"--ulimit",
		"memlock=-1",
		"--ulimit",
		"stack=67108864"
	]
}
```
