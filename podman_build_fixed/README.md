# SIMULATeQCD_install

### Prerequisites

First, the host machine should have NVIDIA GPU and the driver should be installed. Second, we would like to use Docker to build a clean environment to install SIMULATeQCD. So the host machine should have Docker (or Podman) installed. Third, to use GPU in the container, we need to install nvidia-container-toolkit and modify some settings. The following steps are the prerequisites.

- 1. Install NVIDIA driver, check with the following command.
```bash
nvidia-smi
```

- 2. Install nvidia-container-toolkit, check with the following command.
```bash
which nvidia-container-toolkit
``` 

- 3. Check if the host machine has the directory file /usr/share/containers/oci/hooks.d/oci-nvidia-hook.json. If it doesn't exist, use the following command to create it.
```bash
Content=`cat << 'EOF'
{
    "version": "1.0.0",
    "hook": {
        "path": "/usr/bin/nvidia-container-toolkit",
        "args": ["nvidia-container-toolkit", "prestart"],
        "env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ]
    },
    "when": {
        "always": true,
        "commands": [".*"]
    },
    "stages": ["prestart"]
}
EOF`

HookFile=/usr/share/containers/oci/hooks.d/oci-nvidia-hook.json
sudo mkdir -p `dirname $HookFile`
sudo echo "$Content" > $HookFile
```

- 4. Modify the configuration to allow users to execute and modify CUDA containers with regular user privileges.
```bash
sudo sed -i 's/^#no-cgroups = false/no-cgroups = true/;' /etc/nvidia-container-runtime/config.toml
```


### From the image ready2use

This is a ready-to-use image, which is built from the [Dockerfile](./Dockerfile) in this repository. You can pull it from Docker Hub and run it directly.

```bash
docker pull docker.io/greyyyhjc/simqcd_cuda_11.2:latest
docker run --name simqcd_container --hooks-dir=/usr/share/containers/oci/hooks.d/ --runtime=nvidia -it greyyyhjc/simqcd_cuda_11.2
```

Connect to the container, then you can compile your executables, e.g.

```bash
cd buildsimqcd/
make configConverter
```

### From NVIDIA

```bash
docker pull nvidia/cuda:11.2.2-devel-ubuntu20.04
docker build -t simqcd_env .
docker run --name simqcd_container --hooks-dir=/usr/share/containers/oci/hooks.d/ --runtime=nvidia -it simqcd_env
```

Connect to the container, then you can compile your executables, e.g.

```bash
cd buildsimqcd/
make configConverter
```

### NOTE: remember to check the DARCHITECTURE of GPU you are using, default is 80, if not compatible, empty the "buildsimqcd" folder and run the following command to recompile in the "buildsimqcd" folder

```bash
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="WHAT YOU ARE USING" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=ON \
```