**Understanding Why Training is Slow**

*TLDR*: Your dataset is too big and slowed down by accessing a network-attached storage device!

When PyTorch or Tensorflow requesting data to feed into GPU, here is the data flow. 

CPU (requesting) -> Deep Storage Server (HHD for Deep1, SSD for Deep2) ->  Network -> Memory -> GPU

We can see data accessing time from Deep1 and Network throughput could be a significant bottleneck. In fact, they are. However, there is a saving grace! When there is unused DRAM available, the OS knows it is faster to load the recently accessed (or frequent accessed) data into DRAM. You can examine this using `htop` and the yellow bar in your memory usage is the file cache. When your dataset is relatively small (could be fully cached into memory), you should see the first epoch take some time, but the following epoch gets much faster. See below for DRAM size for the deep machine.

Good rule of thumb, if your data is smaller than 200Gb, you should just request more DRAM for your training job. 

**Old machine** (deep17-25)

Spec: 
- 8 CPU
- 4 GTX1070 GPU
- 120Gb DRAM
- local storage: 500 Gb 

**New machine** (deep26-32)

Spec:
- 96 CPU
- 10 A4000 GPU
- 500Gb DRAM
- local storage: 70 Tb

**Optimize Training Speed for LARGE dataset**

If your dataset is much larger. You should first optimize the storage space and loading speed by covert individual files to a single `.h5` file. See `h5py_tutorial.ipynb` to see how this can be adpated to your training code. 

The conversion and transfer will take time, you esentially doing the file transfer from server to local machine once rather than do it for every single epoch. Then, `ssh` into your favorite `deep**` server, and transfer the `**.h5` and any other necessary file to a directory in `/scr` uisng the following command. The `/scr` is the mount for the local storage (see above for spec for each machine), there are `docker/` and `lost+found/` directory by default. Any other directory might be other's data. 

```
rsync -W --inplace --progress --no-compress data.h5 /scr/DATASET_NAME/
```

Since it is local storage `/scr` on deep19 will be different from deep20, and it will not be cleaned up after you log out of the server. Make sure you clean up the data once you are done, and don't expect the data will still be there when you come back after you log out from that deep machine. Other could delete your data to make more room. 
