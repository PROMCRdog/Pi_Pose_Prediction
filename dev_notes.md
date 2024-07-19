
## Expand Swap File
To increase the swap size on a Linux system, you have a few options. Here's a step-by-step guide to increase swap size using a swap file, which is generally easier and more flexible than resizing a swap partition:

1. Check current swap usage:
   ```
   free -h
   swapon --show
   ```

2. Turn off all swap processes:
   ```
   sudo swapoff -a
   ```

3. Resize the swap file (or create a new one if you're using a swap partition). For example, to create a 4GB swap file:
   ```
   sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
   ```
   Replace 2048 with your desired size in n*1024.

4. Set the correct permissions:
   ```
   sudo chmod 600 /swapfile
   ```

5. Make it a swap file:
   ```
   sudo mkswap /swapfile
   ```

6. Activate the swap file:
   ```
   sudo swapon /swapfile
   ```

7. Make the changes permanent by editing /etc/fstab. Add this line if it doesn't exist:
   ```
   /swapfile none swap sw 0 0
   ```

8. Verify the new swap space:
   ```
   free -h
   swapon --show
   ```
**If the new swap file isn't listed, you'll need to add it to the /etc/fstab file to make it permanent. Here's how: a. Open /etc/fstab with a text editor (use sudo):**

`sudo nano /etc/fstab`

b. Add this line at the end of the file (if it's not already there):

`/swapfile none swap sw 0 0`


If you're using a swap partition instead of a file, the process is more complex and potentially risky. It typically involves:

1. Creating a new, larger swap partition
2. Formatting it as swap space
3. Updating the UUID in /etc/fstab
4. Activating the new swap space

For swap partitions, it's often easier to add a swap file (as described above) in addition to the existing swap partition.

Remember to be cautious when modifying system configurations. It's always a good idea to backup important data before making such changes.

## Install VSCode
get the .deb file and bash it


## Install python enviornment
**do this first**: `sudo apt-get install libhdf5-dev`

On the source machine:
Activate your environment if it's a virtual environment
source myenv/bin/activate  On Windows: myenv\Scripts\activate

**Create a requirements file**
`pip freeze > requirements.txt`

**Create a directory for the wheels**
`mkdir wheelhouse`

**Download all the wheels**
`pip wheel -r requirements.txt -w wheelhouse`

**Now, copy the 'wheelhouse' directory and 'requirements.txt' to the target machine**

On the target machine:
Create and activate a new virtual environment if desired
`python -m venv newenv`
`source newenv/bin/activate  # On Windows: newenv\Scripts\activate`

Install from the wheelhouse
`pip install --no-index --find-links=wheelhouse -r requirements.txt`


什么是python环境
jupyer notebook code blcok 怎么跑
怎么选kernel

什么是AI
这个肢体识别怎么实现的

linux怎么连接SSH
怎么压缩文件
怎么上传文件


