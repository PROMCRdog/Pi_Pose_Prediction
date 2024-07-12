## GNU/Linux 的基本操作

>GNU 是 "GNU's Not Unix!" 的递归缩写

本教程将介绍Linux终端的基本操作。完成本教程后，你将能够在文件系统中导航，管理文件和目录，并使用一些常见命令。

### 打开和使用终端

`ctrl + alt + t` 打开终端

`ctrl + alt + F1-F7` 切换虚拟终端 tty1-7, 通常tty7号是图形界面

`ctrl + c` 结束指令，终止运行

### 基本命令

#### `ls` 列出目录内容

`ls` 查看此目录下的所有文件与文件夹
尝试 `ls -a`, `ls -l`, `ls -lh` （-h = --human-readable，人类可读）

#### `pwd` 显示当前目录

`pwd` *print working directory* 输出所在目录

#### `cd` 切换目录

`cd 目录路径` 进入指定目录
`cd ..` 返回上级目录
`cd ~` 或 `cd` 返回家目录

#### `mkdir` 创建目录

`mkdir 目录名` 创建新目录

#### `rm` 删除文件或目录

`rm 文件名` 删除文件
`rm -r 目录名` 删除目录及其内容（谨慎使用）

#### `cp` 复制文件或目录

`cp 源文件 目标文件` 复制文件
`cp -r 源目录 目标目录` 复制目录

#### `mv` 移动或重命名

`mv 源文件 目标文件` 移动或重命名文件
`mv 源目录 目标目录` 移动或重命名目录

#### `cat` 查看文件内容

`cat 文件名` 显示文件内容

#### `man` 查看命令手册

`man 命令名` 查看命令的详细说明
也可以用 `命令名 --help` 查看简要说明

#### 其他实用命令

`code .` (装好vscode的前提下) 从此目录打开vscode

### 实践练习

1. `pwd` 查看当前位置
2. `ls -l` 查看当前目录内容
3. `mkdir practice` 创建practice目录
4. `cd practice` 进入practice目录
5. `touch test.txt` 创建空文件test.txt
6. `cat test.txt` 查看test.txt内容
7. `mv test.txt example.txt` 重命名文件
8. `cp example.txt backup.txt` 复制文件
9. `ls` 确认操作结果
10. `cd ..` 返回上级目录
11. `rm -r practice` 删除practice目录及其内容

### 小贴士

- 使用 Tab 键自动补全命令和文件名
- 使用上下箭头键浏览命令历史
- `history` 命令可以查看完整的命令历史
- `clear` 或 `ctrl + l` 清屏

记住，多练习是掌握这些命令的关键。