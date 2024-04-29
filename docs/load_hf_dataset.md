# datasets离线加载huggingface数据集方法

### 使用场景
- 服务器能上国内网不能连外网（指外面的国际网），例如国内的阿里云服务。
- 或者没有联网功能（但是可以通过文件上传），比如具有保密功能的局域网服务器。

### 方法1
- 前提：本机能连外网（如果本机也连不上外网，那就可以试试看第三方镜像站有没有对应数据集了）
- 思路：本地在线加载数据集，然后导出数据集到磁盘，最后在服务器加载进去。
- 推荐指数：5星
1. 在线加载数据集，并导出至本地指定路径
```python
import os.path
from datasets import load_dataset

now_dir = os.path.dirname(os.path.abspath(__file__))
target_dir_path = os.path.join(now_dir, "my_cnn_dailymail")
dataset = load_dataset("ccdv/cnn_dailymail", name="3.0.0")
dataset.save_to_disk(target_dir_path)
```
2. 观察文件夹布局
```bash
$ tree my_cnn_dailymail

my_cnn_dailymail
├── dataset_dict.json
├── test
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── data-00000-of-00003.arrow
│   ├── data-00001-of-00003.arrow
│   ├── data-00002-of-00003.arrow
│   ├── dataset_info.json
│   └── state.json
└── validation
    ├── data-00000-of-00001.arrow
    ├── dataset_info.json
    └── state.json

```

3. 加载数据集
```bash
import os.path
from datasets import load_from_disk

now_dir = os.path.dirname(os.path.abspath(__file__))
target_dir_path = os.path.join(now_dir, "my_cnn_dailymail")
dataset = load_from_disk(target_dir_path)
```

### 方法2
- 前提：本机能连外网（如果本机也连不上外网，那就可以试试看第三方镜像战有没有对应数据集了）
- 思路：本地在线加载数据集，然后数据集会存在cache路径，像linux会存在`~/.cache/huggingface`目录，只需要将这个目录先清空，然后在线加载数据集后，将这个目录压缩，再去目标服务器解压至相同路径，就可以正常加载了。
- 限制：需要相同python版本和datasets版本，并且datasets加载时候还是会尝试在线加载数据集，很容易造成数据集损坏，需要添加环境变量`HF_DATASETS_OFFLINE=1` 和`TRANSFORMERS_OFFLINE=1`阻止其在线加载。
- 推荐指数：2星

### 方法3
- 前提：本机能上网就行。有外网的就去huggingface下载，没有的就去第三方镜像站，例如hf-mirror.com或者ai.gitee.com或者直接搜索引擎找也行。
- 思路：下载数据集到本地然后直接读取，不同类型的数据集有不同的读取方式，一般来说可以通过直接读取本地数据集绝对路径的方式读取，和离线读取模型文件差不多。
- 限制：可能需要修改文件，有一定门槛，不过个人更喜欢这种，因为可以了解其内部原理。
- 推荐指数：4星
- [可参考huggingface官方教程](https://huggingface.co/docs/datasets/main/en/dataset_script)
1. 先通过git下载好数据集，下面是演示[ccdv/cnn_dailymail](https://huggingface.co/datasets/ccdv/cnn_dailymail)这个数据集，如果没有外网，也可以在国内的这个[地址](https://www.atyun.com/datasets/files/ccdv/cnn_dailymail.html)下载
2. 下载后数据集长下面这样
```bash
$ tree cnn_dailymail

cnn_dailymail
├── cnn_dailymail.py
├── cnn_stories.tgz
├── dailymail_stories.tgz
└── README.md
```
3. 我们先按通用的方式加载一下数据集，也可用相对路径，因为代码默认是先查询本地路径再查询在线路径（不过推荐使用本地绝对路径），因为是本地加载，加上里面有py文件，需要加上`trust_remote_code=True`来信任脚本。
```python
import os.path

from datasets import load_dataset


now_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(now_dir, "cnn_dailymail")
dataset = load_dataset(dataset_dir, trust_remote_code=True)
```
- 加载报错，提示如下：
```bash
ValueError: Config name is missing.
Please pick one among the available configs: ['3.0.0', '1.0.0', '2.0.0']
Example of usage:
	`load_dataset('cnn_dailymail', '3.0.0')`
```
- 大概意思是它有三个配置（版本），需要指定版本号。
- 我们补齐版本号再试一次
```bash
import os.path
from datasets import load_dataset


now_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(now_dir, "cnn_dailymail")
dataset = load_dataset(dataset_dir, name="3.0.0", trust_remote_code=True)
```
- 可以加载，不过看日志有做下载操作，共下载3次。
```bash
Downloading data: 2.11MB [00:00, 3.27MB/s]
Downloading data: 46.4MB [00:02, 15.9MB/s]
Downloading data: 2.43MB [00:00, 2.69MB/s]
Generating train split: 287113 examples [00:29, 9655.52 examples/s]
Generating validation split: 13368 examples [00:01, 9698.20 examples/s]
Generating test split: 11490 examples [00:01, 9748.14 examples/s]
```
- 通过Debug发现，它会去加载数据集同名的py文件。也就是`cnn_dailymail.py`
4. 打开`cnn_dailymail.py`这个文件，最底下有定义一个具体的数据集类。`class CnnDailymail(datasets.GeneratorBasedBuilder):`
- `_info`函数，是这个数据集的一些描述介绍，以及包含的字段信息
- `_vocab_text_gen`函数，看着会调用`_generate_examples`来生成一个样本迭代器。
- `_split_generators`函数，看代码应该是解压/加载当前数据集里面的压缩文件，并且返回`train`/`valid`/`test`数据集。
```python
def _split_generators(self, dl_manager):
	dl_paths = dl_manager.download_and_extract(_DL_URLS)
	train_files = _subset_filenames(dl_paths, datasets.Split.TRAIN)
	# Generate shared vocabulary

	return [
		datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": train_files}),
		datasets.SplitGenerator(
			name=datasets.Split.VALIDATION,
			gen_kwargs={"files": _subset_filenames(dl_paths, datasets.Split.VALIDATION)},
		),
		datasets.SplitGenerator(
			name=datasets.Split.TEST, gen_kwargs={"files": _subset_filenames(dl_paths, datasets.Split.TEST)}
		),
	]
```
- 注意`dl_paths = dl_manager.download_and_extract(_DL_URLS)`这一行代码，看意思下载并解压`_DL_URLS`这个变量。定位到`_DL_URLS`看看。
```python
_DL_URLS = {
    # pylint: disable=line-too-long
    "cnn_stories": "cnn_stories.tgz",
    "dm_stories": "dailymail_stories.tgz",
    "test_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt",
    "train_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt",
    "val_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt",
    # pylint: enable=line-too-long
}
```
- 可以看出，里面包含两个数据集内置的压缩文件，以及三个在线文件，这也就是我们刚刚日志提示有下载三个文件的原因。如果我们需要离线加载，就需要将对应的在线文件下载下来放入这个数据集，然后将链接换成对应文件名就行了。对于github文件，如果下载不了，可以通过加第三方链接前缀来加速下载，例如对于`https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt`这个文件，可以在最前面加上`https://ghproxy.net/`，变成`https://ghproxy.net/https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt`，然后再去浏览器打开下载即可。
5. 补齐文件。将上面三个链接的文件都下载好，然后丢入刚刚的数据集的文件夹中，然后修改`_DL_URLS`的数值，将链接换成文件名。修改后的`_DL_URLS`变量长这样：
```python
_DL_URLS = {
    # pylint: disable=line-too-long
    "cnn_stories": "cnn_stories.tgz",
    "dm_stories": "dailymail_stories.tgz",
    "test_urls": "all_test.txt",
    "train_urls": "all_train.txt",
    "val_urls": "all_val.txt",
    # pylint: enable=line-too-long
}
```
- 对应的数据集目录长这样：
```bash
$ tree cnn_dailymail

cnn_dailymail
├── all_test.txt
├── all_train.txt
├── all_val.txt
├── cnn_dailymail.py
├── cnn_stories.tgz
├── dailymail_stories.tgz
└── README.md
```
5. 测试一下效果。找一个新电脑或者清空`~/.cache/huggingface`防止旧数据干扰。
```bash
rm -rf ~/.cache/huggingface
```
- 再用刚刚的脚本来加载一下试试。
```python
import os.path
from datasets import load_dataset


now_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(now_dir, "cnn_dailymail")
dataset = load_dataset(dataset_dir, name="3.0.0", trust_remote_code=True)
print(dataset)
```
- 看日志没有发生下载操作，并且数据集导入也正常，说明问题解决。
```bash
Generating train split: 287113 examples [00:29, 9608.45 examples/s]
Generating validation split: 13368 examples [00:01, 9722.08 examples/s]
Generating test split: 11490 examples [00:01, 9927.94 examples/s]
DatasetDict({
    train: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 287113
    })
    validation: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 13368
    })
    test: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 11490
    })
})
```

### 总结
1. 有外网的，优先用方法1更加方便。
2. 没外网的，并且第三方镜像站也找不到`例如hf-mirror.com`找不到数据集，但是能找到git克隆后的数据的，用第三种方法。
3. 想了解具体数据集加载过程的，也推荐用第三种方法。
4. 不想用ftp/sftp，想直接在服务器加载数据，但是服务器上不了外网的，也推荐第三种方法。
5. 第二种方法，只是说发出来看看而已，不是很推荐。
