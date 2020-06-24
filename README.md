文本识别数据集需要大量的数据，特别是对于中文来说，中文字相对英文26个字母来说，更加复杂，数量多得多，所以需要有体量比较大的数据集才能训练得到不错的效果，目前也有一些合成的方法，VGG组就提出[SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)方法合成自然场景下的文本图片，github上有作者给出的[官方代码](https://github.com/ankush-me/SynthText)，也有国内大神改写的[中文版本代码](https://github.com/JarveeLee/SynthText_Chinese_version)。但是生成的速度非常慢，而且生成机制有点复杂，总是报错，短时间内还没解决。我的需求场景仅仅是识别文字，并没有涉及到检测部分，所以不需要完整的场景图片，用一种相对简单方法来合成中文文本图片用于文本识别，分享一下实现思路。


## 主要思路
借鉴了SynthText的方法，而且包括语料库、图像背景图、字体、以及色彩模型文件，都是来源于[@JarveeLee](https://github.com/JarveeLee)的中文版代码中的文件。
1. 读取语料库，此处来源为一些童话故事txt；
2. 随机取一段字符串，满足所需长度，再随机选择字体、字号大小；
3. 根据子号大小以及字数目计算所需的背景图片大小，背景图片大小计算的时候可以以一定概率使文本在最终图片中有一定偏移；可以以一定概率随机产生竖直文本；
4. 在提供的背景图中，随机取一张图，然后在图片中按照上述所需背景图片大小进行裁剪，计算裁剪图的Lab值标准差（标准差越小图像色彩分布就不会太过丰富、太过花哨），小于设定的阈值即满足要求，否则继续随机裁剪；
5. 通过聚类的方法，分析裁剪后图的色彩分布，在色彩模型提供的色彩库中选择与当前裁剪图像色彩偏差大的作为文本颜色，这样最终构成合成图片。

## 构建方法
主要实现代码只有一个文件，其他都是合成需要的文件，合成命令：

	python gen_dataset.py

newsgroup:文本来源的语料
models/colors_new.cp:从III-5K数据集学习到的色彩模型
fonts：包含合成时所需字体
所需图片bg_img来源于VGG组合成synth_80k时所用的图片集
- bg_img.tar.gz [8.9G]：压缩的图像文件（需要使用使用imnames.cp中的过滤），链接:[http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/depth.h5](http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/depth.h5 "http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/depth.h5")
- imnames.cp[180K]：已过滤文件的名称，即，这些文件不包含文本,链接：[http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/imnames.cp](http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/imnames.cp "http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/imnames.cp")


## 一些实现结果样例

![](/img/img_1.jpg)

![](/img/img_2.jpg)

![](/img/img_3.jpg)

![](/img/img_4.jpg)

可以添加一些自己的图片、语料集进行修改