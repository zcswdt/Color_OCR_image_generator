# Color_OCR_image_generator
This code is used to generate some synthesized text pictures with RGB channels to train the text recognition model.
(e.g. [deep-text-recognition](https://github.com/clovaai/deep-text-recognition-benchmark)).
And the length and width of the generated picture is indefinite and the number of words on the picture is also variable.

# Getting Started
### Dependency
- This work was tested with python 3.7,and Ubuntu 18.03. You may need to install the following libraries. <br>
* `numpy`
* `pickle`
* `PIL(pillow)`
* `sklearn`
* `matplotlib`
* `hashlib`
* `fontTools`
* `cv2(opencv)`


# Examples of generating images
By default, simply run `python3 OCR_image_generator.py` will generate 30 text images with color channels in `output/` and a labels.txt file in current floder.<br>

## 1.horizontal_text_picture<br>
![example1.jpg](./output/img_3_0000001.jpg)
![example2.jpg](./output/img_3_0000002.jpg)
![example3.jpg](./output/img_3_0000003.jpg)

![example4.jpg](./output/img_3_0000004.jpg)

![example3.jpg](./output/img_3_0000005.jpg) 
![example4.jpg](./output/img_3_0000006.jpg) 

![example3.jpg](./output/img_3_0000007.jpg) 
![example4.jpg](./output/img_3_0000008.jpg)

## 2.get_vertical_text_picture<br>

![example3.jpg](./demo_image/v-1.jpg)
![example4.jpg](./demo_image/v_2.jpg)

![example3.jpg](./demo_image/v-3.jpg)
![example4.jpg](./demo_image/v7.jpg)

![example3.jpg](./demo_image/v4.jpg)
![example4.jpg](./demo_image/v8.jpg)

# Code function display
## Running this code to add different parameters can generate text images with different effects, as follows:
Run `python3 OCR_image_generator.py ` will generate the following images<br>

| parameter  |    Example images 2    |      Example images 2 |        Example images 3 |
| ---         |     ---      |          --- |           --- |
| `--blur `| <img src="./demo_image/img_3_blur.jpg" width="200" height="32" >    |   <img src="./demo_image/img_3_blur2.jpg" width="200" height="32">    |  <img src="./demo_image/img_3_blur46.jpg" width="200" height="32">   |
| `--prydown`| <img src="./demo_image/mi1.jpg" width="200" height="32">      |    <img src="./demo_image/mi2.jpg" width="200" height="32">    |   <img src="./demo_image/mi3.jpg" width="200" height="32">  
| `--lr_motion`| <img src="./demo_image/lf1.jpg" width="200" height="32">  |   <img src="./demo_image/lf2.jpg" width="200" height="32">   |  <img src="./demo_image/lf4.jpg" width="200" height="32"> 
| `--ud_motion`| <img src="./demo_image/img_3_up2.jpg" width="200" height="32">      |    <img src="./demo_image/img_3_up5.jpg" width="200" height="32">    |   <img src="./demo_image/img_3_up22.jpg" width="200" height="32">    |
|`--random_offset` | <img src="./demo_image/rd1.jpg" width="50" height="40">    |   <img src="./demo_image/rd2.jpg" width="60" height="45">    |  <img src="./demo_image/rd3.jpg" width="160" height="60">  
|`noise_enable` | <img src="./demo_image/n1.jpg" width="200" height="32">    |   <img src="./demo_image/n2.jpg" width="200" height="32">    |  <img src="./demo_image/n3.jpg" width="200" height="32">  



## Random spaces about generating text
Because the text in the real scene is likely to have a certain gap, if the distance of the text you generate is fixed,
it is likely that the text in the trained model cannot be recognized with too large a gap. This code solves this very well.
This code solves this problem by adding random spaces to the code. The specific effects are as follows:<br>
|   Example images 2    |      Example images 2 |        Example images 3 |
| ---      |          --- |           --- |
| <img src="./demo_image/img_3_space15.jpg" width="250">    |   <img src="./demo_image/img_3_space57.jpg" width="250">    |  <img src="./demo_image/img_3_space79.jpg" width="250">   |


Of course, if you know in advance what kind of font color you need, you need to generate this type of images, 
you can also use `--customize_color` parameters to customize the font color.


## About the choice of font color
In this paper, the kmeans clustering method is used to calculate the 8 cluster centers in the LAB space 
based on the background image clipped by the selected text, and then load the color library (including 9882 colors), 
and randomly select No. 500 from the color library. For colors, calculate the sum of the standard deviations of each color
 number and the cluster centers of 8 types, and randomly select one of the first 100 colors as the font color of the generated image. 
 (This can enrich the font color, of course, you can also choose the maximum standard deviation color as the font color)

## About the choice of font and corpus
When your corpus dictionary is very large, the font you choose may not support some of the words in your dictionary,
but if you don’t filter, the generated image is likely to be garbled or blank. This code selects the text and At the same time,
the font is selected to ensure that the currently selected text is in the font.


## About horizontal and vertical text
If you need some vertical text in your scene, you can modify the parameters in the code to change the direction of the generated text.

## Arguments
* `--num_img`: Number of images to generate.
* `--font_min_size`: Can help adjust the size of the generated text and the size of the picture.
* `--font_max_size`: Can help adjust the size of the generated text and the size of the picture.
* `--bg_path`: The generated text pictures will use the pictures of this folder as the background.
* `--fonts_path`: he font used to generate the picture.
* `--corpus_path`: The corpus used to generate the text picture.
* `--color_path`: Color font library used to generate text.
* `--chars_file`: Chars allowed to be appear in generated images.
* `--customize_color`: Support font custom color.
* `--blur`: Apply gauss blur to the generated image.
* `--prydown`: Blurred image, simulating the effect of enlargement of small pictures.
* `--lr_motion`: Apply left and right motion blur.
* `--ud_motion`: Apply up and down motion blur.
* `--random_offset`: Randomly add offset.


# About font files
I sorted out about 700 fonts that can be used in generating OCR text pictures,
Downloaded Baidu Cloud Link as follows:<br>
 [here1](https://pan.baidu.com/s/1dRp70rEVeauu9rWB7bfGZw). Extraction code:8kzt.   [here2](https://pan.baidu.com/s/1aHkYSxactHm4u5eEiqHDAA). Extraction code:s58p

![example1.jpg](./demo_image/ziti.png)


# Tools
You can use `sentence_filter.py` script to select different modes(contains `filter` and `split` model) to 
filter the text and remove the text that is not in the dictionary and to cut the text of different lengths.


# Reference
- https://github.com/Sanster/text_renderer
- https://github.com/wang-tf/Chinese_OCR_synthetic_data


# Contact
Feel free to contact me if there is any question: <br>
My mail address is jhsignal@126.com; 
