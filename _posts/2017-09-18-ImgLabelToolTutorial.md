---
layout: post
title: Polygon-RNN Image Label Tool HandBook
description: "基于Polygon-RNN的标注工具的使用手册"
date: 2017-09-18
tags: []
comments: true
share: true
---
## 使用方法

首先明确几个概念。 我们的目的是标注图中的object， 需要给出object的polygon和label。

* 选择待标注图像文件夹 【O】
* 人工模式生成polygon
    * 生成polygon的点 【左击】
    * 删除上个生成的点 【Backspace】
    * 闭合polygon 【左击第一个点】
* 人工模式修改闭合的polygon
    * 移动一个点 【按住鼠标左键并拖动该点】
    * 添加一个点 【在边缘上左击】
    * 删除一个点 【Shift + 左击该点】
* polygon-RNN模式生成并修改polygon
    * 进入Polygon-RNN模式 【P】
    * 给定目标周围的bbox 【按住鼠标左键并拖动】
    * 移动一个点 【按住鼠标左键并拖动该点】
    * 添加一个点 【在边缘上左击】
    * 删除一个点 【Shift + 左击该点】
* 给当前polygon添加label以生成object 【N】
* 删除object
    * 选择object的polygon 【Ctrl + 左击】
    * 删除object 【D】
* 修改object的label
    * 选择object的polygon 【Ctrl + 左击】
    * 修改label 【M / L】
* 取消选择object的polygon 【Q / Esc】
* 选择多个object 【Ctrl + 左击】
* 放大
    * 进入放大模式 【Z / 按住鼠标右键】
    * 调整放大系数 【鼠标滚轮】
    * 调整放大范围 【Shift + 鼠标滚轮】
* 两个polygon合并/取交
    * 合并 【Ctrl + Alt + 左击object的polygon】
    * 取交 【Ctrl + Shift + 左击object的polygon】
* 改变object顺序
    * 上移一层 【Up】
    * 下移一层 【Down】
* 改变mask不透明度
    * 增加不透明度 【+】
    * 减少不透明度 【-】

## 你必须要知道的Tips

1. polygon-RNN模式下修改polygon必须从离第一个点最近的点开始修改 <br> 因为polygon-RNN模式下点是序列生成的，如果先改后面的点然后再改前面的点，那么后面的点的修改是不会被保存的。 注意生成的polygon都是逆时针的。

1. 遇到部分遮挡怎么办？ <br> 比如后面的车被前面车挡住了一部分。 在polygon-RNN模式下，选取遮挡部分第一个点，拖动到遮挡部分最后一个点附近。

1. 遇到截断遮挡怎么办？ <br> 比如一辆自行车被骑车的人遮挡成了两部分。 在polygon-RNN模式下， 一般会生成一个部分的polygon，拖动离另一部分最近的一个点到另一部分附近就可以将另一部分包括进来了。

1. polygon-RNN模式下重新选择bbox  <br> 当前的bbox生成的polygon很不好，按两下【P】就可以重新选择bbox啦。 第一下【P】是退出polygon-RNN模式，第二下【P】是再次进入polygon-RNN模式。 可以感受一下选择不同大小的bbox对Polygon-RNN的影响哦。

1. 快速选择label <br> 在选择label对话框中按类别首字母。 上次选择的label会是下次选择的默认值，所以连续标同一类object就不用选择label了哦。

## 你可能需要的Tips

1. 改变object顺序。 <br> 正确的标注过程是从后往前标注，即先标被遮挡的object，再标遮挡别人的object。 如果不小心先标了前面的object，那么可以继续标被遮挡的object，然后按【Down】键下移一层。 注意只能在小范围使用这个功能，因为要是前移后移很多次，你也会搞不清到底这个object现在在哪一层。所以还是要按照从后往前的顺序标注哦。注意这样的话json文件中的object的位置会前移，但object的id不会改变，所以使用json文件的时候要按object的顺序来确定遮挡顺序，而不是根据object的id大小来确定。

1. 为什么较大的object在polygon-RNN模式下不能精调点的位置？ <br> 因为polygon-RNN的点是在28×28的分辨率下生成的， 所以object较大的情况下拖动点，会感觉点在“跳跃”。想要精修的话退出polygon-RNN模式，在人工模式下精修点的位置。
