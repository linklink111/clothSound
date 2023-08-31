# Cloth Sound
Granular based cloth sound simulation.
## 2023年8月26日
今天改了clothSound的代码，解决了越往后合成的音频段越长的问题（step中t_curr 每次都多加了很多偏移t）。

目前基本实现了音轨编码速度比之前快很多，声音较为相似（在没有用滤波器的情况下），但是布料掉在球体上之后继续下落的过程似乎有很多的噪声，音频整体上也有很多噪声，以及较难设定一个目标长度时间来生成音频。

目前质点振动用的还是正弦波，计划改成点声源的波动方程（Howe等人）。

现在的声音辐射是根据每帧的加速度（由于m=1，等同于force）辐射一个频率固定的正弦波，或许可以根据不同的force辐射不同频率的波，例如Spring force，damping等各辐射出不同频率的波

如果写论文的话，可以写的创新点包括：
1. merge_t
2. 基于粒子的
3. 快速的（接近实时的）
4. 可以捕获关键的碰撞状态的
5. 分段编码时对于声音传播时间大于dt问题的处理

## 2023年8月28日
今天改了声音大小不统一的问题，在声压公式下方除以了一些与空气有关的系数，让声压数值更合理，也更便于归一化。
接下来要解决的是声音分段拼接之后不连续的问题。

## 2023年8月29日

尝试解决声音分段不连续的问题，打印dt和r/c发现r/c比dt大很多，得出结论：每个分段内声音逐渐衰减的原因是滞后的声音被单独编码导致的。
尝试采用重叠编码的方式，不可行。

现在分析miPhysics的音频处理部分。
初步分析，感觉要用多线程。

进一步了解了miPhysics，实在是......太强了。光从他们官网上的文档就能学到很多东西。（但是不知道为什么他们的论文之前都没看到过（可能是因为他们比较偏工程？））

在zlib下载了一些书

## 2023年8月30日

感觉今天还是要注意解决一下声音不连续的问题

先实现个多线程？
等等，既然能多线程了，有没有可能用GPU处理音频也是可行的。（本质上就是数组排序和插值吧？）

突然想起来昨天在miPhysics文档上看到的一句话很有启发性：In waveguide methods, we decompose physical motion into travelling waves in opposite directions, linked together by scattering junctions
这似乎和我模拟布料声音的思路有一点相似，不知道travelling waves和它的link是怎么实现的。

在考虑了后续可能还要实现复杂的控制功能之后，我又向blender求助了（救命）
在前面的研究中我已经验证了将单个质点作为发声单元的可行性（miPhysics的质点声学系统的叫法也印证了这一点（由于我不知道它具体是什么，所以只能通过名称来考量一下hh））
由于质点的独立性，不像之前的纱线一样难以建模，为复杂动画或结构的模拟做了准备。

拆解出一个关键的小问题：通过布料的point cache估计每个时刻质点的加速度（1.直接算斜率；2.神经网络？）

感觉这周还是先出点效果比较好

为了导出blender中的模拟数据稍微研究了一下point cache。看了一圈觉得这篇比较有用https://code.blender.org/2011/01/a-look-at-point-cache/（这里记下来防止以后要用）
原来mdd文件是light weight point cache的意思，之前用过但是不知道。

写了一个类似于数据处理的工作流：（因为菜所以不能实时处理数据只能像数据分析一样处理orz）
1. 读取mdd
2. 差分出速度和加速度
3. 应用诺伊曼边界条件把加速度的变化率转为压强
4. 根据t排序，插值p，写为wav

 长远计划：
 1. 模拟布料的声音（如果有优化的话或许可以多发几篇文章）(差分法求加速度发一篇，神经网络求加速度再发一篇，或许还可以用神经网络把加速度转换成声压？先用神经网络学习对布料动画的分类（摩擦、自碰撞、刚体碰撞...一切从分类开始），分类发一篇，生成发一篇......)
 2. 模拟软体的声音（软体声音目前也只有数据驱动的方法。与布料类似，如果有优化的话或许可以多发几篇文章）
 3. 模拟地面的声音
 4. 毕业论文写非线性物体的声音模拟，把之前做的都写进来。

 （长远地想，怎样用神经网络把加速度转换为声压呢？现有的文献中有刚体和非线性薄壳的数据，而目标布料的数据虽然数据集中不存在，但却是这些数据的极端情况，擅长求导的神经网络或许可以求出来？）


咳咳，回到当下
感觉这周五之前好像做不出声音效果了？要不明天先做动画吧？
今天写的挺多的，可能是因为没出什么bug吧...(bug之前改的差不多了，改bug确实是没有什么可以写的了，但是bug确实是越改越少的。)

## 2023年8月31日

今天在写的时候发现求出的速度、加速度、声压（？）似乎可以可视化在2D图像上或作为布料贴图。

可恶，好像数值太大了算出来都是nan和inf

果然chatGPT一时爽，改bug火葬场啊，今天一直在改昨天chatGPT写的bug，当然也是用chatGPT在改。

终于改好bug了，现在对x，v，a都进行了可视化分析（作了线图、热图），发现a的线图局部有点像脉冲信号，整体像是波形。
发现x，v，a的数据在开始和末尾都存在奇异点，需要对边界稍加处理，也可以考虑数据清洗（去除过大的数据（离群点（？）））。

label 1：
我现在要把加速度编码为音轨了
但是我想临时改变一下策略
就是先把每个粒子自己发出的声音（一段很长的波形）编码出来，然后再加起来————这里的代码卸载massSound里。


突然发现dt和编码的时间是一对不可忽略的变量，之前对于时间的处理方式是根据编码出来的长度scale时间轴，有点盲目，现在来梳理一下dt和编码时间的关系：
在采样率为44100的情况下，
dt*frame_num = duration
编码的时候插值范围是 （0,duration）,增长步是 duration/sample_rate (现在似乎理清了，之前以为是1/duration*sample_rate)

blender模拟的时候设置了一个dt，但是由于缓存到帧上之后这个dt表现不出来，所以这个dt没有太大意义（仅供参考？）
真正要关心的dt与帧数和帧率有关，blender渲染出100帧数据，帧率是24fps，则dt是1/24（dt是帧与帧之间的时间间隔）

原来如此...


写完代码了，label 1的思路写成代码看起来还挺逻辑自洽的，而且这样数据量少不会爆内存，每次把audio直接加到all_audio里看起来也很可行...
希望运行之后能出好结果...

这里运行前先做一些事前怀疑，以便出问题之后考虑修改哪些地方：
1. 首先如果声音完全不能听...等等，是不是还没有把声音归一化？不过单个质点的声音要除以多少呢？等出结果了再慢慢调？
2. 呃总之如果声音完全不能听的话，最有可能的问题就是归一化的问题了；
3. 这回应该不会出现声音断断续续的问题了；
4. 时间的问题，这回的t应该没有加错；
5. 其他可能不合理的地方，或许还应该再分析一下dt，duration等变量。
6. （done）突然想起之前说要清洗数据来着，比如加速度在最开始的时候有一个很大的值，不太对。
7. 运行之前看了最后一眼，发现线性插值可能不可行，试分析，对坐标进行线性插值，导致它的差分（速度）是定值，进而导致加速度是0
8. 但是有没有一种可能，由于声音比较随机，所以7中的问题可以忽略，甚至，有没有一种可能，根本不用插值。
9. 如果7-8的问题导致音频采样率不足的话，可以尝试：a.非线性插值；b. 尝试10
10. 要不要把这种编码方式给之前的taichi程序用用试试？

突然想起来数据驱动那篇文章布料和沙发的摩擦是不是已经算是两种不同的布料耦合发声了？突然难了起来。
所以说似乎只能shakeCloth，dropCloth和blowCloth了...
等等，如果沙发也用布料做并且放到MD模拟再导出point Cache到blender再转为mdd再导出position的话或许化石可以一试的... (为什么导出步骤这么长，好吧虽然麻烦但是可行且简单)（不过说实话我对摩擦不是很有信心，但是理论分析的话，摩擦的时候会有负的加速度，可能也会产生一些噪声，但加速度相同的情况下，直觉上看空气阻力和摩擦阻力总得有区别吧，还是说空气阻力根本产生不了这么大的负向加速度？ ）