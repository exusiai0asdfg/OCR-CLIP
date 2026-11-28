# 项目时间线
1. 11.23 晚
   - 收到 RHOS Stage 0 考核任务，建立项目目录并初始化 Git 仓库。
   - 下载 OCL 论文，克隆官方仓库。
2. 11.24 中午 
   - 阅读 OCL 论文，了解数据集结构、任务定义及基准实验。
   - 配置 Python 环境，读取 pickle 标注和 JSON 词表，统计实例数最多的十个物体类别，生成目标图片清单。
   - 尝试下载 COCO 图片，向师兄请教如何获取指定的 ImageNet 图片。
3. 11.25 晚上
    - 在 Codex 协助下，通过 Hugging Face、Parquet 元数据和 HTTP Range 请求获取全部 309 张 ImageNet 图片。
   - 发现 ImageNet 标注框与原图坐标不匹配，通过定位标注与 IoU 对比验证 256×256 坐标转换方案。
   - 尝试原始 URL、官方归档、Wayback Machine 和 Common Crawl 等方式恢复 Web 图片，并人工找回 31 张图片。
     - 一则趣闻:其中一张来自宜宾新闻的[船的照片]((http://www.ybxww.com/content/2012-10/8/xinhua_sc_yibin20121081011208807032.jpg))直接访问会跳转到宜宾新闻首页,但是经过查询发现有一个`past.ybxww.com`专门存放老新闻,于是将www改为past成功获得该图片
   - 最终获取 401/483 张 Web 图片,仍缺失 82 张
4. 11.28 早上+中午+下午
    - 完成dataset和dataloader
    - 完成top-k,AP和mAP
    - 中途发现部分可供性标签共用相同的提示词,猜测可以改一改
    - 设计三组实验验证猜想,同时设计了一个成对语义消歧指标,验证模型区分同词异义标签的能力
    - 结束jupyter notebook研究部分
    - 封装出`main.py`
    - 使用 Codex 审查并修复缓存失效、数据与词表校验、路径依赖及依赖版本等问题
    - 整理 README,实验报告,设计记录和时间线,准备最终提交


