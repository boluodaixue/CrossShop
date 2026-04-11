"""Build a versioned bilingual CategoryCard corpus from real and synthetic facts.

The v3 corpus preserves all accepted ESCI-derived v1 facts. Model-authored
category specifications drive deterministic synthetic product instances for
coverage only. Every generated field is explicitly marked and must never be
described as an observed marketplace fact.
"""

# ruff: noqa: E501 -- bilingual specification literals are kept visually atomic.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from globex_agent.category_insight import admit_card

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_TIME = "2026-08-15T00:00:00+08:00"
GENERATION_SEED = 20260815
GENERATOR_VERSION = "category-card-bilingual-synthetic-v3"
FACT_DATASET_VERSION = "category-item-facts-bilingual-v3"
CARD_DATASET_VERSION = "category-cards-bilingual-v3"
SYNTHETIC_SOURCE_KIND = "synthetic_llm_template"


def _pairs(raw: str) -> list[tuple[str, str]]:
    return [tuple(token.split("|", 1)) for token in raw.split(";")]


def _facets(
    *rows: tuple[str, str, str],
) -> list[dict[str, Any]]:
    return [
        {"name": name, "name_zh": name_zh, "values": _pairs(values)}
        for name, name_zh, values in rows
    ]


def _spec(
    category: str,
    category_zh: str,
    forms: str,
    bounds: tuple[int, int],
    facets: list[dict[str, Any]],
    aliases: str = "",
) -> dict[str, Any]:
    if len(facets) != 7:
        raise ValueError(f"{category} must define exactly seven facets")
    return {
        "category": category,
        "category_zh": category_zh,
        "aliases": [alias for alias in aliases.split(";") if alias],
        "forms": _pairs(forms),
        "price_bounds_cny": list(bounds),
        "facets": facets,
    }


CATEGORY_SPECS = [
    _spec(
        "home ventilation fans",
        "家用通风扇",
        "bathroom exhaust fan|浴室排风扇;ceiling ventilation fan|吸顶通风扇;heat-recovery ventilator|热回收新风机;wall exhaust fan|壁挂排风扇",
        (250, 3500),
        _facets(
            ("Product form", "产品形态", "exhaust fan|排风扇;ventilation fan|通风扇;heat recovery|热回收新风机"),
            ("Mounting", "安装方式", "ceiling|吸顶式;wall|壁挂式;window|窗式"),
            ("Control feature", "控制功能", "without light|不带灯;with light|带照明;humidity sensor|湿度感应"),
            ("Airflow", "风量", "low airflow|小风量;medium airflow|中等风量;high airflow|大风量"),
            ("Noise level", "噪声水平", "quiet|静音;standard|标准噪声;high power|高功率"),
            ("Room setting", "适用空间", "bathroom|浴室;kitchen|厨房;whole house|全屋"),
            ("Energy profile", "能效", "standard|标准能效;energy saving|节能;heat recovery|热回收"),
        ),
        "bathroom fan;浴室排风扇;卫生间换气扇",
    ),
    _spec(
        "utility tires and wheels",
        "工具车轮胎与轮组",
        "pneumatic tire on wheel|充气轮胎轮组;flat-free replacement wheel|免充气替换轮;solid utility wheel|实心工具轮;lawn mower tire|割草机轮胎",
        (60, 900),
        _facets(
            ("Construction", "轮胎结构", "flat-free|免充气;pneumatic|充气式;solid|实心"),
            ("Package form", "包装形式", "tire on wheel|轮胎轮毂总成;tire only|单轮胎;wheel only|单轮毂"),
            ("Bearing size", "轴承尺寸", "0.5 inch|二分之一英寸;0.625 inch|八分之五英寸;0.75 inch|四分之三英寸"),
            ("Diameter", "轮径", "6 inch|六英寸;10 inch|十英寸;15 inch|十五英寸"),
            ("Tread", "胎纹", "ribbed|直纹;knobby|块状胎纹;smooth|光面"),
            ("Load profile", "承载能力", "light duty|轻载;medium duty|中载;heavy duty|重载"),
            ("Use setting", "适用设备", "garden cart|园艺车;lawn mower|割草机;hand truck|手推车"),
        ),
        "lawn mower tires;工具车轮;割草机轮胎",
    ),
    _spec(
        "lawn mower equipment",
        "割草机设备",
        "manual reel mower|手推滚刀割草机;electric lawn mower|电动割草机;riding mower|骑乘式割草机;mower lift|割草机举升架",
        (500, 18000),
        _facets(
            ("Equipment form", "设备形态", "walk-behind mower|手推式割草机;riding mower|骑乘式割草机;mower lift|割草机举升架"),
            ("Power source", "动力来源", "manual|手动;battery|电池;corded electric|有线电动"),
            ("Grass handling", "草屑处理", "catcher included|带集草袋;mulching|碎草;side discharge|侧排草"),
            ("Cutting width", "切割宽度", "compact|窄幅;standard|标准幅宽;wide|宽幅"),
            ("Yard size", "适用草坪", "small yard|小草坪;medium yard|中等草坪;large yard|大草坪"),
            ("Height adjustment", "高度调节", "fixed|固定高度;three level|三档调节;multi level|多档调节"),
            ("Storage", "收纳方式", "folding handle|折叠手柄;compact|紧凑收纳;standard|普通收纳"),
        ),
        "lawn mower;割草机;草坪机",
    ),
    _spec(
        "outdoor fencing and gates",
        "户外围栏与庭院门",
        "privacy screen|隐私围栏布;metal fence panel|金属围栏片;garden gate|庭院门;decorative border fence|装饰边界围栏",
        (100, 4500),
        _facets(
            ("Material", "材质", "steel|钢制;vinyl|乙烯基;wood|木质"),
            ("Product form", "产品形态", "privacy screen|隐私网;panel|围栏片;gate|庭院门"),
            ("Installation", "安装方式", "no-dig|免挖安装;post mounted|立柱安装;folding|折叠式"),
            ("Privacy level", "隐私程度", "open|开放式;semi-private|半隐私;full privacy|全遮挡"),
            ("Weather resistance", "耐候性", "standard|标准;rust resistant|防锈;waterproof|防水"),
            ("Height", "高度", "low border|低矮边界;medium|中等高度;tall|高围栏"),
            ("Use setting", "使用场景", "garden|花园;backyard|后院;pool boundary|泳池边界"),
        ),
        "privacy fence;庭院围栏;户外栅栏",
    ),
    _spec(
        "seedling and plant trays",
        "育苗盘与植物托盘",
        "seedling tray|育苗盘;propagation tray|繁殖托盘;plant drip tray|植物接水盘;humidity dome tray|育苗保湿罩托盘",
        (20, 350),
        _facets(
            ("Tray form", "托盘形态", "seedling tray|育苗盘;propagation tray|繁殖盘;drip tray|接水盘"),
            ("Drainage", "排水", "without holes|无孔;with holes|带排水孔;removable insert|可拆内盘"),
            ("Material", "材质", "plastic|塑料;rubber|橡胶;metal|金属"),
            ("Cell count", "穴孔数量", "few cells|少穴;medium cells|中等穴数;many cells|多穴"),
            ("Cover", "上盖", "open tray|无盖;clear dome|透明罩;vented dome|可调通风罩"),
            ("Reuse profile", "重复使用", "disposable|一次性;reusable|可重复使用;heavy duty|加厚耐用"),
            ("Use setting", "使用场景", "indoor shelf|室内育苗架;greenhouse|温室;outdoor garden|户外花园"),
        ),
        "seedling trays;育苗托盘;植物接水盘",
    ),
    _spec(
        "label makers and printers",
        "标签机与标签打印机",
        "portable label maker|便携标签机;direct thermal label printer|热敏标签打印机;handheld labeler|手持标签机;desktop shipping printer|桌面面单打印机",
        (80, 1800),
        _facets(
            ("Device form", "设备形态", "portable label maker|便携标签机;thermal printer|热敏打印机;handheld labeler|手持标签机"),
            ("Input method", "输入方式", "QWERTY keyboard|全键盘;app control|手机应用;computer connected|电脑连接"),
            ("Primary use", "主要用途", "shipping labels|快递面单;office organization|办公整理;cable labels|线缆标签"),
            ("Print technology", "打印技术", "direct thermal|热敏;thermal transfer|热转印;embossing|压纹"),
            ("Connectivity", "连接方式", "standalone|独立使用;USB|USB连接;Bluetooth|蓝牙"),
            ("Label width", "标签宽度", "narrow|窄标签;standard|标准宽度;wide shipping|宽面单"),
            ("Power profile", "供电方式", "replaceable battery|干电池;rechargeable|充电式;AC powered|电源适配器"),
        ),
        "label printer;标签打印机;便携标签机",
    ),
    _spec(
        "extension cords and power strips",
        "延长线与插线板",
        "outdoor extension cord|户外延长线;travel power strip|旅行插线板;12V extension cable|车载12V延长线;desktop charging station|桌面充电站",
        (40, 1000),
        _facets(
            ("Product form", "产品形态", "outdoor extension cord|户外延长线;power strip|插线板;12V cable|车载12V线"),
            ("Wire gauge", "线规", "12 AWG|12号线规;14 AWG|14号线规;16 AWG|16号线规"),
            ("Outlet style", "插口形式", "three-prong|三孔插座;USB outlets|USB接口;single socket|单插口"),
            ("Cord length", "线长", "short|短线;medium|中等长度;long|长线"),
            ("Environment", "使用环境", "indoor|室内;outdoor|户外;travel|旅行"),
            ("Safety", "安全功能", "basic|基础保护;surge protection|浪涌保护;overload protection|过载保护"),
            ("Plug profile", "插头形态", "straight plug|直插;flat plug|扁平插头;locking plug|锁定插头"),
        ),
        "extension cord;插线板;户外延长线",
    ),
    _spec(
        "garden warning signs",
        "花园警示牌",
        "plastic warning sign|塑料警示牌;metal lawn sign|金属草坪牌;decorative garden sign|装饰花园牌;yard stake sign|插地警示牌",
        (20, 300),
        _facets(
            ("Material", "材质", "metal|金属;plastic|塑料;aluminum|铝制"),
            ("Message type", "标语类型", "plant protection|植物保护;dog warning|宠物警示;yard warning|庭院提醒"),
            ("Mounting", "安装方式", "yard stake|插地式;hanging|悬挂式;wall mounted|墙面安装"),
            ("Weather resistance", "耐候性", "standard|标准;waterproof|防水;UV resistant|抗紫外线"),
            ("Visibility", "醒目程度", "subtle|低调;high contrast|高对比;reflective|反光"),
            ("Tone", "语气", "polite|礼貌;direct|直接;humorous|幽默"),
            ("Use setting", "使用场景", "front yard|前院;garden bed|花坛;lawn|草坪"),
        ),
        "garden sign;花园警示牌;草坪提示牌",
    ),
    _spec(
        "graphic slogan t-shirts",
        "图案标语T恤",
        "sarcastic slogan t-shirt|讽刺标语T恤;novelty graphic tee|趣味图案T恤;quote print shirt|文字印花T恤;minimal slogan tee|极简标语T恤",
        (60, 500),
        _facets(
            ("Audience", "适用人群", "men|男士;women|女士;unisex|中性"),
            ("Color", "颜色", "black|黑色;white|白色;navy|藏青色"),
            ("Theme", "主题", "sarcastic|讽刺;funny|幽默;inspirational|励志"),
            ("Fit", "版型", "slim|修身;regular|常规;oversized|宽松"),
            ("Fabric weight", "面料厚度", "lightweight|轻薄;midweight|中等;heavyweight|厚实"),
            ("Print style", "印花风格", "minimal text|极简文字;large graphic|大幅图案;vintage|复古"),
            ("Occasion", "穿着场景", "casual|日常;gift|礼物;party|聚会"),
        ),
        "funny t-shirt;标语T恤;趣味印花T恤",
    ),
    _spec(
        "mens zipper wallets",
        "男士拉链钱包",
        "zip-around bifold wallet|拉链对折钱包;RFID card wallet|防盗刷卡包;large-capacity leather wallet|大容量皮钱包;double-zip wallet|双拉链钱包",
        (80, 800),
        _facets(
            ("Material", "材质", "genuine leather|真皮;synthetic leather|合成革;fabric|织物"),
            ("Security feature", "安全功能", "RFID blocking|RFID防盗刷;standard|普通;hidden pocket|隐藏夹层"),
            ("Closure", "闭合方式", "zip-around|全包拉链;double zipper|双拉链;bifold|对折"),
            ("Capacity", "容量", "slim|轻薄;standard|标准;large capacity|大容量"),
            ("Coin storage", "零钱收纳", "no coin pocket|无零钱袋;coin pocket|零钱袋;zip coin pocket|拉链零钱袋"),
            ("Card slots", "卡位", "few slots|少卡位;standard slots|标准卡位;many slots|多卡位"),
            ("Style", "风格", "classic|经典;business|商务;casual|休闲"),
        ),
        "zipper wallet;男士拉链钱包;防盗刷钱包",
    ),
    _spec(
        "pedal go-karts",
        "儿童脚踏卡丁车",
        "kids pedal go-kart|儿童脚踏卡丁车;adjustable-seat ride-on kart|可调座椅脚踏车;four-wheel pedal car|四轮脚踏车;compact toddler kart|幼儿紧凑卡丁车",
        (400, 6000),
        _facets(
            ("Drive type", "驱动方式", "pedal powered|脚踏;chain drive|链条传动;direct drive|直驱"),
            ("Audience", "适用人群", "young children|幼儿;children|儿童;older children|大龄儿童"),
            ("Seat", "座椅", "fixed|固定座椅;adjustable|可调座椅;bucket seat|包裹座椅"),
            ("Brake", "刹车", "hand brake|手刹;coaster brake|倒刹;dual brake|双刹"),
            ("Tire", "轮胎", "plastic|塑料轮;rubber|橡胶轮;pneumatic|充气轮"),
            ("Frame", "车架", "light steel|轻型钢架;reinforced steel|加强钢架;compact frame|紧凑车架"),
            ("Terrain", "适用路面", "indoor|室内;sidewalk|人行道;yard|庭院"),
        ),
        "pedal go kart;儿童卡丁车;脚踏车",
    ),
    _spec(
        "dirt bike tools and accessories",
        "越野摩托工具与配件",
        "metric tool set|公制工具套装;motorcycle tool pack|摩托车工具包;garage pit mat|维修区地垫;trackside repair kit|赛道维修工具包",
        (80, 1800),
        _facets(
            ("Accessory form", "配件形态", "tool set|工具套装;tool pack|工具包;pit mat|维修地垫"),
            ("Tool profile", "工具规格", "metric|公制;T-handle|T形手柄;portable|便携"),
            ("Use setting", "使用场景", "track and pit|赛道维修区;garage|车库;trail|林道"),
            ("Repair task", "维修任务", "tire repair|轮胎维修;chain maintenance|链条维护;general service|通用保养"),
            ("Storage", "收纳方式", "roll pouch|卷式工具袋;zip bag|拉链包;hard case|硬壳箱"),
            ("Tool count", "工具数量", "compact kit|精简套装;standard kit|标准套装;large kit|大型套装"),
            ("Compatibility", "兼容性", "dirt bike|越野摩托;motocross|摩托越野赛;multi-motorcycle|多种摩托"),
        ),
        "dirt bike tools;越野摩托工具;摩托维修工具包",
    ),
    _spec(
        "wireless earbuds",
        "真无线蓝牙耳机",
        "true wireless earbuds|真无线耳机;noise-cancelling earbuds|降噪耳机;open-ear earbuds|开放式耳机;sports earbuds|运动耳机",
        (100, 2200),
        _facets(
            ("Wearing style", "佩戴方式", "in-ear|入耳式;semi-in-ear|半入耳式;open-ear|开放式"),
            ("Noise control", "降噪", "active noise cancellation|主动降噪;passive isolation|被动隔音;ambient mode|通透模式"),
            ("Battery life", "续航", "short|短续航;all-day|全天续航;extended|长续航"),
            ("Microphone", "麦克风", "single mic|单麦克风;dual mic|双麦克风;AI call noise reduction|AI通话降噪"),
            ("Water resistance", "防水", "daily splash|生活防泼水;IPX5|IPX5;IPX7|IPX7"),
            ("Charging", "充电方式", "USB-C|USB-C充电;wireless charging|无线充电;fast charging|快充"),
            ("Use setting", "使用场景", "commuting|通勤;sports|运动;gaming|游戏"),
        ),
        "bluetooth earbuds;蓝牙耳机;主动降噪耳机",
    ),
    _spec(
        "portable bluetooth speakers",
        "便携蓝牙音箱",
        "compact bluetooth speaker|迷你蓝牙音箱;waterproof outdoor speaker|防水户外音箱;party speaker|派对音箱;clip-on speaker|挂扣音箱",
        (100, 2500),
        _facets(
            ("Size", "尺寸", "pocket|口袋尺寸;compact|紧凑;large portable|大型便携"),
            ("Sound profile", "声音风格", "balanced|均衡;bass boosted|重低音;voice focused|人声突出"),
            ("Water resistance", "防水", "indoor only|仅室内;IPX5|IPX5;IPX7|IPX7"),
            ("Battery life", "续航", "short|短续航;all-day|全天续航;extended|长续航"),
            ("Connectivity", "连接", "Bluetooth only|仅蓝牙;Bluetooth and AUX|蓝牙和AUX;multi-speaker pairing|多音箱串联"),
            ("Mounting", "携带方式", "tabletop|桌面;clip-on|挂扣;carry handle|提手"),
            ("Use setting", "使用场景", "home|居家;outdoor|户外;party|派对"),
        ),
        "bluetooth speaker;蓝牙音箱;户外音响",
    ),
    _spec(
        "mechanical keyboards",
        "机械键盘",
        "compact mechanical keyboard|紧凑机械键盘;full-size mechanical keyboard|全尺寸机械键盘;low-profile keyboard|矮轴键盘;gaming keyboard|游戏机械键盘",
        (150, 2500),
        _facets(
            ("Layout", "配列", "60 percent|60配列;75 percent|75配列;full size|全尺寸"),
            ("Switch", "轴体", "linear|线性轴;tactile|段落轴;clicky|有声段落轴"),
            ("Connectivity", "连接方式", "wired|有线;Bluetooth|蓝牙;tri-mode|三模"),
            ("Hot swap", "热插拔", "fixed switch|不可热插拔;three-pin hot swap|三脚热插拔;five-pin hot swap|五脚热插拔"),
            ("Backlight", "背光", "none|无背光;white|白光;RGB|RGB背光"),
            ("Keycap", "键帽", "ABS|ABS键帽;PBT|PBT键帽;low profile|矮键帽"),
            ("Use setting", "使用场景", "office|办公;gaming|游戏;portable|便携"),
        ),
        "mechanical keyboard;机械键盘;游戏键盘",
    ),
    _spec(
        "wireless computer mice",
        "无线鼠标",
        "compact wireless mouse|便携无线鼠标;ergonomic mouse|人体工学鼠标;gaming mouse|无线游戏鼠标;silent office mouse|静音办公鼠标",
        (60, 1800),
        _facets(
            ("Shape", "外形", "compact|紧凑;ergonomic|人体工学;ambidextrous|左右手通用"),
            ("Connectivity", "连接方式", "2.4GHz|2.4G接收器;Bluetooth|蓝牙;dual-mode|双模"),
            ("Sensor", "传感器", "standard optical|标准光学;high DPI|高DPI;gaming sensor|游戏传感器"),
            ("Click sound", "按键声音", "standard|普通;quiet|低噪;silent|静音"),
            ("Power", "供电", "replaceable battery|干电池;rechargeable|充电式;USB-C fast charge|USB-C快充"),
            ("Buttons", "按键", "basic|基础按键;side buttons|侧键;programmable|可编程"),
            ("Use setting", "使用场景", "office|办公;travel|旅行;gaming|游戏"),
        ),
        "wireless mouse;无线鼠标;静音鼠标",
    ),
    _spec(
        "USB-C hubs and docking stations",
        "USB-C扩展坞",
        "compact USB-C hub|便携USB-C扩展坞;multiport adapter|多接口转换器;desktop docking station|桌面拓展坞;dual-monitor dock|双屏扩展坞",
        (120, 3000),
        _facets(
            ("Port count", "接口数量", "compact ports|精简接口;standard ports|标准接口;many ports|多接口"),
            ("Display output", "视频输出", "HDMI|HDMI;dual HDMI|双HDMI;HDMI and DisplayPort|HDMI和DP"),
            ("Power delivery", "PD供电", "no pass-through|无PD;65W PD|65W PD;100W PD|100W PD"),
            ("Network", "网络接口", "none|无网口;gigabit Ethernet|千兆网口;2.5G Ethernet|2.5G网口"),
            ("Card reader", "读卡器", "none|无读卡器;SD|SD卡;SD and microSD|SD和TF卡"),
            ("Host compatibility", "兼容设备", "laptop|笔记本;tablet|平板;multi-device|多设备"),
            ("Use setting", "使用场景", "travel|旅行;office|办公;creator desk|创作桌面"),
        ),
        "USB-C hub;扩展坞;Type-C拓展坞",
    ),
    _spec(
        "power banks",
        "充电宝",
        "compact power bank|便携充电宝;high-capacity power bank|大容量充电宝;laptop power bank|笔记本充电宝;magnetic power bank|磁吸充电宝",
        (80, 1800),
        _facets(
            ("Capacity", "容量", "5000mAh|5000毫安时;10000mAh|10000毫安时;20000mAh|20000毫安时"),
            ("Output power", "输出功率", "20W|20瓦;45W|45瓦;65W|65瓦"),
            ("Ports", "接口", "single USB-C|单USB-C;USB-C and USB-A|C口和A口;dual USB-C|双USB-C"),
            ("Fast charge", "快充", "basic charging|普通充电;USB PD|PD快充;multi-protocol|多协议快充"),
            ("Form", "形态", "slim|轻薄;compact|紧凑;large capacity|大容量"),
            ("Display", "电量显示", "LED dots|LED灯;percentage display|数字百分比;none|无显示"),
            ("Use setting", "使用场景", "phone travel|手机出行;laptop work|笔记本办公;daily carry|日常携带"),
        ),
        "power bank;充电宝;移动电源",
    ),
    _spec(
        "webcams",
        "网络摄像头",
        "1080p webcam|1080P摄像头;4K webcam|4K摄像头;conference webcam|会议摄像头;streaming webcam|直播摄像头",
        (100, 2500),
        _facets(
            ("Resolution", "分辨率", "720p|720P;1080p|1080P;4K|4K"),
            ("Frame rate", "帧率", "30fps|30帧;60fps|60帧;high frame rate|高帧率"),
            ("Focus", "对焦", "fixed focus|定焦;autofocus|自动对焦;AI tracking|AI跟踪"),
            ("Microphone", "麦克风", "none|无麦克风;dual mic|双麦克风;noise-reducing mic|降噪麦克风"),
            ("Privacy", "隐私保护", "none|无;privacy shutter|隐私盖;electronic privacy|电子隐私模式"),
            ("Field of view", "视野", "narrow|窄视角;standard|标准视角;wide|广角"),
            ("Use setting", "使用场景", "video calls|视频会议;streaming|直播;classroom|网课"),
        ),
        "webcam;网络摄像头;电脑摄像头",
    ),
    _spec(
        "Wi-Fi routers",
        "无线路由器",
        "dual-band Wi-Fi router|双频无线路由器;mesh Wi-Fi system|Mesh路由器;gaming router|游戏路由器;travel router|旅行路由器",
        (150, 4000),
        _facets(
            ("Wi-Fi standard", "无线标准", "Wi-Fi 5|Wi-Fi 5;Wi-Fi 6|Wi-Fi 6;Wi-Fi 6E|Wi-Fi 6E"),
            ("Bands", "频段", "dual-band|双频;tri-band|三频;mesh backhaul|Mesh回程"),
            ("Coverage", "覆盖范围", "small home|小户型;medium home|中等户型;large home|大户型"),
            ("Mesh", "Mesh能力", "standalone|单机;mesh ready|支持Mesh;mesh kit|Mesh套装"),
            ("Ethernet", "有线接口", "fast Ethernet|百兆;gigabit Ethernet|千兆;2.5G Ethernet|2.5G"),
            ("Security", "安全功能", "basic WPA|基础WPA;WPA3|WPA3;parental control|家长控制"),
            ("Use setting", "使用场景", "home|家庭;gaming|游戏;travel|旅行"),
        ),
        "wireless router;无线路由器;WiFi路由器",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-facts",
        type=Path,
        default=PROJECT_ROOT / "data" / "category_insight" / "category_item_facts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "category_insight",
    )
    parser.add_argument("--target-per-category", type=int, default=60)
    parser.add_argument("--minimum-synthetic-per-category", type=int, default=30)
    args = parser.parse_args()
    if args.target_per_category < 10:
        parser.error("--target-per-category must be at least 10")
    if args.minimum_synthetic_per_category < 10:
        parser.error("--minimum-synthetic-per-category must be at least 10")

    real_facts = _read_jsonl(args.real_facts)
    specs = {spec["category"]: spec for spec in CATEGORY_SPECS}
    unknown = {fact["category"] for fact in real_facts} - specs.keys()
    if unknown:
        raise ValueError(f"v3 specs missing real categories: {sorted(unknown)}")

    facts = _build_facts(
        real_facts,
        specs,
        target_per_category=args.target_per_category,
        minimum_synthetic=args.minimum_synthetic_per_category,
    )
    cards, provenance, retrieval_rows = _build_cards(facts, specs)
    audit_queue = _audit_queue(cards)
    taxonomy = _build_taxonomy(specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "facts": args.output_dir / "category_item_facts_v3.jsonl",
        "cards": args.output_dir / "category_cards_v3.jsonl",
        "provenance": args.output_dir / "category_card_provenance_v3.jsonl",
        "retrieval": args.output_dir / "category_retrieval_texts_v3.jsonl",
        "audit": args.output_dir / "audit_queue_v3.jsonl",
        "taxonomy": args.output_dir / "category_taxonomy_v3.json",
        "generation_specs": args.output_dir / "category_generation_specs_v3.json",
        "manifest": args.output_dir / "category_card_manifest_v3.json",
    }
    _write_jsonl(paths["facts"], facts)
    _write_jsonl(paths["cards"], cards)
    _write_jsonl(paths["provenance"], provenance)
    _write_jsonl(paths["retrieval"], retrieval_rows)
    _write_jsonl(paths["audit"], audit_queue)
    paths["taxonomy"].write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["generation_specs"].write_text(
        json.dumps(
            {
                "dataset_version": "category-generation-specs-bilingual-v3",
                "intended_use": "deterministic offline synthetic-data generation",
                "categories": CATEGORY_SPECS,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        facts,
        cards,
        audit_queue,
        args.real_facts,
        {key: path for key, path in paths.items() if key != "manifest"},
        args.target_per_category,
        args.minimum_synthetic_per_category,
    )
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built v3 facts={len(facts)} cards={len(cards)} "
        f"categories={len(specs)} audit={len(audit_queue)}"
    )


def _build_facts(
    real_facts: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    *,
    target_per_category: int,
    minimum_synthetic: int,
) -> list[dict[str, Any]]:
    real_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in real_facts:
        real_by_category[fact["category"]].append(fact)

    output: list[dict[str, Any]] = []
    for category, spec in specs.items():
        for fact in sorted(real_by_category[category], key=lambda row: row["product_id"]):
            output.append(_upgrade_real_fact(fact, spec))
        synthetic_count = max(
            target_per_category - len(real_by_category[category]),
            minimum_synthetic,
        )
        for index in range(synthetic_count):
            output.append(_synthetic_fact(spec, index))
    return sorted(output, key=lambda row: (row["category"], row["product_id"]))


def _upgrade_real_fact(fact: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    value_zh = {
        (facet["name"], value_en): value_zh
        for facet in spec["facets"]
        for value_en, value_zh in facet["values"]
    }
    name_zh = {facet["name"]: facet["name_zh"] for facet in spec["facets"]}
    translated_attributes = []
    for attribute in fact["text_supported_attributes"]:
        translated_attributes.append(
            {
                "name": attribute["name"],
                "name_zh": name_zh.get(attribute["name"], attribute["name"]),
                "value": attribute["value"],
                "value_zh": value_zh.get(
                    (attribute["name"], attribute["value"]), attribute["value"]
                ),
                "evidence_kind": "observed_product_text",
            }
        )
    attribute_phrase = "、".join(
        attribute["value_zh"] for attribute in translated_attributes[:3]
    )
    title_zh = spec["category_zh"]
    if attribute_phrase:
        title_zh = f"{title_zh}，{attribute_phrase}"
    upgraded = dict(fact)
    upgraded.update(
        {
            "dataset_version": FACT_DATASET_VERSION,
            "fact_id": f"{_slug(spec['category'])}:{fact['product_id']}",
            "category_zh": spec["category_zh"],
            "title_en": fact["title"],
            "title_zh": title_zh,
            "body_en": fact["title"],
            "body_zh": f"基于ESCI真实英文商品文本生成的离线中文释义：{title_zh}",
            "bilingual_translation": {
                "kind": "model_authored_deterministic_translation",
                "source_product_id": fact["product_id"],
                "review_status": "machine_generated_unreviewed",
            },
            "text_supported_attributes": translated_attributes,
            "source_kind": "real_esci",
        }
    )
    return upgraded


def _synthetic_fact(spec: dict[str, Any], index: int) -> dict[str, Any]:
    form_en, form_zh = spec["forms"][index % len(spec["forms"])]
    attributes = []
    for facet_index, facet in enumerate(spec["facets"]):
        value_en, value_zh = facet["values"][(index + facet_index * 2) % len(facet["values"])]
        attributes.append(
            {
                "name": facet["name"],
                "name_zh": facet["name_zh"],
                "value": value_en,
                "value_zh": value_zh,
                "evidence_kind": SYNTHETIC_SOURCE_KIND,
            }
        )
    key_values_en = ", ".join(attribute["value"] for attribute in attributes[:3])
    key_values_zh = "、".join(attribute["value_zh"] for attribute in attributes[:3])
    product_id = f"SYN-{_slug(spec['category']).upper()}-{index + 1:03d}"
    title_en = f"Globex Offline {form_en}, {key_values_en}, Model {index + 1:03d}"
    title_zh = f"Globex离线样例 {form_zh}，{key_values_zh}，型号{index + 1:03d}"
    body_en = "; ".join(
        f"{attribute['name']}: {attribute['value']}" for attribute in attributes
    )
    body_zh = "；".join(
        f"{attribute['name_zh']}：{attribute['value_zh']}" for attribute in attributes
    )
    return {
        "dataset_version": FACT_DATASET_VERSION,
        "fact_id": f"{_slug(spec['category'])}:{product_id}",
        "category": spec["category"],
        "category_zh": spec["category_zh"],
        "category_kind": "ordinary",
        "product_id": product_id,
        "title": title_en,
        "title_en": title_en,
        "title_zh": title_zh,
        "body_en": body_en,
        "body_zh": body_zh,
        "source_body_sha256": _sha256_text(f"{body_en}\n{body_zh}"),
        "category_assignment": {
            "method": "model_authored_category_specification",
            "strongest_esci_label": None,
            "source_judgments": [],
        },
        "text_supported_attributes": attributes,
        "generated_fields": {
            "typical_price_cny": _generated_number(
                spec["category"], product_id, *spec["price_bounds_cny"], salt="price"
            ),
            "order_count_90d": int(
                _generated_number(spec["category"], product_id, 40, 1000, salt="orders")
            ),
            "price_source": "generated_offline_not_observed",
            "popularity_source": "generated_offline_not_observed",
            "seed": GENERATION_SEED,
            "rule_version": GENERATOR_VERSION,
        },
        "source": {
            "kind": SYNTHETIC_SOURCE_KIND,
            "generator": "Codex model-authored specifications plus deterministic expansion",
            "prompt_version": "bilingual-category-product-v3",
            "generation_seed": GENERATION_SEED,
            "review_status": "machine_generated_unreviewed",
        },
        "source_kind": SYNTHETIC_SOURCE_KIND,
    }


def _build_cards(
    facts: list[dict[str, Any]], specs: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    facts_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        facts_by_category[fact["category"]].append(fact)
    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for category, spec in specs.items():
        category_facts = facts_by_category[category]
        generated = _bestseller_cards(spec, category_facts)
        generated.extend(_attribute_cards(spec, category_facts))
        generated.append(_price_card(spec, category_facts))
        for card, card_provenance, summary_zh in generated:
            admitted = admit_card(card)
            if not admitted.accepted or admitted.card is None:
                raise ValueError(f"v3 card rejected {card['card_id']}: {admitted.reason}")
            serialized = admitted.card.model_dump(mode="json")
            cards.append(serialized)
            provenance.append(card_provenance)
            retrieval_rows.append(
                _retrieval_row(serialized, spec, summary_zh)
            )
    return cards, provenance, retrieval_rows


def _bestseller_cards(
    spec: dict[str, Any], facts: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    ranked = sorted(
        facts,
        key=lambda row: (-row["generated_fields"]["order_count_90d"], row["product_id"]),
    )[:6]
    output = []
    for index in range(2):
        batch = ranked[index * 3 : index * 3 + 3]
        forms = [spec["forms"][(index + offset) % len(spec["forms"])] for offset in range(3)]
        card_id = f"cc-{_slug(spec['category'])}-bestseller-{index + 1:02d}"
        summary = f"{spec['category']}：{' / '.join(form[0] for form in forms)}"
        summary_zh = f"{spec['category_zh']}：{' / '.join(form[1] for form in forms)}"
        evidence = [_bestseller_evidence(row) for row in batch]
        card = {
            "card_id": card_id,
            "category": spec["category"],
            "card_type": "bestseller",
            "summary": summary,
            "raw_evidence": evidence,
            "last_updated": SNAPSHOT_TIME,
            "confidence": _card_confidence(facts),
        }
        provenance = _base_provenance(card_id, facts, batch)
        provenance.update(
            {
                "summary_source": "model_authored_bilingual_category_spec.forms",
                "evidence_source": "mixed real and synthetic offline popularity",
                "not_real_world_bestseller": True,
            }
        )
        output.append((card, provenance, summary_zh))
    return output


def _attribute_cards(
    spec: dict[str, Any], facts: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    by_facet: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fact in facts:
        for attribute in fact["text_supported_attributes"]:
            by_facet[attribute["name"]][attribute["value"]].append(fact)
    output = []
    for index, facet in enumerate(spec["facets"], start=1):
        values = by_facet[facet["name"]]
        valid_count = sum(len(rows) for rows in values.values())
        if valid_count < 10:
            raise ValueError(
                f"{spec['category']} facet {facet['name']} has only {valid_count} facts"
            )
        ranked = sorted(values.items(), key=lambda item: (-len(item[1]), item[0]))[:3]
        value_zh = dict(facet["values"])
        tokens_en = [
            f"{value} {len(rows) / valid_count * 100:.1f}%" for value, rows in ranked
        ]
        tokens_zh = [
            f"{value_zh[value]} {len(rows) / valid_count * 100:.1f}%"
            for value, rows in ranked
        ]
        card_id = f"cc-{_slug(spec['category'])}-attribute-{index:02d}"
        card = {
            "card_id": card_id,
            "category": spec["category"],
            "card_type": "attribute",
            "summary": f"{facet['name']}：{' / '.join(tokens_en)}",
            "raw_evidence": [
                _truncate(f"{value}: {rows[0]['product_id']} {rows[0]['title_en']}")
                for value, rows in ranked
            ],
            "last_updated": SNAPSHOT_TIME,
            "confidence": round(min(0.9, 0.6 + valid_count / len(facts) * 0.3), 2),
        }
        source_rows = [row for _, rows in ranked for row in rows]
        provenance = _base_provenance(card_id, facts, source_rows)
        provenance.update(
            {
                "attribute_name": facet["name"],
                "attribute_name_zh": facet["name_zh"],
                "valid_sample_count": valid_count,
                "distribution_counts": {value: len(rows) for value, rows in ranked},
                "summary_source": "mixed observed and synthetic attribute aggregation",
            }
        )
        output.append(
            (
                card,
                provenance,
                f"{facet['name_zh']}：{' / '.join(tokens_zh)}",
            )
        )
    return output


def _price_card(
    spec: dict[str, Any], facts: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    prices = sorted(row["generated_fields"]["typical_price_cny"] for row in facts)
    minimum = _round_down(prices[0], 10)
    first = _round_up(_percentile(prices, 1 / 3), 10)
    second = _round_up(_percentile(prices, 2 / 3), 10)
    maximum = _round_up(prices[-1], 10)
    first = max(first, minimum + 10)
    second = max(second, first + 10)
    maximum = max(maximum, second + 10)
    card_id = f"cc-{_slug(spec['category'])}-price-range-01"
    summary = (
        f"便宜款 {minimum:.0f}-{first:.0f} / "
        f"中档 {first:.0f}-{second:.0f} / 高端 {second:.0f}-{maximum:.0f}"
    )
    card = {
        "card_id": card_id,
        "category": spec["category"],
        "card_type": "price_range",
        "summary": summary,
        "raw_evidence": [
            f"generated CNY prices: n={len(prices)}, seed={GENERATION_SEED}",
            f"mixed-source offline bounds={spec['price_bounds_cny']}",
        ],
        "last_updated": SNAPSHOT_TIME,
        "confidence": 0.55,
    }
    provenance = _base_provenance(card_id, facts, facts)
    provenance.update(
        {
            "sample_count": len(prices),
            "price_measure": "generated_typical_price_cny",
            "not_observed_transaction_price": True,
        }
    )
    summary_zh = (
        f"入门档 {minimum:.0f}-{first:.0f}元 / "
        f"中端档 {first:.0f}-{second:.0f}元 / 高端档 {second:.0f}-{maximum:.0f}元"
    )
    return card, provenance, summary_zh


def _retrieval_row(
    card: dict[str, Any], spec: dict[str, Any], summary_zh: str
) -> dict[str, Any]:
    type_en = {
        "bestseller": "bestselling product forms",
        "attribute": "product attribute distribution",
        "price_range": "price tiers",
    }[card["card_type"]]
    type_zh = {
        "bestseller": "热门商品形态",
        "attribute": "商品属性分布",
        "price_range": "价格档位",
    }[card["card_type"]]
    summary_en = card["summary"].replace("：", ":")
    forms_en = ", ".join(form[0] for form in spec["forms"])
    forms_zh = "、".join(form[1] for form in spec["forms"])
    text_en = (
        f"Category: {card['category']}. Knowledge type: {type_en}. "
        f"Known product forms: {forms_en}. Summary: {summary_en}"
    )
    text_zh = (
        f"品类：{spec['category_zh']}。知识类型：{type_zh}。"
        f"已知商品形态：{forms_zh}。摘要：{summary_zh}"
    )
    return {
        "card_id": card["card_id"],
        "retrieval_text_en": text_en,
        "retrieval_text_zh": text_zh,
        "retrieval_text_bilingual": f"{text_en} {text_zh}",
    }


def _base_provenance(
    card_id: str, category_facts: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(row["source_kind"] for row in source_rows)
    return {
        "card_id": card_id,
        "source_item_ids": sorted({row["product_id"] for row in source_rows}),
        "category_item_count": len(category_facts),
        "source_kind_counts": dict(sorted(counts.items())),
        "contains_synthetic_evidence": SYNTHETIC_SOURCE_KIND in counts,
        "generator_version": GENERATOR_VERSION,
    }


def _card_confidence(facts: list[dict[str, Any]]) -> float:
    real_ratio = sum(row["source_kind"] == "real_esci" for row in facts) / len(facts)
    return round(min(0.85, 0.55 + real_ratio * 0.25 + min(len(facts), 100) / 1000), 2)


def _audit_queue(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = max(1, math.ceil(len(cards) * 0.1))
    ranked = sorted(
        cards,
        key=lambda card: hashlib.sha256(
            f"{GENERATION_SEED}:{card['card_id']}".encode()
        ).hexdigest(),
    )[:count]
    return [
        {
            "card_id": card["card_id"],
            "review_status": "pending_human_or_model_assisted_review",
            "checks": [
                "category_semantics",
                "bilingual_equivalence",
                "summary_evidence_consistency",
                "synthetic_source_disclosure",
            ],
        }
        for card in ranked
    ]


def _build_taxonomy(specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_version": "category-taxonomy-bilingual-v3",
        "canonical_language": "en",
        "categories": [
            {
                "category": spec["category"],
                "category_zh": spec["category_zh"],
                "category_kind": "ordinary",
                "aliases": [spec["category_zh"], *spec["aliases"]],
            }
            for spec in specs.values()
        ],
    }


def _manifest(
    facts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    audit_queue: list[dict[str, Any]],
    real_facts_path: Path,
    output_paths: dict[str, Path],
    target_per_category: int,
    minimum_synthetic: int,
) -> dict[str, Any]:
    return {
        "dataset_version": CARD_DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generation_seed": GENERATION_SEED,
        "intended_use": "offline learning and separate synthetic bilingual evaluation",
        "truth_boundary": {
            "real_esci": "original accepted ESCI product text facts",
            "translated_from_esci": "machine-generated Chinese interpretation, not observed text",
            SYNTHETIC_SOURCE_KIND: "model-authored specification with deterministic products",
            "price_and_popularity": "generated offline, never observed marketplace values",
        },
        "generation_policy": {
            "target_per_category": target_per_category,
            "minimum_synthetic_per_category": minimum_synthetic,
            "cards_per_category": {
                "bestseller": 2,
                "attribute": 7,
                "price_range": 1,
            },
        },
        "counts": {
            "categories": len({fact["category"] for fact in facts}),
            "facts": len(facts),
            "facts_by_source": dict(sorted(Counter(row["source_kind"] for row in facts).items())),
            "facts_by_category": dict(sorted(Counter(row["category"] for row in facts).items())),
            "cards": len(cards),
            "cards_by_type": dict(sorted(Counter(row["card_type"] for row in cards).items())),
            "cards_by_category": dict(sorted(Counter(row["category"] for row in cards).items())),
            "audit_queue": len(audit_queue),
        },
        "input_sha256": {"real_facts_v1": _sha256_file(real_facts_path)},
        "output_sha256": {
            key: _sha256_file(path) for key, path in output_paths.items()
        },
    }


def _generated_number(
    category: str,
    product_id: str,
    minimum: float,
    maximum: float,
    *,
    salt: str,
) -> float:
    digest = hashlib.sha256(
        f"{GENERATION_SEED}:{salt}:{category}:{product_id}".encode()
    ).hexdigest()
    ratio = int(digest[:12], 16) / float(16**12 - 1)
    return round(minimum + (maximum - minimum) * ratio, 2)


def _percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _round_down(value: float, unit: int) -> float:
    return math.floor(value / unit) * unit


def _round_up(value: float, unit: int) -> float:
    return math.ceil(value / unit) * unit


def _truncate(text: str, limit: int = 80) -> str:
    clean = " ".join(text.replace("|", "/").split())
    return clean if len(clean) <= limit else f"{clean[: limit - 1]}…"


def _bestseller_evidence(row: dict[str, Any], limit: int = 80) -> str:
    """Keep the three course-required pipe-separated evidence fields intact."""
    price = row["generated_fields"]["typical_price_cny"]
    orders = row["generated_fields"]["order_count_90d"]
    suffix = f" | {price:.0f} | generated orders={orders}"
    title_limit = max(1, limit - len(suffix))
    title = " ".join(row["title_en"].replace("|", "/").split())
    if len(title) > title_limit:
        title = f"{title[: max(1, title_limit - 1)]}…"
    return f"{title}{suffix}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
