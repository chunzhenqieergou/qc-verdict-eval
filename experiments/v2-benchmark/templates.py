"""Independent Chinese quality-control templates for v2-benchmark.

Strictly separated between:
  - dev:  2 templates x 12 rules = 24 primary templates (+ 24 semantic paraphrases)
  - test: 4 templates x 12 rules = 48 primary templates (+ 48 semantic paraphrases)

Workpiece / component pools are disjoint between dev, test, and the historical
scenario-sft SETTINGS list.
"""

# Disjoint from historical SETTINGS:
# ["输送机托辊", "泵站垫圈", "冷库门铰链", "分拣机导轮", "包装机滚筒", "纺织机衬套",
#  "水表壳体", "家具滑轨", "纸机轴套", "印刷机隔圈", "饮料线导轨"]
DEV_PARTS: tuple[str, ...] = (
    "空压机阀片",
    "减速箱油封",
    "电动推杆螺母",
    "注塑机顶针",
    "光伏支架卡扣",
    "风机叶轮法兰",
    "叉车链轮销轴",
    "伺服电机端盖",
    "农机齿轮套筒",
    "电梯门机滑块",
    "净水器滤瓶盖",
    "气缸缓冲衬环",
)

TEST_PARTS: tuple[str, ...] = (
    "液压阀块接头",
    "轴承保持架",
    "空调压缩机连杆",
    "汽车雨刮连动杆",
    "智能电表端子座",
    "工业机器人谐波柔轮",
    "储能箱防爆阀盖",
    "缝纫机旋梭座",
    "食品灌装阀芯",
    "激光切割喷嘴",
    "高铁座椅锁舌",
    "热泵换热管卡箍",
    "电动工具齿轮箱壳",
    "医疗输液泵压块",
    "消防喷淋头框架",
    "光伏逆变器散热片",
    "物流滚筒端轴",
    "充电枪锁止插销",
    "数控刀柄拉钉",
    "洗衣机减震吊杆",
    "燃气表膜片压环",
    "水泵机械密封静环",
    "无人机机臂折叠件",
    "风电变桨限位块",
)

# Structure: TEMPLATES[split][rule] = list of dicts with keys:
#   template_id: str
#   style: str
#   primary: str
#   paraphrase: str
TEMPLATES: dict[str, dict[str, list[dict[str, str]]]] = {
    "dev": {
        "single": [
            {
                "template_id": "dev_single_t0",
                "style": "inspection_log",
                "primary": "入库抽检台账（物料：{part}）：本批执行一次计数抽样，样本量{n}件，方案规定接收数为{c}件，实测发现不合格品{x}件。检验员据此提出“该批可按抽样方案接收”。请判断该结论是否成立。",
                "paraphrase": "待核实结论：“该批可按抽样方案接收”。物料为{part}，采用一次计数抽样，共抽取{n}件样本，查出{x}件不合格品，对应接收数为{c}件。请判定上述结论。",
            },
            {
                "template_id": "dev_single_t1",
                "style": "kv_report",
                "primary": "【来料抽验单】品名：{part}；抽样方式：一次计数抽样；样本数：{n}件；接收数：{c}；检出不合格品：{x}件。审核栏填写意见为“本批满足一次抽样接收条件”。该意见是否成立？",
                "paraphrase": "关于{part}的一次计数抽样记录显示，审核意见认为“本批满足一次抽样接收条件”。已知抽检样本{n}件、不合格品数{x}件、方案接收数{c}件，请核实此意见。",
            },
        ],
        "first_accept": [
            {
                "template_id": "dev_first_accept_t0",
                "style": "inspection_log",
                "primary": "二次计数抽样首样记录（{part}）：首样样本量{n1}件，首样接收数{c1}，首样拒收数{r1}。实检首样不合格品{x1}件。现场结论为“首样检验结束即可直接接收该批”。请判断。",
                "paraphrase": "针对{part}批次，现场给出“首样检验结束即可直接接收该批”的判定。其二次抽样首样参数为：样本{n1}件、首样接收数{c1}、首样拒收数{r1}，实测不合格品{x1}件。请核查该判定。",
            },
            {
                "template_id": "dev_first_accept_t1",
                "style": "kv_report",
                "primary": "【首样复核】工件：{part}｜方案：二次抽样｜首样量：{n1}件｜首样接收数：{c1}｜首样拒收数：{r1}｜首样不合格品：{x1}件。提交结论：“无需抽第二样本，首样即可接收”。请审核。",
                "paraphrase": "质检班组就{part}提交结论“无需抽第二样本，首样即可接收”。依据二次抽样记录，首样共抽{n1}件，发现不合格品{x1}件，方案首样接收数为{c1}、拒收数为{r1}。该结论是否成立？",
            },
        ],
        "first_reject": [
            {
                "template_id": "dev_first_reject_t0",
                "style": "inspection_log",
                "primary": "二次抽样首样检验单（{part}）：首样抽检{n1}件，方案设定首样接收数为{c1}、首样拒收数为{r1}，实测检出不合格品{x1}件。报告写明“首样结束即可直接拒收该批”。请核实。",
                "paraphrase": "待审说法：“首样结束即可直接拒收该批”。在{part}的二次抽样首样检查中，共抽{n1}件、检出{x1}件不合格品，首样接收数与拒收数分别为{c1}与{r1}。该说法是否成立？",
            },
            {
                "template_id": "dev_first_reject_t1",
                "style": "kv_report",
                "primary": "【首样拒收核对】物料：{part}；二次抽样首样量{n1}件；首样接收数{c1}，首样拒收数{r1}；实测不合格品{x1}件。检验结论：“本批在首样阶段即应判定拒收”。请判断。",
                "paraphrase": "检验员对{part}给出“本批在首样阶段即应判定拒收”的结论。已知二次抽样首样样本{n1}件，不合格品计{x1}件，对应首样接收数{c1}、拒收数{r1}。请判定结论真伪。",
            },
        ],
        "second_accept": [
            {
                "template_id": "dev_second_accept_t0",
                "style": "inspection_log",
                "primary": "{part}批次已进入二次抽样第二阶段：首样{n1}件中检出不合格品{x1}件，次样{n2}件中检出不合格品{x2}件，方案累计接收数为{c2}。质检员认为“两次抽样合并后可接收该批”。请判断。",
                "paraphrase": "拟核定结论：“两次抽样合并后可接收该批”。{part}二次抽样累计接收数为{c2}；实测首样（{n1}件）有{x1}件不合格，次样（{n2}件）有{x2}件不合格。请审核该结论。",
            },
            {
                "template_id": "dev_second_accept_t1",
                "style": "kv_report",
                "primary": "【二次抽样汇总】产品：{part}；首样{n1}件不合格{x1}件；次样{n2}件不合格{x2}件；两样累计接收数：{c2}。上报意见为“完成二次抽样后准予接收”。该意见成立吗？",
                "paraphrase": "关于{part}的上报意见是“完成二次抽样后准予接收”。已知该二次抽样方案两样累计接收数为{c2}，第一样本{n1}件中不合格{x1}件，第二样本{n2}件中不合格{x2}件。请判断。",
            },
        ],
        "weco3": [
            {
                "template_id": "dev_weco3_t0",
                "style": "inspection_log",
                "primary": "{part}加工工序SPC监控记录：当前巡检点相对中心线的标准化偏差为{z}。质量员标记“该监测点触发上侧三西格玛单点越限告警”。请核实此标记是否成立。",
                "paraphrase": "质量员对{part}当前SPC巡检点提出“该监测点触发上侧三西格玛单点越限告警”的判断。系统测得该点相对中心线的标准化偏差为{z}。请判定该判断。",
            },
            {
                "template_id": "dev_weco3_t1",
                "style": "kv_report",
                "primary": "【控制图单点审核】工件：{part}；本期样本点标准化偏差：{z}；核查规则：上侧单点3σ越限。审核单结论为“本期点已触发上侧3σ越限”。请判断。",
                "paraphrase": "针对{part}的控制图审核单结论称“本期点已触发上侧3σ越限”。已知本期样本点相对中心线的标准化偏差读数为{z}，请核验该结论是否成立。",
            },
        ],
        "weco2": [
            {
                "template_id": "dev_weco2_t0",
                "style": "inspection_log",
                "primary": "{part}工序控制图连续3个监测点相对中心线的标准化偏差依次为{seq}。统计量取其中严格高于+2的点数。现场判断：“触发上侧WECO的3点中至少2点高于+2规则”。请判断。",
                "paraphrase": "现场就{part}控制图给出判断：“触发上侧WECO的3点中至少2点高于+2规则”。最近连续3点标准化偏差读数依次为{seq}，统计量按严格高于+2的点数计。该判断是否成立？",
            },
            {
                "template_id": "dev_weco2_t1",
                "style": "kv_report",
                "primary": "【WECO告警复核】工序件：{part}；最近3点标准化偏差：{seq}；口径：统计量取严格高于+2的点数。提交意见：“满足上侧WECO 3点中至少2点高于+2的报警条件”。请审核。",
                "paraphrase": "关于{part}的SPC提交意见为“满足上侧WECO 3点中至少2点高于+2的报警条件”。记录显示最近3个点相对中心线的标准化偏差依次为{seq}，统计量记为严格高于+2的点数。请判断。",
            },
        ],
        "weco1": [
            {
                "template_id": "dev_weco1_t0",
                "style": "inspection_log",
                "primary": "{part}产线控制图最近连续5个样本点相对中心线的标准化偏差依次为{seq}。统计量取其中严格高于+1的点数。值班记录写明：“触发上侧WECO的5点中至少4点高于+1规则”。请判断。",
                "paraphrase": "值班记录对{part}产线写明“触发上侧WECO的5点中至少4点高于+1规则”。最近5点相对中心线的标准化偏差为{seq}，统计量计为严格高于+1的点数。请复核此结论。",
            },
            {
                "template_id": "dev_weco1_t1",
                "style": "kv_report",
                "primary": "【5点窗口监控】产品：{part}；连续5点标准化偏差：{seq}；统计量定义：严格高于+1的点数。审核声明：“上侧WECO 5点中至少4点高于+1告警成立”。请判定。",
                "paraphrase": "审核声明称{part}的“上侧WECO 5点中至少4点高于+1告警成立”。已知连续5个监测点相对中心线的标准化偏差序列为{seq}，统计量取严格高于+1的点数。请核查。",
            },
        ],
        "weco8": [
            {
                "template_id": "dev_weco8_t0",
                "style": "inspection_log",
                "primary": "{part}关键尺寸控制图最近连续8点相对中心线的标准化偏差依次为{seq}。统计量取其中严格高于+0的点数。工程部结论：“触发上侧WECO的连续8点均高于中心线规则”。请判断。",
                "paraphrase": "工程部针对{part}提出“触发上侧WECO的连续8点均高于中心线规则”的结论。最近8个监测点相对中心线的标准化偏差依次为{seq}，统计量按严格高于+0的点数计算。该结论成立吗？",
            },
            {
                "template_id": "dev_weco8_t1",
                "style": "kv_report",
                "primary": "【8点连侧核查】零件：{part}；最近8点标准化偏差：{seq}；统计量：严格高于+0的点数。报告结论：“已触发上侧WECO 8点中至少8点高于+0告警”。请审核。",
                "paraphrase": "报告对{part}作出“已触发上侧WECO 8点中至少8点高于+0告警”的判断。查得连续8点相对中心线的标准化偏差依次为{seq}，统计量取严格高于+0的点数。请核实。",
            },
        ],
        "cp": [
            {
                "template_id": "dev_cp_t0",
                "style": "inspection_log",
                "primary": "{part}尺寸加工过程稳定且近似正态。已知过程标准差为{sd}，规格限为{lo}至{hi}，合同要求Cp门槛为{target}。试产报告称“该工序Cp达到合同门槛{target}”。请判断。",
                "paraphrase": "试产报告宣称{part}“该工序Cp达到合同门槛{target}”。已知该稳定正态工序的标准差为{sd}，下规格限{lo}、上规格限{hi}。请复核该结论是否成立。",
            },
            {
                "template_id": "dev_cp_t1",
                "style": "kv_report",
                "primary": "【过程能力评估】产品：{part}（稳定正态分布）；下规格限：{lo}；上规格限：{hi}；标准差：{sd}；Cp目标值：{target}。待审结论：“工序潜在能力指数Cp满足目标{target}”。请判定。",
                "paraphrase": "待审结论为“工序潜在能力指数Cp满足目标{target}”。{part}工序处于统计稳定且近似正态状态，标准差{sd}，规格区间{lo}至{hi}。请判断该结论。",
            },
        ],
        "cpu": [
            {
                "template_id": "dev_cpu_t0",
                "style": "inspection_log",
                "primary": "{part}单侧上限受控工序稳定且近似正态。实测过程均值为{mean}，标准差为{sd}，上规格限为{hi}，要求Cpu门槛为{target}。质量总结认为“该工序Cpu达到门槛{target}”。请判断。",
                "paraphrase": "质量总结对{part}给出“该工序Cpu达到门槛{target}”的判定。已知稳定正态过程均值{mean}、标准差{sd}、上规格限{hi}。请审核此判定是否成立。",
            },
            {
                "template_id": "dev_cpu_t1",
                "style": "kv_report",
                "primary": "【上单侧能力审核】物料：{part}（稳定近似正态）；过程均值：{mean}；上规格限：{hi}；标准差：{sd}；Cpu要求：{target}。结论栏写明：“CPU达到规定门槛{target}”。请核查。",
                "paraphrase": "结论栏针对{part}写明“CPU达到规定门槛{target}”。查该稳定近似正态工序的均值为{mean}，标准差为{sd}，上规格限为{hi}。该结论是否成立？",
            },
        ],
        "cpl": [
            {
                "template_id": "dev_cpl_t0",
                "style": "inspection_log",
                "primary": "{part}单侧下限受控工序稳定且近似正态。过程均值为{mean}，标准差为{sd}，下规格限为{lo}，规定Cpl门槛为{target}。工艺单称“该工序Cpl达到门槛{target}”。请判断。",
                "paraphrase": "工艺单声称{part}“该工序Cpl达到门槛{target}”。已知该稳定近似正态过程均值为{mean}、标准差为{sd}、下规格限为{lo}。请判定该声称是否成立。",
            },
            {
                "template_id": "dev_cpl_t1",
                "style": "kv_report",
                "primary": "【下单侧能力审核】工件：{part}（稳定近似正态）；均值：{mean}；下规格限：{lo}；标准差：{sd}；Cpl目标：{target}。提交结论：“CPL满足门槛要求{target}”。请审核。",
                "paraphrase": "提交结论认为{part}的“CPL满足门槛要求{target}”。其稳定近似正态工序参数为：过程均值{mean}、下规格限{lo}、标准差{sd}。请核实。",
            },
        ],
        "cpk": [
            {
                "template_id": "dev_cpk_t0",
                "style": "inspection_log",
                "primary": "{part}关键特征加工过程稳定且近似正态。已知均值{mean}、标准差{sd}、规格限{lo}至{hi}，客户规定Cpk门槛为{target}。报告结论为“该工序Cpk达到门槛{target}”。请判断。",
                "paraphrase": "报告对{part}关键特征作出“该工序Cpk达到门槛{target}”的结论。已知该稳定近似正态过程的规格限为{lo}至{hi}，均值为{mean}，标准差为{sd}。请审核。",
            },
            {
                "template_id": "dev_cpk_t1",
                "style": "kv_report",
                "primary": "【Cpk准入复核】零件：{part}（稳定近似正态）；规格限：{lo}～{hi}；均值：{mean}；标准差：{sd}；门槛：{target}。拟签意见：“过程能力指数CPK达到{target}”。请判断。",
                "paraphrase": "拟签意见称{part}的“过程能力指数CPK达到{target}”。依据稳定近似正态统计数据，标准差为{sd}，均值为{mean}，下规格限{lo}，上规格限{hi}。该意见是否成立？",
            },
        ],
    },
    "test": {
        "single": [
            {
                "template_id": "test_single_t0",
                "style": "receiving_ticket",
                "primary": "供应商发来一批{part}，进料检验按一次计数抽样方案抽取{n}件。方案规定的接收数为{c}件，本次实际查出不合格品{x}件。检验员在放行单上勾选了“本批符合一次抽样接收标准”。请判断该结论是否成立。",
                "paraphrase": "进料放行单上对{part}勾选了“本批符合一次抽样接收标准”。核对原始抽检数据：一次计数抽样共取样{n}件，发现{x}件不合格，对应方案接收数为{c}件。请判定该结论。",
            },
            {
                "template_id": "test_single_t1",
                "style": "claim_first_audit",
                "primary": "待审核声明：“该批{part}可依据一次计数抽样方案予以接收”。检验台账记载：抽检样本量{n}件，方案允许的接收数c为{c}，实测不合格件数为{x}。请复核该声明。",
                "paraphrase": "检验台账显示{part}一次计数抽样取样{n}件，实测不合格{x}件，方案允许接收数c为{c}。据此有人提出“该批{part}可依据一次计数抽样方案予以接收”，请判断是否成立。",
            },
            {
                "template_id": "test_single_t2",
                "style": "shift_handover",
                "primary": "夜班交接时反馈：刚完成的{part}一次计数抽样共检了{n}件，挑出{x}件不合格品，而这批的接收数是{c}件，班长认为“按方案本批可以通过验收”。请核查班长说法是否正确。",
                "paraphrase": "班长在交接班时称“按方案本批可以通过验收”。该批{part}执行一次计数抽样，接收数为{c}件，从{n}件样本中检出了{x}件不合格品。请判断这一说法是否成立。",
            },
            {
                "template_id": "test_single_t3",
                "style": "structured_card",
                "primary": "质量放行核验卡｜品名：{part}｜抽样类型：一次计数抽样｜抽样件数：{n}｜接收数：{c}｜实测不合格数：{x}｜拟定判定：准予按方案接收。请问拟定判定是否成立？",
                "paraphrase": "针对{part}的质量放行核验卡拟定判定为“准予按方案接收”。卡片参数如下：一次计数抽样、样本{n}件、实测不合格{x}件、接收数{c}件。请审核该判定。",
            },
        ],
        "first_accept": [
            {
                "template_id": "test_first_accept_t0",
                "style": "receiving_ticket",
                "primary": "品控部对{part}实施二次计数抽样检验。第一样本抽检{n1}件，方案规定首样接收数为{c1}、首样拒收数为{r1}，实检发现不合格品{x1}件。记录结论称“仅凭第一样本即可直接接收本批”。请判断。",
                "paraphrase": "品控部记录对{part}作出“仅凭第一样本即可直接接收本批”的结论。已知本批采用二次计数抽样，首样样本{n1}件、查出不合格品{x1}件，首样接收数{c1}、拒收数{r1}。请核实。",
            },
            {
                "template_id": "test_first_accept_t1",
                "style": "claim_first_audit",
                "primary": "待确认意见：“{part}在二次抽样首样结束后即满足直接接收条件”。原始检验数据：首样样本量{n1}件，不合格品实测{x1}件，首样接收数{c1}，首样拒收数{r1}。请审核。",
                "paraphrase": "查阅{part}二次抽样原始数据：首样{n1}件中不合格品{x1}件，首样接收数与拒收数分别为{c1}和{r1}。现需判断“{part}在二次抽样首样结束后即满足直接接收条件”是否成立。",
            },
            {
                "template_id": "test_first_accept_t2",
                "style": "shift_handover",
                "primary": "检验组汇报称，{part}批次按二次抽样先抽了{n1}件，查出{x1}件不良，方案写明首样接收数{c1}、拒收数{r1}，组长打算按“首样直接接收、不再抽次样”放行。该处置是否成立？",
                "paraphrase": "检验组长拟对{part}执行“首样直接接收、不再抽次样”的放行方案。已知二次抽样首样接收数为{c1}、拒收数为{r1}，首样实抽{n1}件并检出{x1}件不良。请判定该方案是否成立。",
            },
            {
                "template_id": "test_first_accept_t3",
                "style": "structured_card",
                "primary": "二次抽样首阶段核查表｜物料：{part}｜首样量：{n1}｜首样接收数：{c1}｜首样拒收数：{r1}｜首样不合格数：{x1}｜申报结论：首样即可直接接收。请判断该结论。",
                "paraphrase": "关于{part}的二次抽样首阶段核查表申报结论为“首样即可直接接收”。表内数据为：首样不合格数{x1}、首样量{n1}、首样接收数{c1}、首样拒收数{r1}。请审核。",
            },
        ],
        "first_reject": [
            {
                "template_id": "test_first_reject_t0",
                "style": "receiving_ticket",
                "primary": "这批{part}采用二次计数抽样。首样共抽检{n1}件，方案设定的首样接收数为{c1}、首样拒收数为{r1}，现场检出不合格品{x1}件。检验报告提出“无需加抽次样，首样即可直接拒收”。请判断。",
                "paraphrase": "针对{part}的检验报告提出“无需加抽次样，首样即可直接拒收”。复查二次计数抽样首样记录：样本{n1}件、不合格品{x1}件、首样接收数{c1}、首样拒收数{r1}。请判定。",
            },
            {
                "template_id": "test_first_reject_t1",
                "style": "claim_first_audit",
                "primary": "待复核结论：“本批{part}在二次抽样首样阶段即达到直接拒收标准”。已知首样抽检{n1}件，实测不合格品{x1}件，方案首样接收数为{c1}，首样拒收数为{r1}。请判断。",
                "paraphrase": "已知{part}二次抽样首样抽检{n1}件，首样接收数{c1}、首样拒收数{r1}，实测不合格品{x1}件。试判定“本批{part}在二次抽样首样阶段即达到直接拒收标准”是否成立。",
            },
            {
                "template_id": "test_first_reject_t2",
                "style": "shift_handover",
                "primary": "来料区反映：对{part}按二次抽样抽了首样{n1}件，查出{x1}件不合格，方案的首样接收数是{c1}、拒收数是{r1}，值班品管直接开了“首样拒收退货”的结论。该结论成立吗？",
                "paraphrase": "值班品管就{part}开出“首样拒收退货”结论。核实其二次抽样首样信息：首样接收数{c1}、首样拒收数{r1}，实抽{n1}件中发现不合格品{x1}件。请判断此结论是否成立。",
            },
            {
                "template_id": "test_first_reject_t3",
                "style": "structured_card",
                "primary": "二次抽样首样判定卡｜品名：{part}｜首样样本量：{n1}｜首样接收数：{c1}｜首样拒收数：{r1}｜实测不合格品：{x1}｜申报判定：首样结束即判拒收。请核实。",
                "paraphrase": "{part}二次抽样首样判定卡的申报判定为“首样结束即判拒收”。卡片记载：实测不合格品{x1}件（首样样本量{n1}件），首样接收数{c1}、拒收数{r1}。请判断。",
            },
        ],
        "second_accept": [
            {
                "template_id": "test_second_accept_t0",
                "style": "receiving_ticket",
                "primary": "{part}批次因首样未决而按二次抽样方案追加了第二样本。首样{n1}件中发现{x1}件不合格，次样{n2}件中发现{x2}件不合格，方案两样累计接收数为{c2}。报告认为“完成两次抽样后该批可予接收”。请判断。",
                "paraphrase": "检验报告认为{part}“完成两次抽样后该批可予接收”。该批已执行完二次抽样：方案累计接收数{c2}，第一样本{n1}件查出{x1}件不合格，第二样本{n2}件查出{x2}件不合格。请核实。",
            },
            {
                "template_id": "test_second_accept_t1",
                "style": "claim_first_audit",
                "primary": "待审意见：“{part}两阶段抽样汇总后符合累计接收要求”。检验记录：第一样本量{n1}件、不合格{x1}件；第二样本量{n2}件、不合格{x2}件；方案累计接收数为{c2}。请审核。",
                "paraphrase": "根据{part}二次抽样记录，首样（{n1}件）不合格{x1}件，次样（{n2}件）不合格{x2}件，累计接收数为{c2}。请判断“{part}两阶段抽样汇总后符合累计接收要求”是否成立。",
            },
            {
                "template_id": "test_second_accept_t2",
                "style": "shift_handover",
                "primary": "车间品检转来记录：{part}走完了二次抽样，首样{n1}件有{x1}件不良，补抽的次样{n2}件又有{x2}件不良，方案累计接收数是{c2}。班组申请“按合格批放行接收”。请判断是否成立。",
                "paraphrase": "班组就{part}申请“按合格批放行接收”。已知该批走完二次抽样，累计接收数为{c2}，首样{n1}件检出{x1}件不良，补抽次样{n2}件检出{x2}件不良。请审核该申请。",
            },
            {
                "template_id": "test_second_accept_t3",
                "style": "structured_card",
                "primary": "二次抽样终审单｜物料：{part}｜首样：{n1}件（不合格{x1}件）｜次样：{n2}件（不合格{x2}件）｜累计接收数：{c2}｜拟定意见：两次抽样合计可接收。请核验。",
                "paraphrase": "{part}二次抽样终审单的拟定意见是“两次抽样合计可接收”。单据数据：累计接收数{c2}，首样{n1}件中不合格{x1}件，次样{n2}件中不合格{x2}件。请判断意见是否成立。",
            },
        ],
        "weco3": [
            {
                "template_id": "test_weco3_t0",
                "style": "receiving_ticket",
                "primary": "在{part}关键尺寸的计量控制图上，本班最新监测点相对中心线的标准化偏差为{z}。SPC监控员据此填写“该点触发上侧三西格玛越限报警”。请判断该结论是否成立。",
                "paraphrase": "SPC监控员对{part}最新监测点填写“该点触发上侧三西格玛越限报警”。查控制图记录，该点相对中心线的标准化偏差为{z}。请复核此结论。",
            },
            {
                "template_id": "test_weco3_t1",
                "style": "claim_first_audit",
                "primary": "待核查判断：“{part}当前样本点已触发上侧3σ单点越限规则”。监控系统记录该点相对中心线的标准化偏差为{z}。请判定上述判断。",
                "paraphrase": "监控系统测得{part}当前样本点相对中心线的标准化偏差为{z}。据此能否得出“{part}当前样本点已触发上侧3σ单点越限规则”的结论？请判断。",
            },
            {
                "template_id": "test_weco3_t2",
                "style": "shift_handover",
                "primary": "产线交接日志提到，{part}工位最新打出的一个控制图点，相对中心线标准化偏差读数为{z}，操作员随即标注为“上侧3西格玛单点超限触发”。请判断标注是否成立。",
                "paraphrase": "操作员在交接日志中把{part}最新控制图点标注为“上侧3西格玛单点超限触发”。已知该点相对中心线的标准化偏差读数为{z}，请审核该标注。",
            },
            {
                "template_id": "test_weco3_t3",
                "style": "structured_card",
                "primary": "SPC单点判异卡｜部件：{part}｜监测项：上侧3σ越限｜当前点标准化偏差：{z}｜系统结论：触发上侧三西格玛越限。请问系统结论是否成立？",
                "paraphrase": "关于{part}的SPC单点判异卡给出系统结论“触发上侧三西格玛越限”。卡片显示当前点相对中心线的标准化偏差为{z}。请核验该结论。",
            },
        ],
        "weco2": [
            {
                "template_id": "test_weco2_t0",
                "style": "receiving_ticket",
                "primary": "{part}加工线控制图记录了最近连续3个监测点相对中心线的标准化偏差：{seq}。统计量取其中严格高于+2的点数。质量报告称“已触发上侧WECO的3点中至少2点高于+2规则”。请判断。",
                "paraphrase": "质量报告称{part}加工线“已触发上侧WECO的3点中至少2点高于+2规则”。核对最近连续3点相对中心线的标准化偏差为{seq}，统计量计为严格高于+2的点数。请审核。",
            },
            {
                "template_id": "test_weco2_t1",
                "style": "claim_first_audit",
                "primary": "待审结论：“{part}满足上侧WECO 3点中至少2点高于+2的判异条件”。最近3个样本点相对中心线的标准化偏差依次为{seq}，统计量按严格高于+2的点数计。请判断。",
                "paraphrase": "{part}最近3个样本点相对中心线的标准化偏差依次为{seq}，统计量取严格高于+2的点数。请判断“{part}满足上侧WECO 3点中至少2点高于+2的判异条件”是否成立。",
            },
            {
                "template_id": "test_weco2_t2",
                "style": "shift_handover",
                "primary": "巡检员抄录了{part}控制图末尾3点的标准化偏差，依次是{seq}（统计量取严格高于+2的点数），并在异常单上写了“触发上侧WECO三点中至少两点高于+2报警”。请核实。",
                "paraphrase": "巡检员在异常单上就{part}写下“触发上侧WECO三点中至少两点高于+2报警”。所抄末尾3点相对中心线的标准化偏差依次为{seq}，统计量按严格高于+2的点数计。请判断。",
            },
            {
                "template_id": "test_weco2_t3",
                "style": "structured_card",
                "primary": "控制图窗口判异表｜零件：{part}｜规则：上侧WECO 3点中至少2点高于+2｜连续3点标准化偏差：{seq}｜统计量：严格高于+2的点数｜结论：规则已触发。请判断。",
                "paraphrase": "{part}控制图窗口判异表的结论为“规则已触发”（规则：上侧WECO 3点中至少2点高于+2）。表中连续3点标准化偏差依次为{seq}，统计量取严格高于+2的点数。请复核。",
            },
        ],
        "weco1": [
            {
                "template_id": "test_weco1_t0",
                "style": "receiving_ticket",
                "primary": "在{part}工序的SPC巡检中，最近连续5个点相对中心线的标准化偏差依次为{seq}。统计量取其中严格高于+1的点数。品质科判定“触发上侧WECO的5点中至少4点高于+1规则”。请判断。",
                "paraphrase": "品质科对{part}工序判定“触发上侧WECO的5点中至少4点高于+1规则”。原始5点窗口相对中心线的标准化偏差依次为{seq}，统计量记为严格高于+1的点数。请核实。",
            },
            {
                "template_id": "test_weco1_t1",
                "style": "claim_first_audit",
                "primary": "待审核声明：“{part}控制图已触发上侧WECO 5点中至少4点高于+1报警”。已知连续5个监测点相对中心线的标准化偏差为{seq}，统计量按严格高于+1的点数统计。请判断。",
                "paraphrase": "已知{part}连续5个监测点相对中心线的标准化偏差依次为{seq}，统计量取严格高于+1的点数。据此判断“{part}控制图已触发上侧WECO 5点中至少4点高于+1报警”是否成立。",
            },
            {
                "template_id": "test_weco1_t2",
                "style": "shift_handover",
                "primary": "早班汇总指出，{part}控制图最近5点标准化偏差分别是{seq}，按严格高于+1的点数作统计量，工程师据此发出“上侧WECO五点中至少四点高于+1越区告警”。该告警是否成立？",
                "paraphrase": "工程师针对{part}发出“上侧WECO五点中至少四点高于+1越区告警”。查早班控制图最近5点相对中心线的标准化偏差为{seq}，统计量取严格高于+1的点数。请审核。",
            },
            {
                "template_id": "test_weco1_t3",
                "style": "structured_card",
                "primary": "SPC 5点规则复核单｜部件：{part}｜规则：上侧WECO 5点中至少4点高于+1｜近5点标准化偏差：{seq}｜统计量：严格高于+1的点数｜申报结论：触发报警。请判断。",
                "paraphrase": "{part}的SPC 5点规则复核单申报结论为“触发报警”（对应上侧WECO 5点中至少4点高于+1规则）。单据所列近5点标准化偏差为{seq}，统计量按严格高于+1的点数计。请判断。",
            },
        ],
        "weco8": [
            {
                "template_id": "test_weco8_t0",
                "style": "receiving_ticket",
                "primary": "{part}产线计量控制图最近连续8个监测点相对中心线的标准化偏差依次为{seq}。统计量取其中严格高于+0的点数。分析单写明“触发上侧WECO的8点中至少8点高于+0规则”。请判断。",
                "paraphrase": "分析单对{part}产线写明“触发上侧WECO的8点中至少8点高于+0规则”。已知最近连续8点相对中心线的标准化偏差为{seq}，统计量按严格高于+0的点数计。请审核。",
            },
            {
                "template_id": "test_weco8_t1",
                "style": "claim_first_audit",
                "primary": "待核验意见：“{part}触发上侧WECO连续8点均严格高于中心线（+0）规则”。记录显示连续8点相对中心线的标准化偏差依次为{seq}，统计量取严格高于+0的点数。请判断。",
                "paraphrase": "记录显示{part}连续8点相对中心线的标准化偏差依次为{seq}，统计量取严格高于+0的点数。请核验“{part}触发上侧WECO连续8点均严格高于中心线（+0）规则”是否成立。",
            },
            {
                "template_id": "test_weco8_t2",
                "style": "shift_handover",
                "primary": "工艺员抄录了{part}控制图最近8个点的标准化偏差：{seq}，以严格高于+0的点数为统计量，认为“已经满足上侧WECO连续8点高于中心线的报警条件”。请判断是否成立。",
                "paraphrase": "工艺员认为{part}“已经满足上侧WECO连续8点高于中心线的报警条件”。所抄最近8点相对中心线的标准化偏差依次为{seq}，统计量取严格高于+0的点数。请核实。",
            },
            {
                "template_id": "test_weco8_t3",
                "style": "structured_card",
                "primary": "SPC 8点连侧审核卡｜产品：{part}｜规则：上侧WECO 8点中至少8点高于+0｜近8点标准化偏差：{seq}｜统计量：严格高于+0的点数｜结论：已触发告警。请判断。",
                "paraphrase": "{part}的SPC 8点连侧审核卡给出“已触发告警”的结论（规则为上侧WECO 8点中至少8点高于+0）。卡片记录近8点标准化偏差依次为{seq}，统计量取严格高于+0的点数。请判断。",
            },
        ],
        "cp": [
            {
                "template_id": "test_cp_t0",
                "style": "receiving_ticket",
                "primary": "{part}精加工工序处于统计稳定且近似正态。检验报告给出规格限为{lo}至{hi}，过程标准差为{sd}，技术协议规定Cp门槛为{target}。报告结论为“CP达到协议门槛{target}”。请判断。",
                "paraphrase": "检验报告对{part}精加工工序作出“CP达到协议门槛{target}”的结论。已知该稳定近似正态过程的标准差为{sd}，规格限为{lo}至{hi}。请复核该结论是否成立。",
            },
            {
                "template_id": "test_cp_t1",
                "style": "claim_first_audit",
                "primary": "待审核结论：“{part}工序的过程能力指数CP达到门槛{target}”。已知该工序稳定且近似正态，下规格限{lo}，上规格限{hi}，过程标准差{sd}。请判断。",
                "paraphrase": "已知{part}工序稳定且近似正态，过程标准差为{sd}，规格区间为{lo}至{hi}。请审核“{part}工序的过程能力指数CP达到门槛{target}”这一结论。",
            },
            {
                "template_id": "test_cp_t2",
                "style": "shift_handover",
                "primary": "量产评审会上，工艺组汇报{part}工序稳定且近似正态，标准差{sd}，规格上下限分别是{hi}和{lo}，声称“本工序CP满足合同门槛{target}”。请核查该说法是否成立。",
                "paraphrase": "工艺组在评审会上声称{part}“本工序CP满足合同门槛{target}”。该稳定近似正态工序的规格下限为{lo}、上限为{hi}、标准差为{sd}。请判断。",
            },
            {
                "template_id": "test_cp_t3",
                "style": "structured_card",
                "primary": "工序能力核验单（CP）｜零件：{part}｜分布前提：稳定近似正态｜下规格限：{lo}｜上规格限：{hi}｜标准差：{sd}｜合同门槛：{target}｜结论：CP达标。请判断。",
                "paraphrase": "{part}工序能力核验单（CP）给出的结论是“CP达标”（合同门槛{target}）。单据参数：稳定近似正态、标准差{sd}、下规格限{lo}、上规格限{hi}。请审核。",
            },
        ],
        "cpu": [
            {
                "template_id": "test_cpu_t0",
                "style": "receiving_ticket",
                "primary": "{part}关键参数仅考核上规格限，加工过程稳定且近似正态。实测均值{mean}、标准差{sd}、上规格限{hi}，要求Cpu不低于门槛{target}。报告写明“CPU达到门槛{target}”。请判断。",
                "paraphrase": "质量报告对{part}写明“CPU达到门槛{target}”。该稳定近似正态过程的上规格限为{hi}，实测均值为{mean}，标准差为{sd}。请判断该结论是否成立。",
            },
            {
                "template_id": "test_cpu_t1",
                "style": "claim_first_audit",
                "primary": "待复核判断：“{part}的上单侧过程能力指数CPU达到合同门槛{target}”。已知过程稳定且近似正态，均值为{mean}，上规格限为{hi}，标准差为{sd}。请核实。",
                "paraphrase": "已知{part}过程稳定且近似正态，上规格限{hi}、均值{mean}、标准差{sd}。请问“{part}的上单侧过程能力指数CPU达到合同门槛{target}”是否成立？",
            },
            {
                "template_id": "test_cpu_t2",
                "style": "shift_handover",
                "primary": "工程部提交{part}单侧能力数据：过程稳定且近似正态，均值{mean}，标准差{sd}，上规格限{hi}，并据此主张“CPU已达到要求门槛{target}”。请核查该主张。",
                "paraphrase": "工程部主张{part}的“CPU已达到要求门槛{target}”。核对其稳定近似正态过程数据：上规格限为{hi}，均值为{mean}，标准差为{sd}。请判断是否成立。",
            },
            {
                "template_id": "test_cpu_t3",
                "style": "structured_card",
                "primary": "上单侧能力审核卡（CPU）｜物料：{part}｜前提：稳定近似正态｜均值：{mean}｜上规格限：{hi}｜标准差：{sd}｜门槛：{target}｜申报结论：CPU达标。请判定。",
                "paraphrase": "{part}上单侧能力审核卡（CPU）申报结论为“CPU达标”（要求门槛{target}）。卡面数据：稳定近似正态、标准差{sd}、均值{mean}、上规格限{hi}。请判断。",
            },
        ],
        "cpl": [
            {
                "template_id": "test_cpl_t0",
                "style": "receiving_ticket",
                "primary": "{part}强度指标重点考核下规格限，生产过程稳定且近似正态。测得过程均值{mean}、标准差{sd}、下规格限{lo}，客户规定Cpl门槛为{target}。报告宣称“CPL达到门槛{target}”。请判断。",
                "paraphrase": "检测报告宣称{part}的“CPL达到门槛{target}”。已知该稳定近似正态过程的下规格限为{lo}，均值为{mean}，标准差为{sd}。请审核该宣称是否成立。",
            },
            {
                "template_id": "test_cpl_t1",
                "style": "claim_first_audit",
                "primary": "待确认结论：“{part}的下单侧过程能力指数CPL满足门槛{target}”。前提为过程稳定且近似正态，均值{mean}，下规格限{lo}，标准差{sd}。请判定。",
                "paraphrase": "{part}生产过程稳定且近似正态，测得均值{mean}、标准差{sd}，下规格限为{lo}。据此判断“{part}的下单侧过程能力指数CPL满足门槛{target}”是否成立。",
            },
            {
                "template_id": "test_cpl_t2",
                "style": "shift_handover",
                "primary": "质量例会上反馈：{part}工序稳定且近似正态，均值为{mean}、标准差为{sd}、下规格限为{lo}，车间据此填报“CPL已达到合同门槛{target}”。请复核填报内容。",
                "paraphrase": "车间在例会上填报{part}的“CPL已达到合同门槛{target}”。已知该稳定近似正态工序的下规格限为{lo}、均值为{mean}、标准差为{sd}。请判断是否成立。",
            },
            {
                "template_id": "test_cpl_t3",
                "style": "structured_card",
                "primary": "下单侧能力审核卡（CPL）｜部件：{part}｜前提：稳定近似正态｜均值：{mean}｜下规格限：{lo}｜标准差：{sd}｜门槛：{target}｜申报结论：CPL达标。请判断。",
                "paraphrase": "{part}下单侧能力审核卡（CPL）的申报结论是“CPL达标”（门槛{target}）。记录显示过程稳定近似正态，标准差{sd}、均值{mean}、下规格限{lo}。请审核。",
            },
        ],
        "cpk": [
            {
                "template_id": "test_cpk_t0",
                "style": "receiving_ticket",
                "primary": "{part}核心配合尺寸工序稳定且近似正态。实测均值{mean}、标准差{sd}，规格限为{lo}至{hi}，合同要求Cpk门槛为{target}。试制总结写明“CPK达到合同门槛{target}”。请判断。",
                "paraphrase": "试制总结对{part}写明“CPK达到合同门槛{target}”。已知该稳定近似正态工序的规格限为{lo}至{hi}，实测均值为{mean}，标准差为{sd}。请复核此结论。",
            },
            {
                "template_id": "test_cpk_t1",
                "style": "claim_first_audit",
                "primary": "待审核声明：“{part}的过程能力指数CPK达到门槛{target}”。统计记录：过程稳定且近似正态，均值为{mean}，标准差为{sd}，下规格限{lo}，上规格限{hi}。请判断。",
                "paraphrase": "统计记录显示{part}加工过程稳定且近似正态，规格限{lo}至{hi}，均值{mean}，标准差{sd}。请判定“{part}的过程能力指数CPK达到门槛{target}”是否成立。",
            },
            {
                "template_id": "test_cpk_t2",
                "style": "shift_handover",
                "primary": "产线验收材料称，{part}工序稳定且近似正态，规格区间{lo}到{hi}，实测均值{mean}、标准差{sd}，由此得出“工序CPK满足准入门槛{target}”。请审核是否成立。",
                "paraphrase": "产线验收材料对{part}得出“工序CPK满足准入门槛{target}”的结论。已知该稳定近似正态工序标准差{sd}、均值{mean}，规格下限{lo}、上限{hi}。请判断。",
            },
            {
                "template_id": "test_cpk_t3",
                "style": "structured_card",
                "primary": "过程能力验收卡（CPK）｜产品：{part}｜分布：稳定近似正态｜下规格限：{lo}｜上规格限：{hi}｜均值：{mean}｜标准差：{sd}｜门槛：{target}｜结论：CPK达标。请判断。",
                "paraphrase": "{part}过程能力验收卡（CPK）的结论为“CPK达标”（要求门槛{target}）。卡面记录：稳定近似正态、均值{mean}、标准差{sd}、下规格限{lo}、上规格限{hi}。请审核。",
            },
        ],
    },
}
