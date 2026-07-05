# -*- coding: utf-8 -*-
"""eval —— 离线召回评测脚手架（13-1 / 13-2 章）

评测是离线工具，不属于运行时应用，因此放在 `scripts/eval/` 而非 `app/`：

    metrics.py               Recall@K / MRR / NDCG@K 纯函数 + 聚合 + 发版门禁
    validate_datasets.py     标注集自检（拿去评测前先跑这个）
    run_product_recall.py    商品检索（product_search）召回评测
    run_category_recall.py   品类知识库（CategoryInsight）召回评测
    rubric_contract.py       端到端 Rubric 的证据与 P0/P1/P2 离线契约
    rubric_cases.py          加载版本化 Rubric 回归用例
    rubric_ground_truth.py   从当前 ProductRepository 生成有界商品事实窗口
    rubric_evidence.py       将本地 EventBus/OTel 轨迹归一为脱敏评测证据
    rubric_judge.py          构造并校验证据约束的 Judge 协议
    rubric_report.py         输出质量总分、P0 门禁、场景覆盖与 FAIL/ERROR
    rubric_runner.py         真实编排器、事件、Trace、事实与 Judge 运行入口

两个跑测脚本共用 metrics，直连 UseCase / KnowledgeBase，不过 HTTP、不过 Agent——
召回评测的定位是模块级「日常体检」，必须快且便宜，才可能常驻 CI。
端到端质量由 `python -m scripts.eval.rubric_runner` 的 Rubric v3 评测负责；
根目录旧 `scripts/eval_regression.py` 仅保留历史实现对照。
"""
