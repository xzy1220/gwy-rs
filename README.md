# 公务员考试岗位个性化推荐系统

基于 Streamlit 的公务员岗位推荐系统，使用 CRITIC-TOPSIS 算法进行智能推荐。

## 功能特性

- 📊 多年份岗位数据（2018-2026）
- 🎯 智能推荐算法（CRITIC-TOPSIS）
- 🔍 多维度筛选（学历、政治面貌、工作地点等）
- 📥 结果导出功能

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run src/app.py
```

## 部署

### 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub
2. 访问 https://share.streamlit.io
3. 点击 "New app"
4. 选择仓库和分支，设置主文件为 `src/app.py`
5. 点击 "Deploy!"

## 项目结构

```
├── src/
│   ├── app.py              # 主应用
│   └── preprocess_data.py  # 数据预处理
├── data/
│   └── cache/              # 缓存数据
├── requirements.txt        # 依赖
└── README.md              # 说明文档
```
