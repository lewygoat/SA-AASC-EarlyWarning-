# 上传到 GitHub 的步骤

## 1. 在 GitHub 建仓库

登录后点 New repository，名称建议 `SA-AASC-EarlyWarning-`，可见性按论文投稿需要选择。
**不要**勾选 Add a README、.gitignore 或 license，本项目已自带，避免冲突。

## 2. 上传

网页方式最省事。进入仓库页面点 Add file → Upload files，把本目录下全部内容拖进去，
注意不要拖成一个大文件夹，应当是 `src/`、`configs/`、`results/`、`figures/`、
`data/`、`docs/` 六个目录加上 `README.md`、`requirements.txt`、`LICENSE`、
`CITATION.cff`、`.gitignore` 五个文件位于仓库根目录。

命令行方式在仓库根目录执行

```bash
git init
git add .
git commit -m "Initial commit: MA-TempNet+GBDT and SA-AASC reference implementation"
git branch -M main
git remote add origin https://github.com/<用户名>/SA-AASC-EarlyWarning-.git
git push -u origin main
```

## 3. 上传前必须替换的占位符

`CITATION.cff` 里的 `repository-code` 一项现在写的是
`https://github.com/lewygoat/SA-AASC-EarlyWarning-`，仓库地址已指向 `lewygoat`。README 末段若你的仓库地址与之不同也一并改。

## 4. 数据不入库

`data/` 目录只放说明文件，三个数据集按 `data/README.md` 自行下载。`.gitignore` 已排除
`data/*.csv`、`data/oulad/*.csv` 与 `data/*.zip`，即使本地放入数据也不会被提交。

OULAD 单个 `final.csv` 有 20 MB，超过 GitHub 建议的仓库体积，放进仓库会让克隆变慢，
这也是保持数据外置的原因。

## 5. 上传后自检

克隆到本地空目录跑一次冒烟测试，确认他人能复现。

```bash
git clone https://github.com/<用户名>/SA-AASC-EarlyWarning-.git
cd SA-AASC-EarlyWarning
pip install -r requirements.txt
python src/run_experiments.py --mode smoke
```

冒烟模式在子采样数据上运行，几分钟内可看到数据加载、模型训练与结果写出的完整链路。

## 6. 与论文关联

投稿时在 Data Availability 一节填入仓库地址。若期刊要求匿名评审，先把仓库设为
Private，接收后再转 Public。仓库的 `CITATION.cff` 已写好 BibTeX 元数据，
GitHub 会在仓库侧栏显示 Cite this repository 按钮。
