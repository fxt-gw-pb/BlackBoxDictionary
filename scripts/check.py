#!/usr/bin/env python3
"""黑箱字典一致性检查器。

在仓库根目录运行: python3 scripts/check.py
检查项:
  1. 每张方法卡 frontmatter 必填字段齐全(title/english_name/slug/aliases/category/status)
  2. aliases 包含全名(文件名去掉 NN- 前缀); category 与所在目录一致
  3. slug 格式合法([a-z0-9-])且全库唯一; status 属于词表 {草稿, 已建, 已校}
  4. 全库双链可解析(指向未建卡片的链接列为提示,不算错误)
  5. 方法卡引用的图像存在; 04_示例图像 中无未被引用的孤儿图
  6. 索引表与卡片一一对应; 索引状态列与卡片 frontmatter status 一致;
     索引中无对应文件的行必须为「待写」

退出码: 有错误时为 1,仅有提示时为 0。
"""
import re
import sys
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_CARD = {"草稿", "已建", "已校"}
REQUIRED = ["title", "english_name", "slug", "aliases", "category", "status"]

errors, notes = [], []


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def parse_frontmatter(txt):
    m = re.match(r"---\n(.*?)\n---\n", txt, re.S)
    if not m:
        return None
    fm = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^(\w+):\s*(.*?)\s*(?:#.*)?$", line)
        if km:
            fm[km.group(1)] = km.group(2).strip()
    return fm


def parse_aliases(raw):
    inner = raw.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    quoted = re.findall(r'"([^"]+)"', inner)
    rest = re.sub(r'"[^"]+"', "", inner)
    return set(quoted) | {a.strip() for a in rest.split(",") if a.strip()}


cards = sorted(glob.glob(os.path.join(ROOT, "01_方法词典", "*", "*.md")))
targets, slugs, fullname2status = set(), {}, {}

for path in cards:
    rel = os.path.relpath(path, ROOT)
    base = os.path.basename(path)[:-3]
    fullname = re.sub(r"^\d+(\.\d+)?-", "", base)
    dirname = re.sub(r"^\d+_", "", os.path.basename(os.path.dirname(path)))
    txt = read(path)
    fm = parse_frontmatter(txt)
    if fm is None:
        errors.append(f"{rel}: 缺少 YAML frontmatter")
        continue
    for key in REQUIRED:
        if key not in fm or not fm[key]:
            errors.append(f"{rel}: frontmatter 缺少必填字段 {key}")
    aliases = parse_aliases(fm.get("aliases", ""))
    if fullname not in aliases:
        errors.append(f"{rel}: aliases 未包含全名「{fullname}」")
    if fm.get("category") and fm["category"] != dirname:
        errors.append(f"{rel}: category「{fm['category']}」与目录「{dirname}」不一致")
    if fm.get("status") and fm["status"] not in STATUS_CARD:
        errors.append(f"{rel}: status「{fm['status']}」不在词表 {sorted(STATUS_CARD)}")
    slug = fm.get("slug", "")
    if slug:
        if not re.fullmatch(r"[a-z0-9-]+", slug):
            errors.append(f"{rel}: slug「{slug}」含非法字符")
        if slug in slugs:
            errors.append(f"{rel}: slug「{slug}」与 {slugs[slug]} 重复")
        slugs[slug] = rel
    if "related_methods" in fm:
        errors.append(f"{rel}: frontmatter 不应包含 related_methods(相关方法只维护在正文第 11 节)")
    targets.add(base)
    targets.add(fullname)
    targets.update(aliases)
    fullname2status[fullname] = fm.get("status", "")

# --- 双链解析 ---
index_path = os.path.join(ROOT, "03_索引", "01-方法总索引.md")
for path in cards + [index_path]:
    rel = os.path.relpath(path, ROOT)
    txt = read(path)
    txt = re.sub(r"```.*?```", "", txt, flags=re.S)
    txt = re.sub(r"`[^`]*`", "", txt)
    for link in re.findall(r"\[\[([^\]|\\]+?)(?:\\?\|[^\]]+)?\]\]", txt):
        if link.strip() not in targets:
            notes.append(f"{rel}: [[{link.strip()}]] 未解析(若为计划中卡片属正常)")

# --- 图像 ---
img_dir = os.path.join(ROOT, "04_示例图像")
existing = {f for f in os.listdir(img_dir) if f.lower().endswith(".png")}
referenced = set()
for path in cards:
    rel = os.path.relpath(path, ROOT)
    for img in re.findall(r"04_示例图像/([\w.-]+\.png)", read(path)):
        referenced.add(img)
        if img not in existing:
            errors.append(f"{rel}: 引用的图像不存在 04_示例图像/{img}")
for orphan in sorted(existing - referenced):
    errors.append(f"04_示例图像/{orphan}: 孤儿图,未被任何方法卡引用")

# --- 索引一致性 ---
seen_in_index = set()
for line in read(index_path).splitlines():
    # 先保护 wikilink 内的 \| 转义管道,再按表格分隔符切分
    raw = line.replace("\\|", "\x00")
    cells = [c.strip().replace("\x00", "\\|") for c in raw.split("|")]
    if len(cells) < 9 or "---" in cells[1] or cells[1] in ("", "方法"):
        continue
    m = re.match(r"\[\[(.+?)\\\|", cells[1])
    if not m:
        errors.append(f"索引: 方法列不是 [[全名\\|短名]] 链接 -> {cells[1][:40]}")
        continue
    fullname, status = m.group(1), cells[7]
    seen_in_index.add(fullname)
    if fullname in fullname2status:
        if status != fullname2status[fullname]:
            errors.append(
                f"索引:「{fullname}」状态「{status}」与卡片 frontmatter「{fullname2status[fullname]}」不一致"
            )
    elif status != "待写":
        errors.append(f"索引:「{fullname}」无对应卡片文件,状态却是「{status}」而非「待写」")
for fullname in sorted(set(fullname2status) - seen_in_index):
    errors.append(f"索引: 缺少已建卡片「{fullname}」的行")

# --- 汇总 ---
for n in notes:
    print(f"[提示] {n}")
for e in errors:
    print(f"[错误] {e}")
print(f"\n检查完毕: {len(cards)} 张卡, {len(errors)} 个错误, {len(notes)} 条提示")
sys.exit(1 if errors else 0)
