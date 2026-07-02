#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""黑盒词典静态站点生成器。

把 01_方法词典/ 下的全部方法卡渲染成一个静态网站（Hero 首页、目录、
分类页、方法详情页），视觉沿用「黑盒词典」Claude Design 稿：纸白/墨黑
配色 + 朱砂强调、Noto Serif SC / Spectral / IBM Plex Mono 三字体、KaTeX
公式、Python/R 代码分栏、粘性目录、深色模式。输出到 docs/（供 GitHub Pages）。

用法: python3 scripts/build_site.py
依赖: markdown, pyyaml
"""
import os
import re
import glob
import json
import html
import shutil

import yaml
import markdown as md_lib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT = os.path.join(ROOT, "01_方法词典")
IMG_SRC = os.path.join(ROOT, "04_示例图像")
OUT = os.path.join(ROOT, "docs")

ACCENT = "#B23A2B"
STATUS_GLYPH = {"草稿": "○", "已建": "◑", "已校": "●"}

# 分类英文名与一句话定位（顺序按目录 NN）
CAT_META = {
    "研究设计与数据理解": ("Study Design & Data Understanding", "研究怎么设计、数据怎么认识：分布探索、样本量与数据质控。"),
    "描述统计与统计推断": ("Descriptive & Inferential Statistics", "均值、分布、区间估计与假设检验的地基。"),
    "回归与广义线性模型": ("Regression & Generalized Linear Models", "以线性预测子连接期望与协变量的通用框架。"),
    "生存分析与纵向数据": ("Survival Analysis & Longitudinal Data", "处理事件时间、删失与重复测量的方法族。"),
    "因果推断": ("Causal Inference", "减少混杂、尽量逼近因果解释。"),
    "正则化与变量选择": ("Regularization & Variable Selection", "筛选变量并控制模型复杂度。"),
    "树模型与集成学习": ("Tree Models & Ensemble Learning", "用树结构与集成方法建模。"),
    "支持向量机与核方法": ("SVM & Kernel Methods", "间隔最大化与核技巧。"),
    "聚类与无监督学习": ("Clustering & Unsupervised Learning", "发现潜在亚群与结构。"),
    "降维与表征学习": ("Dimensionality Reduction & Representation", "压缩维度并保留主要结构。"),
    "神经网络与深度学习": ("Neural Networks & Deep Learning", "建模复杂非线性映射。"),
    "时间序列与时序建模": ("Time Series & Temporal Modeling", "处理带顺序依赖的数据。"),
    "数据预处理与特征工程": ("Data Preprocessing & Feature Engineering", "把原始数据变成可建模输入。"),
    "高维数据与多组学": ("High-Dimensional Data & Multi-Omics", "小样本高维与组学建模（待补）。"),
    "模型评估与解释": ("Model Evaluation & Interpretation", "区分度、校准、临床效用与可解释性。"),
    "分类与判别方法": ("Classification & Discriminant Methods", "把个体分到预定义类别。"),
}

MD_EXTS = ["tables", "fenced_code", "sane_lists"]


# ----------------------------- 解析 -----------------------------

def parse_card(path):
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    m = re.match(r"---\n(.*?)\n---\n(.*)", txt, re.S)
    if not m:
        return None
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)
    base = os.path.basename(path)[:-3]
    prefix = re.match(r"^(\d+)(?:\.\d+)?-", base)
    no = int(prefix.group(1)) if prefix else 999
    fullname = re.sub(r"^\d+(\.\d+)?-", "", base)
    cat_dir = os.path.basename(os.path.dirname(path))
    cat = re.sub(r"^\d+_", "", cat_dir)
    cat_no = re.match(r"^(\d+)_", cat_dir).group(1)

    # 一句话定义：### 1.1 定义 后第一段
    definition = ""
    dm = re.search(r"###\s*1\.1[^\n]*\n+(.+?)(?:\n\s*\n|\n###|\Z)", body, re.S)
    if dm:
        definition = re.sub(r"\s+", " ", dm.group(1)).strip()
        definition = re.sub(r"[*`]", "", definition)

    return {
        "path": path,
        "no": no,
        "cat_no": cat_no,
        "fullname": fullname,
        "cat": cat,
        "title": str(fm.get("title", fullname)),
        "en": str(fm.get("english_name", "")),
        "slug": str(fm.get("slug", "")),
        "aliases": fm.get("aliases", []) or [],
        "subcategory": str(fm.get("subcategory", "")),
        "tags": fm.get("tags", []) or [],
        "status": str(fm.get("status", "")),
        "difficulty": str(fm.get("difficulty", "")),
        "definition": definition,
        "body": body,
    }


def load_all():
    cards = []
    for path in glob.glob(os.path.join(DICT, "*", "*.md")):
        c = parse_card(path)
        if c:
            cards.append(c)
    # 分类 -> 有序卡列表
    cats = {}
    for c in cards:
        cats.setdefault((c["cat_no"], c["cat"]), []).append(c)
    for k in cats:
        cats[k].sort(key=lambda x: x["no"])
    return cards, dict(sorted(cats.items()))


# --------------------------- Markdown ---------------------------

def md_to_html(text, index):
    """把一段 markdown 转 HTML；保护数学与双链，重写图片路径。"""
    store = {}
    counter = [0]

    def token():
        counter[0] += 1
        return f"ZZTOKEN{counter[0]}ZZ"

    def keep_display(m):
        k = token(); store[k] = m.group(0); return k

    def keep_inline(m):
        k = token(); store[k] = m.group(0); return k

    def keep_wiki(m):
        target = m.group(1).strip()
        disp = (m.group(2) or target).strip()
        rec = index.get(target)
        if rec:
            k = token()
            store[k] = (
                f'<a class="xlink" href="m-{rec["slug"]}.html" '
                f'data-cn="{html.escape(rec["title"])}" '
                f'data-en="{html.escape(rec["en"])}" '
                f'data-def="{html.escape(rec["definition"][:120])}">'
                f'{html.escape(disp)}</a>'
            )
            return k
        return html.escape(disp)

    text = re.sub(r"\$\$.+?\$\$", keep_display, text, flags=re.S)
    text = re.sub(r"\$[^\$\n]+?\$", keep_inline, text)
    text = re.sub(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]", keep_wiki, text)

    out = md_lib.markdown(text, extensions=MD_EXTS)

    # 图片路径重写到站点 img/
    out = out.replace("../../04_示例图像/", "img/")
    # 恢复 token
    for k, v in store.items():
        out = out.replace(k, v)
    return out


def split_sections(body):
    """返回 (intro_html_text, [(num, title, content)])。intro 为首个 ## 之前的引言块。"""
    body = re.sub(r"^#\s+.+?\n", "", body, count=1)  # 去掉 # 大标题
    marks = list(re.finditer(r"(?m)^##\s+(\d+)\.\s+(.+?)\s*$", body))
    intro = body[: marks[0].start()].strip() if marks else body.strip()
    secs = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        secs.append((m.group(1), m.group(2).strip(), body[start:end].strip()))
    return intro, secs


def render_impl(content, index, uid):
    """§5 实现：按 ### 5.x 标签拆分为 Python/R 代码分栏。"""
    parts = re.split(r"(?m)^###\s+(5\.\d+)\s+(.+?)\s*$", content)
    if len(parts) < 4:
        return md_to_html(content, index)
    pre = parts[0].strip()
    tabs = []
    it = parts[1:]
    for j in range(0, len(it), 3):
        label = it[j + 1].strip()
        tabbody = it[j + 2].strip()
        tabs.append((label, md_to_html(tabbody, index)))
    btns, panels = [], []
    for ti, (label, tab_html) in enumerate(tabs):
        active = " active" if ti == 0 else ""
        btns.append(
            f'<button class="codetab{active}" data-uid="{uid}" data-i="{ti}">{html.escape(label)}</button>'
        )
        panels.append(
            f'<div class="codepanel{active}" data-uid="{uid}" data-i="{ti}">{tab_html}</div>'
        )
    pre_html = md_to_html(pre, index) if pre else ""
    return (
        f'{pre_html}<div class="impl">'
        f'<div class="codebar">{"".join(btns)}</div>'
        f'{"".join(panels)}</div>'
    )


# ----------------------------- 页面 -----------------------------

def page(title, body_html, extra_head="", body_class="", scripts=""):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@300;400;500;600;700&family=Spectral:ital,wght@0,300;0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
{extra_head}
<script>(function(){{try{{if(localStorage.getItem('bb-dark')==='1')document.documentElement.classList.add('dark');}}catch(e){{}}}})();</script>
</head>
<body class="{body_class}">
{body_html}
<script src="assets/app.js"></script>
{scripts}
</body>
</html>
"""


def masthead(active=""):
    def cls(name):
        return ' class="nav-on"' if active == name else ""
    return f"""<header class="mast">
  <div class="wrap mast-in">
    <a class="brand" href="index.html">
      <span class="brand-cn">黑盒词典</span>
      <span class="brand-en">Blackbox&nbsp;Dictionary</span>
    </a>
    <nav class="mast-nav">
      <a href="catalog.html"{cls('catalog')}>目录</a>
      <a href="about.html"{cls('about')}>关于</a>
      <button id="darkToggle" class="darkbtn" type="button"><span class="dot"></span><span class="dlab">深色</span></button>
    </nav>
  </div>
</header>"""


def render_hero(total, ncats):
    body = f"""<div class="hero">
  <div class="hero-seam"></div>
  <svg class="hero-curve" viewBox="0 0 1200 160" preserveAspectRatio="none">
    <path d="M0,150 C220,150 300,30 460,26 C560,24 640,120 760,138 C880,150 980,64 1200,58" fill="none" stroke="#EDE6D6" stroke-width="1"></path>
    <path d="M0,150 C260,150 340,74 520,70 C660,66 720,130 880,142 C1000,150 1080,100 1200,96" fill="none" stroke="{ACCENT}" stroke-width="1"></path>
  </svg>
  <div class="wrap hero-top">
    <div class="brand"><span class="brand-cn light">黑盒词典</span><span class="brand-en">Blackbox Dictionary</span></div>
    <span class="hero-ed">2026 版 · {total} 张方法卡 · {ncats} 个分类</span>
  </div>
  <div class="wrap hero-main">
    <div class="hero-block">
      <div class="kicker">打开统计方法的黑箱</div>
      <h1 class="hero-h1">黑盒词典</h1>
      <p class="hero-sub">A working dictionary of statistical methods for medical research &amp; data science —<span class="hero-sub-cn">&nbsp;为医学研究与数据科学整理的方法字典。</span></p>
      <div class="preface">
        <div class="preface-bar"></div>
        <div class="preface-label">前言 · Preface</div>
        <p class="preface-text">每一种统计方法都是一只黑箱：数据从一端进入，结论从另一端走出，中间的机理常被交给软件与惯例。这部词典想做的，是把箱子打开——写清每种方法的数学、假设、代价与边界，让使用者既会用，也知道它何时会失效。</p>
        <div class="preface-sign">— 编者</div>
      </div>
      <div class="hero-cta">
        <a class="cta-main" href="catalog.html">进入词典 →</a>
        <a class="cta-sub" href="catalog.html">浏览目录</a>
      </div>
    </div>
  </div>
</div>"""
    return page("黑盒词典 · Blackbox Dictionary", body, body_class="hero-page")


def render_catalog(cats, search_index):
    rows = []
    for (cno, cname), items in cats.items():
        en, desc = CAT_META.get(cname, ("", ""))
        count = len(items)
        clickable = count > 0
        href = f'cat-{cno}.html' if clickable else ""
        tag = "a" if clickable else "div"
        attr = f' href="{href}"' if clickable else ""
        dim = "" if clickable else " dim"
        rows.append(f"""<{tag}{attr} class="cat-row{dim}">
      <span class="cat-no">{cno}</span>
      <div class="cat-mid">
        <span class="cat-cn">{html.escape(cname)}</span>
        <span class="cat-en">{html.escape(en)}</span>
        <span class="cat-desc">{html.escape(desc)}</span>
      </div>
      <span class="cat-count">{count}<span class="cat-count-u">&nbsp;个</span></span>
    </{tag}>""")
    body = f"""{masthead('catalog')}
<main class="wrap catalog">
  <div class="cat-head">
    <div class="cat-head-l">
      <div class="kicker dark-ink">目录 · Contents</div>
      <h2 class="page-h2">目录</h2>
      <p class="lede">共 {len(CAT_META)} 个分类，收录 {len(search_index)} 张方法卡。按分类浏览，或用中文名 / 英文名 / 标签搜索。</p>
    </div>
    <div class="cat-head-r">
      <div class="kicker dark-ink">搜索 · Search</div>
      <div class="searchbox">
        <span class="search-icon">⌕</span>
        <input id="searchInput" placeholder="Cox / 生存 / hazard / 标签…" autocomplete="off">
      </div>
      <div id="searchHint" class="search-hint">输入关键词搜索</div>
    </div>
  </div>
  <div id="searchResults" class="search-results"></div>
  <div id="radicalTable">
    <div class="cat-th"><span>序号</span><span>分类 · Category</span><span class="ra">数量</span></div>
    {''.join(rows)}
  </div>
</main>
<script>window.BB_INDEX = {json.dumps(search_index, ensure_ascii=False)};</script>"""
    return page("目录 · 黑盒词典", body, body_class="page")


def render_category(cno, cname, items):
    en, desc = CAT_META.get(cname, ("", ""))
    entries = []
    for c in items:
        glyph = STATUS_GLYPH.get(c["status"], "◑")
        entries.append(f"""<a class="entry" href="m-{c['slug']}.html">
      <span class="entry-no">{c['cat_no']}·{c['no']:02d}</span>
      <div class="entry-mid">
        <div class="entry-heads">
          <span class="entry-cn">{html.escape(c['title'])}</span>
          <span class="entry-en">{html.escape(c['en'])}</span>
        </div>
        <p class="entry-def">{html.escape(c['definition'][:90])}</p>
      </div>
      <div class="entry-meta">
        <span class="entry-status">{glyph}&nbsp;{html.escape(c['status'])}</span>
        <span class="entry-diff">{html.escape(c['difficulty'])}</span>
      </div>
    </a>""")
    body = f"""{masthead()}
<main class="wrap category">
  <div class="crumb"><a href="catalog.html">目录</a><span class="sep">/</span><span class="cur">{html.escape(cname)}</span></div>
  <div class="cat-title">
    <div class="cat-title-l">
      <div class="kicker dark-ink">第 {cno} 类</div>
      <h2 class="page-h2">{html.escape(cname)}</h2>
      <div class="cat-title-en">{html.escape(en)}</div>
      <p class="lede">{html.escape(desc)}</p>
    </div>
    <div class="cat-title-r">
      <div class="cat-big">{len(items)}</div>
      <div class="kicker">收录条目</div>
    </div>
  </div>
  <div class="entries">{''.join(entries)}</div>
</main>"""
    return page(f"{cname} · 黑盒词典", body, body_class="page")


def render_detail(c, cats, index):
    cat_key = (c["cat_no"], c["cat"])
    siblings = cats[cat_key]
    pos = [i for i, s in enumerate(siblings) if s["slug"] == c["slug"]][0]
    prev_c = siblings[pos - 1] if pos > 0 else None
    next_c = siblings[pos + 1] if pos < len(siblings) - 1 else None

    intro, secs = split_sections(c["body"])
    toc_items, sec_html = [], []
    for num, title, content in secs:
        toc_items.append(
            f'<a class="toc-item" href="#sec-{num}" data-sec="{num}">'
            f'<span class="toc-num">{num.zfill(2)}</span>'
            f'<span class="toc-title">{html.escape(title)}</span></a>'
        )
        if num == "5":
            inner = render_impl(content, index, c["slug"])
        else:
            inner = md_to_html(content, index)
        sec_html.append(f"""<section id="sec-{num}" data-sec="{num}" class="sec">
      <div class="sec-head"><span class="sec-no">§{num}</span><h2 class="sec-h2">{html.escape(title)}</h2></div>
      <div class="sec-body">{inner}</div>
    </section>""")

    glyph = STATUS_GLYPH.get(c["status"], "◑")
    tags = " / ".join(html.escape(t) for t in c["tags"])
    intro_html = f'<div class="intro-note">{md_to_html(intro, index)}</div>' if intro else ""

    pager = []
    if prev_c:
        pager.append(f'<a class="pg pg-prev" href="m-{prev_c["slug"]}.html"><div class="pg-lab">← 上一篇</div><div class="pg-t">{html.escape(prev_c["title"])}</div></a>')
    else:
        pager.append('<span></span>')
    if next_c:
        pager.append(f'<a class="pg pg-next" href="m-{next_c["slug"]}.html"><div class="pg-lab">下一篇 →</div><div class="pg-t">{html.escape(next_c["title"])}</div></a>')
    else:
        pager.append('<span></span>')

    katex = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>"""

    body = f"""{masthead()}
<main class="wrap detail">
  <aside class="toc-wrap">
    <div class="toc-sticky">
      <div class="toc-label">目录 · Contents</div>
      <nav id="toc">{''.join(toc_items)}</nav>
    </div>
  </aside>
  <article id="article">
    <div class="crumb"><a href="catalog.html">目录</a><span class="sep">/</span><a href="cat-{c['cat_no']}.html">{html.escape(c['cat'])}</a><span class="sep">/</span><span class="cur">{html.escape(c['title'])}</span></div>
    <div class="titleblock">
      <div class="kicker dark-ink">{html.escape(c['cat'])} · {c['cat_no']}·{c['no']:02d}</div>
      <h1 class="detail-h1">{html.escape(c['title'])}</h1>
      <div class="detail-en">{html.escape(c['en'])}</div>
      <div class="detail-meta">
        <span>状态&nbsp;·&nbsp;<b>{glyph} {html.escape(c['status'])}</b></span>
        <span>难度&nbsp;·&nbsp;<b>{html.escape(c['difficulty'])}</b></span>
        <span>标签&nbsp;·&nbsp;<b>{tags}</b></span>
      </div>
    </div>
    {intro_html}
    {''.join(sec_html)}
    <div class="pager">{''.join(pager)}</div>
    <div class="colophon">— 黑盒词典 · {c['cat_no']}·{c['no']:02d} —</div>
  </article>
</main>
<div id="hovercard" class="hovercard"></div>
<script>
document.addEventListener('DOMContentLoaded', function(){{
  if(window.renderMathInElement){{
    renderMathInElement(document.getElementById('article'), {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});
  }} else {{
    var t=setInterval(function(){{if(window.renderMathInElement){{clearInterval(t);renderMathInElement(document.getElementById('article'),{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});}}}},150);
  }}
}});
</script>"""
    return page(f"{c['title']} · 黑盒词典", body, extra_head=katex, body_class="page detail-page")


def render_about(total, cats):
    parts = []
    for (cno, cname), items in cats.items():
        parts.append(f'<li><span class="ab-no">{cno}</span> {html.escape(cname)} <span class="ab-c">{len(items)} 个</span></li>')
    body = f"""{masthead('about')}
<main class="wrap about">
  <div class="cat-head">
    <div>
      <div class="kicker dark-ink">关于 · Colophon</div>
      <h2 class="page-h2">关于本词典</h2>
    </div>
  </div>
  <div class="about-body">
    <p>黑盒词典是一个面向医学研究与数据科学的统计方法字典，收录 {total} 张方法卡，分属 {len(CAT_META)} 个分类。每张方法卡按同一套结构编写：方法概览、数学形式、数据形式与输入输出、适用场景、Python/R 实现、结果解释、推荐可视化、优势局限与常见坑、与相近方法的区别、医学应用、相关方法与参考资料。</p>
    <p>编写原则是「先说解决什么问题，再讲怎么推导」：公式配参数解释，代码给最小可运行版本，并为医学场景标注结局类型、缺失机制、删失与混杂等关键前提。方法卡之间以相关方法互链，构成一张可检索的方法网络。</p>
    <p>本站由知识库中的 Markdown 方法卡自动生成，源码见 <a href="https://github.com/fxt-gw-pb/BlackBoxDictionary" class="xlink">GitHub · BlackBoxDictionary</a>。</p>
    <div class="kicker dark-ink" style="margin-top:40px;">分类一览</div>
    <ul class="about-cats">{''.join(parts)}</ul>
  </div>
</main>"""
    return page("关于 · 黑盒词典", body, body_class="page")


# ------------------------------ CSS/JS ------------------------------

CSS = r""":root{
  --paper:#F5F1E8; --warm:#EEE8DA; --ink:#211C17; --body:#332C24;
  --faint:#8A8276; --rule:rgba(33,28,23,0.13); --accent:#B23A2B;
  --accent-faint:rgba(178,58,43,0.38); --code-bg:#EEE8DA; --card:#F5F1E8;
}
html.dark{
  --paper:#17130E; --warm:#1E1A13; --ink:#ECE6D8; --body:#D6CFBF;
  --faint:#948C7B; --rule:rgba(236,230,216,0.15); --accent:#D5715F;
  --accent-faint:rgba(213,113,95,0.4); --code-bg:#221D15; --card:#1B160F;
}
*{box-sizing:border-box;-webkit-font-smoothing:antialiased;}
html,body{margin:0;padding:0;}
body{font-family:'Spectral','Noto Serif SC',serif;background:var(--paper);color:var(--ink);}
::selection{background:var(--accent);color:var(--paper);}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-thumb{background:rgba(138,130,118,0.4);border:3px solid transparent;background-clip:padding-box;border-radius:10px;}
a{color:inherit;text-decoration:none;}
.wrap{max-width:1180px;margin:0 auto;padding:0 40px;}
.kicker{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.28em;text-transform:uppercase;color:var(--faint);}
.kicker.dark-ink{color:var(--accent);}
.page-h2{font-family:'Noto Serif SC',serif;font-weight:500;font-size:46px;line-height:1.1;margin:14px 0;}
.lede{font-family:'Spectral','Noto Serif SC',serif;font-weight:300;font-size:17px;line-height:1.65;color:var(--faint);max-width:520px;margin:0;}

/* masthead */
.mast{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:saturate(1.1) blur(6px);border-bottom:1px solid var(--rule);}
.mast-in{height:66px;display:flex;align-items:center;justify-content:space-between;}
.brand{display:flex;align-items:baseline;gap:12px;}
.brand-cn{font-family:'Noto Serif SC',serif;font-weight:600;font-size:19px;letter-spacing:0.04em;color:var(--ink);}
.brand-cn.light{color:#EDE6D6;}
.brand-en{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.28em;text-transform:uppercase;color:var(--faint);}
.mast-nav{display:flex;align-items:center;gap:28px;font-family:'IBM Plex Mono',monospace;font-size:11.5px;letter-spacing:0.16em;text-transform:uppercase;}
.mast-nav a{color:var(--faint);padding-bottom:2px;border-bottom:1px solid transparent;}
.mast-nav a:hover,.mast-nav a.nav-on{color:var(--ink);border-bottom-color:var(--accent);}
.darkbtn{background:none;border:none;cursor:pointer;display:inline-flex;align-items:center;gap:7px;font-family:'IBM Plex Mono',monospace;font-size:11.5px;letter-spacing:0.16em;text-transform:uppercase;color:var(--faint);}
.darkbtn .dot{width:9px;height:9px;border-radius:50%;background:transparent;border:1px solid var(--faint);display:inline-block;}
html.dark .darkbtn .dot{background:var(--accent);}

/* hero */
.hero-page{background:#141210;}
.hero{position:relative;min-height:100vh;background:#141210;color:#EDE6D6;overflow:hidden;}
.hero-seam{position:absolute;top:0;bottom:0;left:61%;width:2px;background:linear-gradient(to bottom,transparent,rgba(255,243,214,0.9) 40%,rgba(255,243,214,0.4) 62%,transparent);box-shadow:0 0 34px 8px rgba(255,240,206,0.28);animation:glow 7s ease-in-out infinite;}
@keyframes glow{0%,100%{opacity:.6;}50%{opacity:1;}}
.hero-curve{position:absolute;left:0;right:0;bottom:0;width:100%;height:150px;opacity:.2;}
.hero-top{display:flex;align-items:baseline;justify-content:space-between;padding-top:44px;position:relative;}
.hero-ed{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:0.22em;text-transform:uppercase;color:#8C8375;}
.hero-main{position:relative;min-height:calc(100vh - 120px);display:flex;align-items:center;}
.hero-block{max-width:640px;padding:72px 0;}
.hero-block .kicker{color:var(--accent);margin-bottom:26px;}
.hero-h1{font-family:'Noto Serif SC',serif;font-weight:500;font-size:74px;line-height:1.14;margin:0 0 22px;}
.hero-sub{font-family:'Spectral',serif;font-weight:300;font-size:20px;line-height:1.5;color:#C9C1B0;max-width:480px;margin:0 0 52px;}
.hero-sub-cn{font-family:'Noto Serif SC',serif;color:#B7AE9C;}
.preface{position:relative;padding-left:26px;border-left:1px solid rgba(237,230,214,0.22);max-width:520px;}
.preface-bar{position:absolute;left:-1px;top:2px;width:1px;height:34px;background:var(--accent);}
.preface-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.3em;text-transform:uppercase;color:#8C8375;margin-bottom:14px;}
.preface-text{font-family:'Spectral','Noto Serif SC',serif;font-style:italic;font-weight:300;font-size:19px;line-height:1.85;color:#D8D0BF;margin:0;}
.preface-sign{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:0.14em;color:#8C8375;margin-top:16px;}
.hero-cta{margin-top:56px;display:flex;align-items:center;gap:28px;}
.cta-main{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:0.2em;text-transform:uppercase;color:#EDE6D6;border-bottom:1px solid var(--accent);padding-bottom:6px;}
.cta-sub{font-family:'Noto Serif SC',serif;font-size:14px;letter-spacing:0.1em;color:#8C8375;}

/* catalog */
.catalog{padding:72px 40px 120px;}
.cat-head{display:grid;grid-template-columns:1fr 1fr;gap:48px;align-items:end;padding-bottom:40px;border-bottom:1px solid var(--rule);}
.searchbox{display:flex;align-items:center;gap:12px;border-bottom:1.5px solid var(--ink);padding-bottom:10px;margin-top:10px;}
.search-icon{font-family:'IBM Plex Mono',monospace;color:var(--accent);font-size:15px;}
#searchInput{flex:1;border:none;outline:none;background:transparent;font-family:'Spectral','Noto Serif SC',serif;font-size:18px;color:var(--ink);}
.search-hint{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--faint);margin-top:9px;}
.search-results{margin-top:28px;}
.sr{display:grid;grid-template-columns:64px 1fr auto;gap:20px;align-items:baseline;padding:14px 12px;border-bottom:1px solid var(--rule);cursor:pointer;}
.sr:hover{background:var(--accent-faint);}
.sr-no{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);}
.sr-cn{font-family:'Noto Serif SC',serif;font-size:18px;}
.sr-en{font-family:'Spectral',serif;font-style:italic;font-size:14px;color:var(--faint);margin-left:8px;}
.sr-cat{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--faint);}
.cat-th{display:grid;grid-template-columns:64px 1fr 92px;gap:24px;padding:44px 12px 10px;border-bottom:1px solid var(--rule);font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.2em;text-transform:uppercase;color:var(--faint);}
.cat-th .ra{text-align:right;}
.cat-row{display:grid;grid-template-columns:64px 1fr 92px;gap:24px;align-items:center;padding:20px 12px;border-bottom:1px solid var(--rule);transition:background .18s;}
.cat-row:hover{background:var(--accent-faint);}
.cat-row.dim{opacity:.5;}
.cat-no{font-family:'IBM Plex Mono',monospace;font-size:14px;color:var(--faint);}
.cat-mid{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;}
.cat-cn{font-family:'Noto Serif SC',serif;font-weight:500;font-size:22px;}
.cat-en{font-family:'Spectral',serif;font-style:italic;font-size:14px;color:var(--faint);}
.cat-desc{font-family:'Noto Serif SC',serif;font-weight:300;font-size:14px;color:var(--faint);flex-basis:100%;}
.cat-count{text-align:right;font-family:'IBM Plex Mono',monospace;font-size:15px;}
.cat-count-u{font-size:11px;color:var(--faint);}

/* category page */
.category{padding:56px 40px 120px;}
.crumb{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;color:var(--faint);margin-bottom:40px;}
.crumb a:hover{color:var(--ink);}
.crumb .sep{color:var(--faint);margin:0 10px;}
.crumb .cur{color:var(--ink);}
.cat-title{display:grid;grid-template-columns:1fr 300px;gap:60px;align-items:end;padding-bottom:34px;border-bottom:1.5px solid var(--ink);}
.cat-title-en{font-family:'Spectral',serif;font-style:italic;font-size:18px;color:var(--faint);margin:8px 0 18px;}
.cat-title-r{text-align:right;}
.cat-big{font-family:'IBM Plex Mono',monospace;font-size:52px;line-height:1;}
.entries{margin-top:8px;}
.entry{display:grid;grid-template-columns:80px 1fr 190px;gap:24px;align-items:baseline;padding:26px 12px;border-bottom:1px solid var(--rule);transition:background .18s;}
.entry:hover{background:var(--accent-faint);}
.entry-no{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--faint);padding-top:6px;}
.entry-heads{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:9px;}
.entry-cn{font-family:'Noto Serif SC',serif;font-weight:500;font-size:25px;}
.entry-en{font-family:'Spectral',serif;font-style:italic;font-size:15px;color:var(--faint);}
.entry-def{font-family:'Noto Serif SC',serif;font-weight:300;font-size:15px;line-height:1.6;color:var(--faint);margin:0;max-width:560px;}
.entry-meta{text-align:right;display:flex;flex-direction:column;gap:8px;align-items:flex-end;padding-top:4px;}
.entry-status{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--body);}
.entry-diff{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:var(--faint);border:1px solid var(--rule);padding:2px 8px;border-radius:2px;}

/* detail */
.detail{display:grid;grid-template-columns:214px 1fr;gap:64px;padding:0 40px;}
.toc-wrap{padding-top:56px;}
.toc-sticky{position:sticky;top:98px;}
.toc-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.24em;text-transform:uppercase;color:var(--faint);margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--rule);}
.toc-item{display:grid;grid-template-columns:24px 1fr;gap:10px;align-items:baseline;padding:7px 0;color:var(--faint);}
.toc-item .toc-num{font-family:'IBM Plex Mono',monospace;font-size:10px;}
.toc-item .toc-title{font-family:'Noto Serif SC',serif;font-size:13.5px;line-height:1.45;border-left:2px solid transparent;padding-left:10px;margin-left:-12px;}
.toc-item:hover{color:var(--ink);}
.toc-item.on{color:var(--ink);}
.toc-item.on .toc-title{border-left-color:var(--accent);}
article#article{padding:56px 0 130px;max-width:760px;min-width:0;}
.titleblock{border-bottom:1px solid var(--rule);padding-bottom:34px;margin-bottom:14px;}
.detail-h1{font-family:'Noto Serif SC',serif;font-weight:600;font-size:52px;line-height:1.16;margin:22px 0 12px;}
.detail-en{font-family:'Spectral',serif;font-style:italic;font-size:22px;color:var(--faint);margin-bottom:28px;}
.detail-meta{display:flex;flex-wrap:wrap;gap:10px 26px;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.06em;color:var(--faint);}
.detail-meta b{color:var(--ink);font-weight:500;}
.intro-note{border-left:2px solid var(--accent);background:var(--warm);padding:16px 22px;margin:28px 0 0;font-family:'Spectral','Noto Serif SC',serif;font-size:15px;line-height:1.7;color:var(--body);}
.intro-note p{margin:0;}
.sec{padding-top:52px;}
.sec-head{display:flex;align-items:baseline;gap:14px;margin-bottom:20px;}
.sec-no{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--accent);letter-spacing:0.1em;}
.sec-h2{font-family:'Noto Serif SC',serif;font-weight:600;font-size:26px;margin:0;}
.sec-body{font-family:'Spectral','Noto Serif SC',serif;font-size:16.5px;line-height:1.9;color:var(--body);}
.sec-body h3{font-family:'Noto Serif SC',serif;font-weight:600;font-size:17px;color:var(--ink);margin:26px 0 10px;letter-spacing:0.01em;}
.sec-body h4{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.16em;text-transform:uppercase;color:var(--faint);margin:18px 0 8px;font-weight:500;}
.sec-body p{margin:0 0 16px;}
.sec-body ul,.sec-body ol{margin:0 0 16px;padding-left:0;list-style:none;}
.sec-body li{position:relative;padding:6px 0 6px 24px;line-height:1.75;}
.sec-body ul li::before{content:'—';position:absolute;left:0;color:var(--accent);font-family:'IBM Plex Mono',monospace;}
.sec-body ol{counter-reset:li;}
.sec-body ol li{counter-increment:li;}
.sec-body ol li::before{content:counter(li);position:absolute;left:0;color:var(--accent);font-family:'IBM Plex Mono',monospace;font-size:12px;}
.sec-body strong{font-weight:600;color:var(--ink);}
.sec-body em{font-style:italic;}
.sec-body code{font-family:'IBM Plex Mono',monospace;font-size:0.86em;background:var(--warm);padding:1px 5px;border-radius:2px;}
.sec-body a.xlink{color:var(--accent);border-bottom:1px solid var(--accent-faint);cursor:pointer;}
.sec-body a.xlink:hover{background:var(--accent-faint);}
.sec-body table{width:100%;border-collapse:collapse;margin:8px 0 20px;font-size:14.5px;}
.sec-body thead tr{border-bottom:1.5px solid var(--ink);}
.sec-body th{text-align:left;padding:11px 8px;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:var(--faint);font-weight:400;}
.sec-body td{padding:12px 8px;border-bottom:1px solid var(--rule);}
.sec-body img{max-width:100%;height:auto;display:block;margin:18px 0;border:1px solid var(--rule);}
.sec-body blockquote{margin:0 0 16px;padding:12px 20px;border-left:2px solid var(--accent);background:var(--warm);color:var(--body);}
.katex{font-size:1.02em;}
.sec-body .katex-display{margin:0;padding:22px 26px;background:var(--warm);border-left:2px solid var(--accent);overflow-x:auto;}
.sec-body pre{margin:14px 0;padding:22px 24px;background:var(--code-bg);overflow-x:auto;border:1px solid var(--rule);}
.sec-body pre code{font-family:'IBM Plex Mono',monospace;font-size:13px;line-height:1.85;background:none;padding:0;color:var(--body);}
/* code tabs */
.impl{border:1px solid var(--rule);margin:14px 0;}
.codebar{display:flex;border-bottom:1px solid var(--rule);}
.codetab{background:none;border:none;cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:0.1em;padding:13px 20px;color:var(--faint);border-bottom:2px solid transparent;}
.codetab.active{color:var(--ink);border-bottom-color:var(--accent);}
.codepanel{display:none;padding:6px 22px 4px;}
.codepanel.active{display:block;}
.codepanel p{margin:12px 0 4px;font-size:14px;color:var(--faint);}
.codepanel pre{border:none;margin:8px 0 16px;}
.codepanel ul{margin-bottom:6px;}
.pager{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:64px;padding-top:28px;border-top:1px solid var(--rule);}
.pg-lab{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.2em;text-transform:uppercase;color:var(--faint);margin-bottom:8px;}
.pg-t{font-family:'Noto Serif SC',serif;font-size:17px;color:var(--ink);}
.pg-next{text-align:right;}
.colophon{text-align:center;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.2em;color:var(--faint);margin-top:50px;}
.hovercard{position:fixed;z-index:80;width:288px;background:var(--card);border:1px solid var(--accent);box-shadow:0 18px 46px rgba(20,18,16,0.28);padding:18px 20px;display:none;pointer-events:none;}
.hovercard .hc-cat{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:0.2em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;}
.hovercard .hc-cn{font-family:'Noto Serif SC',serif;font-weight:600;font-size:19px;margin-bottom:3px;}
.hovercard .hc-en{font-family:'Spectral',serif;font-style:italic;font-size:13px;color:var(--faint);margin-bottom:12px;}
.hovercard .hc-def{font-family:'Spectral','Noto Serif SC',serif;font-size:13.5px;line-height:1.65;color:var(--body);margin:0;}

/* about */
.about{padding:56px 40px 120px;}
.about-body{max-width:680px;margin-top:36px;font-family:'Spectral','Noto Serif SC',serif;font-size:17px;line-height:1.95;color:var(--body);}
.about-body p{margin:0 0 20px;}
.about-body a.xlink{color:var(--accent);border-bottom:1px solid var(--accent-faint);}
.about-cats{list-style:none;padding:0;margin:16px 0 0;columns:2;column-gap:48px;}
.about-cats li{padding:8px 0;border-bottom:1px solid var(--rule);font-size:15px;break-inside:avoid;}
.ab-no{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--faint);margin-right:10px;}
.ab-c{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);float:right;}

@media(max-width:900px){
  .wrap{padding:0 22px;}
  .detail{grid-template-columns:1fr;gap:0;}
  .toc-wrap{display:none;}
  .cat-head,.cat-title{grid-template-columns:1fr;gap:24px;}
  .hero-h1{font-size:52px;}
  .detail-h1{font-size:38px;}
  .entry{grid-template-columns:1fr;gap:8px;}
  .entry-meta{flex-direction:row;align-items:baseline;}
  .about-cats{columns:1;}
}
"""

JS = r"""// 深色模式
(function(){
  var btn=document.getElementById('darkToggle');
  function sync(){var on=document.documentElement.classList.contains('dark');if(btn){btn.querySelector('.dlab').textContent=on?'浅色':'深色';}}
  if(btn){btn.addEventListener('click',function(){document.documentElement.classList.toggle('dark');try{localStorage.setItem('bb-dark',document.documentElement.classList.contains('dark')?'1':'0');}catch(e){}sync();});}
  sync();
})();
// 代码分栏
document.addEventListener('click',function(e){
  var t=e.target.closest('.codetab');if(!t)return;
  var uid=t.getAttribute('data-uid'),i=t.getAttribute('data-i');
  document.querySelectorAll('.codetab[data-uid="'+uid+'"]').forEach(function(b){b.classList.toggle('active',b.getAttribute('data-i')===i);});
  document.querySelectorAll('.codepanel[data-uid="'+uid+'"]').forEach(function(p){p.classList.toggle('active',p.getAttribute('data-i')===i);});
});
// 目录滚动高亮
(function(){
  var secs=[].slice.call(document.querySelectorAll('[data-sec]'));if(!secs.length)return;
  var links={};document.querySelectorAll('.toc-item').forEach(function(a){links[a.getAttribute('data-sec')]=a;});
  var io=new IntersectionObserver(function(ents){ents.forEach(function(en){if(en.isIntersecting){var n=en.target.getAttribute('data-sec');Object.values(links).forEach(function(l){l.classList.remove('on');});if(links[n])links[n].classList.add('on');}});},{rootMargin:'-76px 0px -68% 0px',threshold:0});
  secs.forEach(function(s){io.observe(s);});
  document.querySelectorAll('.toc-item').forEach(function(a){a.addEventListener('click',function(e){e.preventDefault();var el=document.getElementById('sec-'+a.getAttribute('data-sec'));if(el)window.scrollTo({top:el.getBoundingClientRect().top+window.pageYOffset-78,behavior:'smooth'});});});
})();
// 内链悬停提要
(function(){
  var hc=document.getElementById('hovercard');if(!hc)return;
  document.addEventListener('mouseover',function(e){
    var a=e.target.closest('a.xlink');if(!a||!a.getAttribute('data-cn'))return;
    hc.innerHTML='<div class="hc-cat">相关方法</div><div class="hc-cn">'+a.dataset.cn+'</div><div class="hc-en">'+(a.dataset.en||'')+'</div><p class="hc-def">'+(a.dataset.def||'')+'</p>';
    var r=a.getBoundingClientRect();var x=r.left;if(x+300>window.innerWidth)x=window.innerWidth-300;
    hc.style.left=Math.max(12,x)+'px';hc.style.top=(r.bottom+8)+'px';hc.style.display='block';
  });
  document.addEventListener('mouseout',function(e){if(e.target.closest('a.xlink'))hc.style.display='none';});
})();
// 检索
(function(){
  var input=document.getElementById('searchInput');if(!input||!window.BB_INDEX)return;
  var results=document.getElementById('searchResults'),table=document.getElementById('radicalTable'),hint=document.getElementById('searchHint');
  input.addEventListener('input',function(){
    var q=input.value.trim().toLowerCase();
    if(!q){results.innerHTML='';table.style.display='';hint.textContent='输入关键词搜索';return;}
    table.style.display='none';
    var hits=window.BB_INDEX.filter(function(m){return (m.cn+' '+m.en+' '+m.cat+' '+m.tags).toLowerCase().indexOf(q)>=0;});
    hint.textContent='搜索结果 · '+hits.length;
    if(!hits.length){results.innerHTML='<div style="font-family:Noto Serif SC;font-size:15px;color:var(--faint);padding:12px;">没有找到匹配的条目。</div>';return;}
    results.innerHTML=hits.map(function(m){return '<a class="sr" href="'+m.url+'"><span class="sr-no">'+m.catNo+'</span><span><span class="sr-cn">'+m.cn+'</span><span class="sr-en">'+m.en+'</span></span><span class="sr-cat">'+m.cat+'</span></a>';}).join('');
  });
})();
"""


# ------------------------------ 主流程 ------------------------------

def main():
    cards, cats = load_all()
    index = {}
    for c in cards:
        index[c["fullname"]] = c
        for a in c["aliases"]:
            index.setdefault(str(a), c)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)

    # 资源
    with open(os.path.join(OUT, "assets", "style.css"), "w", encoding="utf-8") as f:
        f.write(CSS)
    with open(os.path.join(OUT, "assets", "app.js"), "w", encoding="utf-8") as f:
        f.write(JS)
    # 图片
    if os.path.isdir(IMG_SRC):
        for png in glob.glob(os.path.join(IMG_SRC, "*.png")):
            shutil.copy(png, os.path.join(OUT, "img", os.path.basename(png)))
    # GitHub Pages: 关闭 Jekyll（保留下划线目录/文件）
    open(os.path.join(OUT, ".nojekyll"), "w").close()

    # 检索索引
    search_index = []
    for (cno, cname), items in cats.items():
        for c in items:
            search_index.append({
                "cn": c["title"], "en": c["en"], "cat": cname, "catNo": f"{cno}·{c['no']:02d}",
                "tags": " ".join(str(t) for t in c["tags"]), "url": f"m-{c['slug']}.html",
            })

    def write(name, content):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(content)

    write("index.html", render_hero(len(cards), len(CAT_META)))
    write("catalog.html", render_catalog(cats, search_index))
    write("about.html", render_about(len(cards), cats))
    for (cno, cname), items in cats.items():
        if items:
            write(f"cat-{cno}.html", render_category(cno, cname, items))
    for c in cards:
        write(f"m-{c['slug']}.html", render_detail(c, cats, index))

    print(f"生成完成: {len(cards)} 张方法卡 -> {OUT}")
    print(f"页面: index, catalog, about, {sum(1 for v in cats.values() if v)} 分类页, {len(cards)} 详情页")


if __name__ == "__main__":
    main()
