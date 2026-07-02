// 深色模式
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
