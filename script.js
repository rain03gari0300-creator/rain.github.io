// ====== Utilidades ======
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

// ====== Menú responsive ======
(() => {
  const menu = $('.nav-menu');
  const btn = $('.hamburger');
  if (!menu || !btn) return;

  btn.addEventListener('click', () => menu.classList.toggle('open'));
  $$('.nav-link').forEach(a => a.addEventListener('click', () => menu.classList.remove('open')));
})();

// ====== Cursor personalizado ======
(() => {
  const cursor = $('.custom-cursor');
  if (!cursor) return;

  window.addEventListener('mousemove', e => {
    const x = e.clientX, y = e.clientY;
    cursor.style.transform = `translate(${x - 7}px, ${y - 7}px)`;
  });
})();

// ====== Efecto typing ======
(() => {
  const el = $('#typing-text');
  const cursor = $('.cursor');
  if (!el) return;

  const lines = [
    'Hola, soy Rain.',
    'Construyo sitios web y herramientas.',
    'Me enfoco en claridad y performance.'
  ];
  let li = 0, ci = 0, dir = 1;

  function tick() {
    const text = lines[li];
    ci += dir;
    el.textContent = text.slice(0, ci);

    if (ci === text.length + 10) dir = -1;
    if (ci === 0) { dir = 1; li = (li + 1) % lines.length; }
    setTimeout(tick, dir > 0 ? 70 : 35);
  }
  tick();

  if (cursor) cursor.style.opacity = '1';
})();

// ====== Tabs de proyectos ======
(() => {
  const buttons = $$('.tab-button');
  if (!buttons.length) return;

  const sections = {
    web: $('#web-content'),
    datos: $('#datos-content'),
    otros: $('#otros-content'),
  };

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      Object.values(sections).forEach(s => s?.classList.remove('active'));
      const key = btn.dataset.category;
      sections[key]?.classList.add('active');
    });
  });
})();

// ====== Modal de proyecto ======
(() => {
  const modal = $('#projectModal');
  if (!modal) return;
  const closeBtn = $('.close-modal', modal);
  const title = $('#modalTitle', modal);
  const tags = $('#modalTags', modal);
  const desc = $('#modalDescription', modal);
  const tech = $('#modalTechnologies', modal);
  const link = $('#modalGithubLink', modal);

  function open(card) {
    title.textContent = card.dataset.title || 'Proyecto';
    desc.textContent = card.dataset.description || '';
    tags.innerHTML = '';
    tech.innerHTML = '';
    (card.dataset.tags || '').split(',').map(s => s.trim()).filter(Boolean).forEach(t => {
      const span = document.createElement('span');
      span.className = 'tech-tag';
      span.textContent = t;
      tags.append(span.cloneNode(true));
      tech.append(span);
    });
    link.href = card.dataset.link || '#';

    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
  }

  function close() {
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
  }

  $$('.project-card').forEach(card => {
    card.style.cursor = 'pointer';
    card.addEventListener('click', (e) => {
      if (e.target.closest('.btn')) return; // no abrir si clic en un botón
      open(card);
    });
  });

  closeBtn?.addEventListener('click', close);
  modal.addEventListener('click', e => { if (e.target === modal) close(); });
  window.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
})();

// ====== Descargar CV (placeholder) ======
(() => {
  const btn = $('#downloadCV');
  if (!btn) return;
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    const blob = new Blob(
      ['Currículum - reemplaza este archivo por tu PDF real.'],
      { type: 'text/plain;charset=utf-8' }
    );
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'CV.txt';
    document.body.append(a);
    a.click();
    a.remove();
  });
})();
