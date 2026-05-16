// Highlight the sidebar TOC entry whose section is currently in view.
// IntersectionObserver-based, no scroll listeners.

(function () {
  const sideToc = document.querySelector('.page-side-toc');
  if (!sideToc) return;

  const links = Array.from(sideToc.querySelectorAll('a[href^="#"]'));
  if (links.length === 0) return;

  // Map heading-id → toc link
  const linkByHash = new Map();
  for (const a of links) {
    linkByHash.set(a.getAttribute('href').slice(1), a);
  }

  const headings = Array.from(document.querySelectorAll('.content :is(h2, h3)[id]'));
  if (headings.length === 0) return;

  // Top-of-page threshold: highlight when a heading enters the top 30% of the viewport.
  const observer = new IntersectionObserver((entries) => {
    // Among all visible entries, pick the topmost one as active.
    const visible = entries.filter(e => e.isIntersecting);
    if (visible.length > 0) {
      const top = visible.sort((a, b) =>
        a.boundingClientRect.top - b.boundingClientRect.top
      )[0];
      setActive(top.target.id);
    }
  }, {
    rootMargin: '-15% 0px -70% 0px',
    threshold: 0,
  });

  for (const h of headings) observer.observe(h);

  function setActive(id) {
    for (const a of links) a.classList.remove('active');
    const a = linkByHash.get(id);
    if (a) a.classList.add('active');
  }

  // Initial state: first heading.
  if (headings.length) setActive(headings[0].id);
})();
