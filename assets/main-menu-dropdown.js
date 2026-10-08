document.querySelectorAll('[data-nav-toggle]').forEach(function (button) {
  var item = button.parentElement;
  var panel = document.getElementById(button.getAttribute('aria-controls'));
  var hovering = false;
  function setOpen(open) {
    button.setAttribute('aria-expanded', String(open));
    panel.hidden = !open;
  }
  // A click while the pointer already opened the menu keeps it open.
  button.addEventListener('click', function () { setOpen(hovering || panel.hidden); });
  // Keep the existing hover behavior for pointer devices.
  if (window.matchMedia('(hover: hover)').matches) {
    item.addEventListener('mouseenter', function () { hovering = true; setOpen(true); });
    item.addEventListener('mouseleave', function () { hovering = false; setOpen(false); });
  }
  item.addEventListener('focusout', function (event) { if (!item.contains(event.relatedTarget)) setOpen(false); });
  item.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { setOpen(false); button.focus(); }
    if (event.key === 'ArrowDown' && event.target === button) {
      event.preventDefault(); setOpen(true); panel.querySelector('a').focus();
    }
  });
  document.addEventListener('click', function (event) { if (!item.contains(event.target)) setOpen(false); });
});
