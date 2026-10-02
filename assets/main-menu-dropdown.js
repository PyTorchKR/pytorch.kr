document.querySelectorAll('[data-nav-toggle]').forEach(function (button) {
  var item = button.parentElement;
  var panel = document.getElementById(button.getAttribute('aria-controls'));
  function setOpen(open) {
    button.setAttribute('aria-expanded', String(open));
    panel.hidden = !open;
  }
  button.addEventListener('click', function () { setOpen(panel.hidden); });
  item.addEventListener('focusout', function (event) { if (!item.contains(event.relatedTarget)) setOpen(false); });
  item.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { setOpen(false); button.focus(); }
    if (event.key === 'ArrowDown' && event.target === button) {
      event.preventDefault(); setOpen(true); panel.querySelector('a').focus();
    }
  });
  document.addEventListener('click', function (event) { if (!item.contains(event.target)) setOpen(false); });
});
