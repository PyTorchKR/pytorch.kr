var mobileMenu = {
  bind: function () {
    var dialog = document.getElementById('mobile-navigation');
    var opener = document.querySelector('[data-behavior="open-mobile-menu"]');
    var closer = dialog.querySelector('[data-behavior="close-mobile-menu"]');
    var inertElements = [];
    function close() {
      dialog.classList.remove('open');
      dialog.hidden = true;
      document.body.classList.remove('no-scroll');
      opener.setAttribute('aria-expanded', 'false');
      inertElements.forEach(function (element) { element.inert = false; });
      inertElements = [];
      if (window.innerWidth >= 1100) document.querySelector('.header-holder .header-logo').focus();
      else opener.focus();
    }
    opener.addEventListener('click', function () {
      dialog.hidden = false;
      dialog.classList.add('open');
      document.body.classList.add('no-scroll');
      opener.setAttribute('aria-expanded', 'true');
      Array.from(document.body.children).forEach(function (element) {
        if (element !== dialog && !element.contains(dialog) && !element.inert && !['SCRIPT','STYLE'].includes(element.tagName)) {
          element.inert = true; inertElements.push(element);
        }
      });
      closer.focus();
    });
    closer.addEventListener('click', close);
    dialog.querySelectorAll('.mobile-nav-toggle').forEach(function (button) {
      button.addEventListener('click', function () {
        var panel = document.getElementById(button.getAttribute('aria-controls'));
        panel.hidden = !panel.hidden;
        button.setAttribute('aria-expanded', String(!panel.hidden));
      });
    });
    dialog.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') { close(); return; }
      if (event.key !== 'Tab') return;
      var items = Array.from(dialog.querySelectorAll('a, button')).filter(function (el) { return el.getClientRects().length > 0; });
      var first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    });
    window.addEventListener('resize', function () { if (window.innerWidth >= 1100 && !dialog.hidden) close(); });
  }
};
