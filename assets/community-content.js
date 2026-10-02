function refreshRegistrationLinks() {
  document.querySelectorAll('[data-registration-closes]').forEach(function (link) {
    if (Date.now() >= Date.parse(link.dataset.registrationCloses)) {
      link.textContent = '참가 안내 보기 →';
      link.removeAttribute('data-registration-closes');
    }
  });
}
refreshRegistrationLinks();
setInterval(refreshRegistrationLinks, 60000);
