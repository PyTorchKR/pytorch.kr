// Load a player only when requested; no autoplay or YouTube API dependency.
document.querySelectorAll('.session-player').forEach(function (details) {
  details.addEventListener('toggle', function () {
    var container = details.querySelector('.video-container');
    if (!details.open) {
      // Removing a closed player also stops playback.
      container.replaceChildren();
      return;
    }
    var id = details.dataset.youtubeId;
    if (!/^[A-Za-z0-9_-]{11}$/.test(id)) return;
    var start = Math.max(0, parseInt(details.dataset.start, 10) || 0);
    var frame = document.createElement('iframe');
    frame.src = 'https://www.youtube-nocookie.com/embed/' + id + '?start=' + start;
    frame.title = container.dataset.videoTitle;
    frame.allow = 'encrypted-media; picture-in-picture; fullscreen';
    frame.allowFullscreen = true;
    frame.referrerPolicy = 'strict-origin-when-cross-origin';
    container.replaceChildren(frame);
  });
});
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
