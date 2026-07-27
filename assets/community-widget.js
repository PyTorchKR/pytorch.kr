/**
 * PyTorch.KR Community Widget
 * Fetches latest topics from discuss.pytorch.kr and renders them on the homepage.
 * Requires CORS to be enabled on Discourse:
 *   Admin → Settings → Security → cors origins → https://pytorch.kr
 */
(function () {
  'use strict';

  var DISCOURSE_URL = 'https://discuss.pytorch.kr';
  var API_URL = DISCOURSE_URL + '/latest.json';
  var CATEGORIES_URL = DISCOURSE_URL + '/categories.json';
  var TOPIC_COUNT = 12;
  var FETCH_TIMEOUT_MS = 8000;
  var CONTAINER_ID = 'community-topics-list';
  var DEFAULT_THUMBNAIL = '/assets/images/community-topic-placeholder.svg';

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function formatDate(isoString) {
    var d = new Date(isoString);
    var now = new Date();
    var diffMs = Math.max(0, now - d);
    var diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    var diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) return '방금 전';
    if (diffHours < 24) return diffHours + '시간 전';
    if (diffDays < 7) return diffDays + '일 전';

    var month = d.getMonth() + 1;
    var day = d.getDate();
    return month + '월 ' + day + '일';
  }

  function buildUserMap(users) {
    var map = {};
    if (!users) return map;
    for (var i = 0; i < users.length; i++) {
      map[users[i].id] = users[i];
    }
    return map;
  }

  function buildCategoryMap(categories) {
    var map = {};
    if (!categories) return map;
    for (var i = 0; i < categories.length; i++) {
      map[categories[i].id] = categories[i].name;
      var subs = categories[i].subcategory_list || [];
      for (var j = 0; j < subs.length; j++) {
        if (subs[j] && subs[j].id) map[subs[j].id] = subs[j].name;
      }
    }
    return map;
  }

  function getCategoryName(topic, categoryMap) {
    if (topic._category) return topic._category;
    return categoryMap[topic.category_id] || '';
  }

  function getAuthorUsername(topic, userMap) {
    if (topic._author) return topic._author;
    var posters = topic.posters || [];
    for (var i = 0; i < posters.length; i++) {
      var user = userMap[posters[i].user_id];
      if (user) return user.username;
    }
    return '';
  }

  function buildTopicCard(topic, userMap, categoryMap) {
    var url = DISCOURSE_URL + '/t/' + encodeURIComponent(topic.slug) + '/' + encodeURIComponent(topic.id);
    var rawImg = topic.image_url || '';
    if (rawImg && rawImg.charAt(0) === '/') rawImg = DISCOURSE_URL + rawImg;
    var imgSrc = rawImg ? escapeHtml(rawImg) : DEFAULT_THUMBNAIL;
    var author = escapeHtml(getAuthorUsername(topic, userMap));
    var category = escapeHtml(getCategoryName(topic, categoryMap || {}));

    return (
      '<div class="col-sm-6 col-lg-4 col-xl-3 mb-2">' +
        '<a href="' + url + '" target="_blank" rel="noopener ugc" class="community-topic-item">' +
          '<div class="topic-thumbnail">' +
            '<img src="' + imgSrc + '" alt="" loading="lazy" onerror="this.onerror=null;this.src=\'' + DEFAULT_THUMBNAIL + '\';">' +
          '</div>' +
          '<div class="topic-body">' +
            '<div class="topic-title">' + escapeHtml(topic.title) + '</div>' +
            '<div class="topic-meta">' +
              (category ? '<span class="topic-category">' + category + '</span>' : '') +
              (author ? '<span class="topic-author">' + author + '</span>' : '') +
              '<span class="topic-date">' + formatDate(topic.last_posted_at || topic.created_at) + '</span>' +
            '</div>' +
          '</div>' +
        '</a>' +
      '</div>'
    );
  }

  function renderTopics(topics, userMap, categoryMap) {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    var filtered = topics.filter(function (t) { return !t.pinned; });
    var display = filtered.slice(0, TOPIC_COUNT);

    if (display.length === 0) {
      renderFallback(container);
      return;
    }

    var html = '<div class="row g-2">' +
      display.map(function (t) { return buildTopicCard(t, userMap || {}, categoryMap || {}); }).join('') +
      '</div>';
    container.innerHTML = html;
  }

  function renderFallback(container) {
    container.innerHTML =
      '<div class="community-fallback">' +
        '<p>커뮤니티의 최신 글을 확인해보세요!</p>' +
        '<a href="' + DISCOURSE_URL + '" target="_blank" rel="noopener" class="btn btn-lg community-btn">커뮤니티 바로가기 →</a>' +
      '</div>';
  }

  function fetchWithTimeout(url, timeoutMs) {
    if (typeof AbortController === 'function') {
      var controller = new AbortController();
      var timer = setTimeout(function () { controller.abort(); }, timeoutMs);
      var p = fetch(url, { signal: controller.signal });
      return p.then(function (r) { clearTimeout(timer); return r; },
                    function (e) { clearTimeout(timer); throw e; });
    }
    return fetch(url);
  }

  var DEV_MOCK_TOPICS = [
    { id: 9110, title: 'Google, OpenClaw 및 OpenCode 등으로 인한 OAuth 약관 위배로 블럭된 Antigravity 계정에 복구 방법을 제공한다고 밝혀', slug: 'google-openclaw-opencode-oauth-antigravity', image_url: 'https://picsum.photos/seed/pyt1/200/200', last_posted_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(), _author: '9bow', _category: '정보공유' },
    { id: 9106, title: 'Anthropic, AI 기반 취약점 탐지 도구 Claude Code Security 발표 및 연구용 미리보기 사용 신청 시작', slug: 'anthropic-ai-claude-code-security', image_url: 'https://picsum.photos/seed/pyt2/200/200', last_posted_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(), _author: '9bow', _category: '정보공유' },
    { id: 9045, title: '저 github에 pose-transfer모델을 한번 올려봤어요. 괜찮을까요?', slug: 'github-pose-transfer', image_url: null, last_posted_at: new Date(Date.now() - 7 * 60 * 60 * 1000).toISOString(), _author: 'ryujanghyun00', _category: '자유게시판' },
    { id: 4833, title: '4070 Ti같은 12GB 메모리를 가진 GPU에서 작동하는 LLM이 있을까요?', slug: '4070-ti-12gb-gpu-llm', image_url: null, last_posted_at: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(), _author: '18NAO', _category: '질문답변' },
    { id: 9091, title: 'Codex 하네스의 비밀 풀기: App Server 구축기 (Unlocking the Codex harness: how we built the App Server)', slug: 'codex-app-server', image_url: 'https://picsum.photos/seed/pyt3/200/200', last_posted_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), _author: '9bow', _category: '정보공유' },
    { id: 9080, title: 'PyTorch 2.6 새로운 기능 살펴보기 - torch.compile 성능 개선 및 FlexAttention 안정화', slug: 'pytorch-2-6-new-features', image_url: null, last_posted_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), _author: 'pytorch_dev', _category: '공지사항' },
    { id: 9072, title: 'LoRA fine-tuning 시 메모리 부족 오류 해결 방법 공유합니다', slug: 'lora-fine-tuning-oom-fix', image_url: null, last_posted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), _author: 'ml_engineer', _category: '질문답변' },
    { id: 9060, title: 'vLLM vs TGI - 추론 서버 성능 비교 및 운영 경험 공유', slug: 'vllm-vs-tgi-comparison', image_url: 'https://picsum.photos/seed/pyt4/200/200', last_posted_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(), _author: 'infra_team', _category: '자유게시판' },
    { id: 9050, title: 'Hugging Face Transformers v5 주요 변경사항 정리', slug: 'huggingface-transformers-v5', image_url: null, last_posted_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), _author: 'hf_user', _category: '정보공유' },
    { id: 9040, title: 'CUDA 13.0 설치 시 드라이버 호환성 문제 해결 방법', slug: 'cuda-13-driver-compat', image_url: null, last_posted_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), _author: 'cuda_dev', _category: '질문답변' },
    { id: 9030, title: 'PyTorch Conference 2026 발표자 모집 안내', slug: 'pytorch-conf-2026-cfp', image_url: 'https://picsum.photos/seed/pyt5/200/200', last_posted_at: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(), _author: '9bow', _category: '공지사항' },
    { id: 9020, title: 'Distributed Training 시 NCCL 타임아웃 에러 디버깅 팁', slug: 'nccl-timeout-debug-tips', image_url: null, last_posted_at: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(), _author: 'dist_trainer', _category: '질문답변' }
  ];

  function isLocalhost() {
    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  }

  function init() {
    if (typeof fetch !== 'function') return;

    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    if (isLocalhost()) {
      renderTopics(DEV_MOCK_TOPICS, {}, {});
      return;
    }

    Promise.all([
      fetchWithTimeout(API_URL, FETCH_TIMEOUT_MS).then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      }),
      fetchWithTimeout(CATEGORIES_URL, FETCH_TIMEOUT_MS).then(function (res) {
        if (!res.ok) return { category_list: { categories: [] } };
        return res.json();
      }).catch(function () { return { category_list: { categories: [] } }; })
    ])
      .then(function (results) {
        var data = results[0];
        var catData = results[1];
        var topics = (data.topic_list && data.topic_list.topics) || [];
        var userMap = buildUserMap(data.users);
        var categoryMap = buildCategoryMap((catData.category_list && catData.category_list.categories) || []);
        renderTopics(topics, userMap, categoryMap);
      })
      .catch(function () {
        renderFallback(container);
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
