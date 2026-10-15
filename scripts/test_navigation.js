const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = (name) => fs.readFileSync(path.join(__dirname, '..', 'assets', name), 'utf8');

function anchorFixture(tabindex) {
  let delegate;
  let prevented = false;
  const target = {
    tabindex,
    hasAttribute: () => target.tabindex !== undefined,
    setAttribute: (name, value) => { target.tabindex = value; },
    focus: () => { document.activeElement = target; }
  };
  const document = { title: 'Page', getElementById: (id) => id === 'main-content' ? target : null };
  const window = { document, location: { hash: '', pathname: '/events/' }, history: { pushState() {} } };
  const $ = () => ({
    ready: (fn) => fn(),
    on: (event, selector, fn) => { if (event === 'click') delegate = fn; },
    height: () => 90,
    offset: () => ({ top: 300 }),
    scrollTop() {}
  });
  $.proxy = (object, method) => object[method].bind(object);
  const context = vm.createContext({ window, $ });
  vm.runInContext(source('scroll-to-anchor.js'), context);
  context.scrollToAnchor.bind();
  return {
    target, document,
    click: (href) => {
      delegate({
        target: { getAttribute: () => null },
        currentTarget: { getAttribute: () => href },
        preventDefault: () => { prevented = true; }
      });
      return prevented;
    }
  };
}

test('an anchor click from a child element moves keyboard focus to the destination', () => {
  const fixture = anchorFixture();
  assert.equal(fixture.click('#main-content'), true);
  assert.equal(fixture.document.activeElement, fixture.target);
  assert.equal(fixture.target.tabindex, '-1');
});

test('anchor navigation preserves an existing tab position and leaves other links alone', () => {
  const fixture = anchorFixture('0');
  assert.equal(fixture.click('/projects/'), false);
  assert.equal(fixture.click('#missing'), false);
  assert.equal(fixture.document.activeElement, undefined);
  fixture.click('#main-content');
  assert.equal(fixture.target.tabindex, '0');
});

test('registration links expire in an already open page', () => {
  let now = Date.parse('2026-10-06T23:58:59+09:00');
  let tick;
  const link = {
    dataset: { registrationCloses: '2026-10-06T23:59:00+09:00' },
    textContent: '참가 신청하기 →',
    removeAttribute: () => { delete link.dataset.registrationCloses; }
  };
  vm.runInNewContext(source('community-content.js'), {
    document: { querySelectorAll: () => link.dataset.registrationCloses ? [link] : [] },
    Date: { now: () => now, parse: Date.parse },
    setInterval: (fn) => { tick = fn; }
  });
  assert.equal(link.textContent, '참가 신청하기 →');
  now += 1000;
  tick();
  assert.equal(link.textContent, '참가 안내 보기 →');
  assert.equal(link.dataset.registrationCloses, undefined);
});
