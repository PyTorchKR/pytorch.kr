# Activity navigation verification

The global menu links directly to projects, groups, and events. Featured entries provide a short introduction. Full directories preserve all public records and URLs. Featured status is an editorial choice, not proof of current activity.

Group records distinguish communities from event series. Existing relationship IDs supply links in both directions. Cards and detail pages expose these links and event counts. Event ownership still comes from the explicit organizer fields.

The event area separates upcoming dates, public presentations, and past records. The homepage lists upcoming events in date order. Cancelled and postponed events remain in the change notice section. Public presentation links open the corresponding session in its event page.

## Automated results

- `ruby scripts/check_content.rb`: 6 projects, 4 groups, and 13 events passed.
- `ruby scripts/test_content.rb`: 16 invalid input cases and restored valid content passed.
- `bundle exec ruby scripts/test_content_render.rb`: directory selection, hidden content, event series, relationships, public media, registration states, and preview URL prefixes passed.
- `node --test scripts/test_navigation.js`: all 3 tests passed, including registration expiry in an open page.
- `bundle exec jekyll build`: passed with the existing GitHub metadata authentication warning.
- `python3 scripts/check_content_links.py`: 32 pages and 1,645 internal destinations passed.
- `git diff --check`: passed.

## Browser results

The local site used desktop widths of 1,440 and 1,100 pixels and a mobile width of 390 pixels. The desktop menu fits without overlapping the logo. Keyboard testing covered dropdown opening, Escape, mobile menu opening, and focus return. Project and group links work in both directions.

The project directories, group directory, related detail pages, event archive, and public presentations fit the mobile viewport. Korean titles wrap within the page. The mobile presentation tabs retain visible gaps and an active indicator. Existing global navigation styles initially overrode the tabs. A scoped rule resolves that conflict.

Future, cancelled, postponed, expired registration, unavailable media, and hidden content cases use isolated rendering fixtures. Browser testing covers existing public pages. External videos were not played, and publication permission was not reassessed. No dependency or submodule revision changed.

Deployment targets the existing PR 123 preview. Production merge and deployment are outside the requested scope.
