# Preview navigation audit

The audit followed links on the PR 123 preview, starting from the groups page. It covered desktop and 390-pixel mobile layouts.

The existing group-to-project-to-event links worked. Browser Back also returned to the previous list. Three return paths needed improvement.

- CoreSIG opens from the full directory, but its project breadcrumb returned to the featured directory, which excludes CoreSIG.
- An event series opened from the full group directory offered no direct link back to that directory.
- A presentation opened at its session anchor, but that position offered no link back to the presentation list.

Activity detail breadcrumbs now link to the same card in the full directory. Event detail pages include the existing event navigation. Public sessions link back to the corresponding presentation item. Destination cards, sessions, and presentation items accept keyboard focus and clear the fixed header.

## Local browser results

The following paths passed after the changes:

- Projects → full directory → CoreSIG → full directory at the CoreSIG card.
- Presentations → PyTorch Foundation introduction → presentation list at the same item.
- Event detail → past events → conference → technical seminar series → full group directory at that series.
- PhysicalAI Study → Awesome Physical AI → related seminar → PhysicalAI Study, as tested on the original preview.

Enter activated the return links. The destination card and presentation item received focus and appeared approximately 115 pixels below the viewport top. The mobile pages had no horizontal overflow. The desktop event page retained its title, event navigation, and group relationship links.

## Automated results

- `ruby scripts/check_content.rb`: all 6 projects, 4 groups, and 13 events passed.
- `ruby scripts/test_content.rb`: 16 rejection cases and restored valid content passed.
- `bundle exec ruby scripts/test_content_render.rb`: passed, including full-directory return paths and public session return destinations.
- `node --test scripts/test_navigation.js`: all 3 tests passed.
- `bundle exec jekyll build`: passed with GitHub metadata authentication and rate-limit warnings.
- `python3 scripts/check_content_links.py`: 32 pages and 1,725 internal links and assets passed.
- `git diff --check`: passed.

Rendering fixtures cover future, cancelled, postponed, expired registration, unavailable media, and slides-only sessions. Two events reuse the same session UID to exercise distinct return anchors. Videos were not played. This audit evaluates navigation, not external media availability or publication permission.
