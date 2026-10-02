# PyTorchKR website

## Scope and language

- Keep repository work inside this project. Use `/tmp` for intermediate artifacts. Do not access unrelated files or follow project symlinks outside the project. Explicit user-provided source paths may be read for the requested task.
- Interact with the user in Korean. Use English for implementation comments and internal reasoning.
- Do not commit, push, publish content, upload source files or deploy without authorization. Local reversible edits and builds are part of implementation.

## Content maintenance

- Read `docs/content-guide.md` before changing projects, groups or events. It is the canonical contract; do not duplicate its rules in a separate skill.
- Copy the relevant `docs/templates/` file. Use stable filenames and matching `uid` values. Check for existing records before adding one.
- Verify public sources, dates, names, organizational roles and publication status. Record source URLs and verification date. Do not infer attendance, a completed event, permission to publish, or media availability.
- Keep private records, attendee lists, local paths and draft placeholders out of public content. `published: false` does not make a public Git repository private.
- Keep vLLM.KR Hands-on as introductory text only. Track event records for Korea Meetup and Community Meetup.
- Do not put large presentation binaries or video files in the website repository or add a media submodule. Link verified public assets as described in the guide.

## Implementation and verification

- State material assumptions and a brief plan before implementation. Prefer the smallest change that meets the request. Preserve unrelated edits.
- Reuse Jekyll layouts/includes, existing brand assets, fonts and SCSS. Preserve established blog, hub and learning URLs and the bilingual community disclaimer.
- Run `ruby scripts/check_content.rb`, `ruby scripts/test_content.rb`, `bundle exec ruby scripts/test_content_render.rb`, `bundle exec jekyll build` and `python3 scripts/check_content_links.py` (with the locally configured bundle path if needed).
- For layout/navigation changes, inspect actual desktop and mobile rendering and keyboard navigation. Check long Korean titles, missing media, future/cancelled/postponed events and registration expiry. Report exact checks and limitations.
- Do not refresh submodule revisions or upgrade dependencies as an incidental part of content work. Initialize pinned submodules only when needed for a build.
