# Community homepage audit

Audit date: 2026-10-15, Korea time.
Branch: `feat/community-homepage`.
Baseline commit: `085121756e43f0d911f3b084b4e18b230b631452`.
PR: [#123](https://github.com/PyTorchKR/pytorch.kr/pull/123).

## Scope and acceptance criteria

The final agreement in [the renewal plan](homepage-renewal-plan-2026-10-02.md) controls this audit.
The homepage keeps its existing hero, blog, community, features, installation, ecosystem, and footer sections.
The new project and event sections follow the community feed.
The earlier proposal to replace the main hero does not describe the final agreement.

The audit covers content relationships, shared templates, navigation, the existing design system, local validation, and preview deployment.
Completion requires fresh desktop and mobile inspection, a pushed commit, a successful preview workflow, and inspection of the deployed result.
Local inspection and the first deployed browser inspection passed.
The deployment inspection identified one additional mobile contrast correction.

## Findings and local changes

### Keyboard navigation

The anchor handler prevented native navigation but moved only the scroll position.
The destination did not receive keyboard focus.
The handler also read the clicked child element instead of the delegated link.
An arrow or other element inside a link therefore skipped the custom navigation behavior.

The handler now uses the delegated link and focuses the destination after an intentional anchor navigation.
It adds `tabindex="-1"` only when the destination has no existing value.
Initial hash positioning does not move focus.
The new regression test fails against the baseline and passes against the modified code.

The focus behavior follows [MDN guidance on tabindex](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/tabindex).
Browser inspection confirms focus transfer for the skip link and event section links.
A screen reader audit was not part of this inspection.

### Footer contrast

The community link selector also changed links in the dark footer callouts.
The resulting foreground color had a contrast ratio of 3.21:1 against the background.
The selector now applies only inside the main content.
The shared footer uses a lighter orange from the existing palette, with a contrast ratio of 5.18:1.
The deployed mobile menu also exposed active links with a contrast ratio of 4.11:1.
Those 17-pixel links now use the same lighter orange, with a contrast ratio of 5.18:1.
Body action buttons retain white text and their existing orange background.

### Preview indexing

The shared metadata template emitted `index, follow` in preview builds.
Preview builds now emit `noindex, nofollow`.
Production builds retain their existing indexing directive and canonical URLs.
Tests render both environments and reject duplicate robots directives.

### Build and deployment workflow

The workflows installed dependencies with Yarn without a tracked Yarn lock file.
They also moved submodules to remote revisions during the build.
The hub preparation script performed another pull.
These operations allowed the deployed input to differ from the reviewed repository state.

Both workflows now use `npm ci --ignore-scripts` and the recorded submodule revisions.
The Makefile and hub preparation script no longer advance submodule revisions during a build.
The existing package versions and submodule pointers remain intact.

Manual preview deployment now requires an open PR from the same repository before checkout.
It checks out the inspected head commit.
Fork PRs retain build validation without deployment or comment writes.
Closed fork PRs are excluded from preview cleanup.

Preview comment updates now select the GitHub Actions bot comment and paginate the comment list.
This prevents a human comment containing “PR Preview” from becoming the update target.

The workflow writes the built commit to `preview-revision.txt`.
After deployment, it requests that file over HTTPS and compares it with the local build revision.
A mismatch fails the job before the success comment.
The ordinary PR build still uses the GitHub merge ref, so this file identifies the tested merge commit.

The manual target rules follow [GitHub's workflow event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).
The README now describes Pages artifacts instead of deployment to a `gh-pages` branch.
The content guide distinguishes daily production builds from previews refreshed on PR changes or manual runs.

## Evidence from this audit

| Requirement | Current evidence | Status |
|---|---|---|
| Six projects, four groups, thirteen events | Content validator and current collection files | Passed locally |
| Existing homepage order, three distinct projects, three newest events | Current homepage source and Liquid fixture tests | Passed locally |
| One event file updates related pages | Fixture event appears in its group and project, with reciprocal links | Passed locally |
| Future, cancelled, postponed, unpublished, and past events | Liquid fixture tests | Passed locally |
| Registration opening, early closure, deadline, and an already open page | Liquid fixtures and JavaScript timer regression | Passed locally |
| Missing or private media | Liquid fixtures omit unavailable players and buttons | Passed locally |
| Public video embeds | Generated HTML contains 31 titled, lazy YouTube embeds without autoplay | Passed locally |
| Physical AI relationships | Study, technical seminar group, and project link to one event URL | Passed locally |
| vLLM activity split | Three descriptions and two archive sections, without a Hands-on archive | Passed locally |
| Blog, hub, and installation destinations | Generated destinations exist | Passed locally |
| Shared photos, fonts, layout, colors, and bilingual footer | Desktop and mobile screenshots, computed styles | Passed |
| New menu entries and unpublished exclusion | Liquid fixtures and actual desktop/mobile menus | Passed |
| Long Korean titles and responsive layouts | Ten routes at 320, 390, 768, 1100, and 1440 CSS pixels | Passed, no horizontal overflow |
| Keyboard navigation | Skip link, section links, menu arrows, Escape, Tab wrapping, resize cleanup | Passed |
| Representative video playback | End To End speech recognition, 2022 seminar, reached 3:08 of 34:55 | Passed |
| Revised workflow | YAML, shell, embedded JavaScript, and target validation cases | Local review passed |
| Commit, push, preview deployment, and live revision match | Run 38697768760 and live browser inspection | Passed for first deployment |

The browser inspection covered the homepage, collection indexes, vLLM group, Physical AI group, project, and event, and representative archived events.
All 50 route and viewport combinations had one main heading and no horizontal overflow.
The inspection also confirmed the CoreSIG redirect and the bilingual footer.
The mobile menu trapped focus, closed on Escape, and released the background when the viewport changed to desktop.

Isolated fixtures covered cancelled, postponed, future, private-media, missing-media, early-closed, and not-yet-open event states at 320 CSS pixels.
The fixtures used the real rendered templates and stylesheet. They were not added to public content.
Public pages retained their original long Korean titles and shared photo headers.

One representative video played inside the page. The audit did not play all 31 videos in full.
Some lazy players remained unloaded during automated inspection. Each public video also has a direct YouTube link.
The community feed showed its fallback link when the preview origin could not load the external feed.

Commands completed successfully:

```sh
ruby scripts/check_content.rb
ruby scripts/test_content.rb
BUNDLE_PATH=vendor/bundler BUNDLE_USER_HOME=/tmp/pytorchkr-bundle bundle exec ruby scripts/test_content_render.rb
node --test scripts/test_navigation.js
BUNDLE_PATH=vendor/bundler BUNDLE_USER_HOME=/tmp/pytorchkr-bundle JEKYLL_ENV=preview bundle exec jekyll build
BUNDLE_PATH=vendor/bundler BUNDLE_USER_HOME=/tmp/pytorchkr-bundle JEKYLL_ENV=production bundle exec jekyll build
python3 scripts/check_content_links.py
git diff --check
```

The link checker inspected 28 generated pages and 1,912 local links or asset references.
The three JavaScript tests passed.
The workflow review parsed both YAML files, compiled five JavaScript blocks, and parsed the shell blocks.
Four manual target cases covered an open local PR, a closed PR, a fork, and a deleted head repository.

An isolated clean dependency installation passed with npm 10.9.2.
The host's npm 11.20.0 rejected the existing lock metadata with `Missing: pytorch.kr@ from lock file`.
The audit did not upgrade dependencies or regenerate the lock file.

Local builds reported missing GitHub metadata authentication.
One baseline build also reported an API rate limit.
The builds succeeded, but contributor metadata completeness is not proven.

## Deployment evidence

Commit `4a98b80fb26fdc87501ae16c3305c117a849dc6d` passed [preview run 38697768760](https://github.com/PyTorchKR/pytorch.kr/actions/runs/38697768760).
The public revision file matched the tested merge commit `5d8accd0db0f541637f14a9de3624d606c2b0d0d`.
The workflow passed dependency installation, content tests, site generation, local link inspection, deployment, and the public revision comparison.

The live browser inspected ten routes at 320 and 1440 CSS pixels after page load.
All 20 combinations had one main heading, no horizontal overflow, and `noindex, nofollow` metadata.
The skip link focused the main content. The mobile menu restored focus and removed background isolation after Escape.
The Physical AI event displayed “지난 일정” and “참가 안내 보기”.
The deployed site retained the PyTorch release update from the base branch through the PR merge build.

A second commit applies the mobile contrast correction found during the deployed inspection.
The final task result records the follow-up workflow and public revision comparison.
The production workflow is reviewed and tested locally, but this audit deploys only the PR preview.

Unpublished slides, private photos, draft recaps, and proposed storage services remain outside the confirmed public content.
The audit does not infer that an event occurred merely because its scheduled date passed.
