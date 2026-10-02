# 프로젝트·모임·행사 운영 가이드

홈페이지는 Jekyll의 Markdown 컬렉션을 사용합니다. 새 행사를 추가할 때 HTML이나 메뉴를 수정할 필요가 없습니다. 이 가이드가 사람과 AI Agent의 공통 기준입니다.

## 어디를 수정하나요?

| 목적 | 파일 | 자동 반영되는 곳 |
|---|---|---|
| 공개 산출물 소개 | `_projects/<uid>.md` | 프로젝트 목록·상세 |
| 지속적인 모임·행사 시리즈 소개 | `_groups/<uid>.md` | 모임 목록·상세 |
| 날짜가 있는 개별 행사 | `_events/<uid>.md` | 행사 목록, 홈페이지, 연결된 모임·프로젝트 |
| 긴 회고·기술 글 | 기존 `_posts/` | 블로그; 행사에서 `related_links`로 연결 |

1. `docs/templates/`의 해당 파일을 복사하고 파일명과 `uid`를 같은 영문 소문자·숫자·하이픈으로 정합니다. 행사 파일명에 Jekyll 게시일 접두사를 쓰지 않습니다.
2. 실제 출처에 따라 제목·설명·일정·관계를 입력합니다. 예시의 `example.org` 주소와 예시 날짜를 그대로 발행하지 않습니다.
3. 출처를 확인한 날짜를 `last_verified_at`에 기록합니다. 기존 행사라면 새 파일을 만들지 않고 같은 파일을 갱신합니다.
4. 아래 검증과 빌드를 실행하고 해당 페이지와 연결된 목록을 확인합니다.
5. 공개 검토가 끝나면 `published: false`를 제거한 변경을 검토받습니다. 발행 후 URL과 `uid`는 유지합니다.

`published: false`는 사이트 출력을 막을 뿐입니다. 공개 저장소에 올린 원문까지 비공개가 되지는 않으므로 참가자 명단·연락처·비공개 자료·작업용 공유 링크를 넣지 않습니다.

## 필드와 관계

공통 필수값은 `uid`, `title`, `summary`, `sources`(공개 HTTPS URL 배열), `last_verified_at`입니다. 날짜는 반드시 문자열로 따옴표를 붙입니다. 프로젝트에는 `website_url` 또는 `repository_url`이 필요합니다. `order`는 목록 순서를 정합니다.

프로젝트의 `group_ids`는 산출물을 함께 만드는 모임을 가리킵니다. 행사의 `group_ids`·`project_ids`는 관련 모임·산출물을 가리키며, 다른 파일에 같은 회차 목록을 중복 기입하지 않습니다. 단발 협력 행사는 관계 없이 등록할 수 있습니다.

vLLM.KR은 본문에서 세 가지 활동을 소개합니다. 회차를 관리하는 `programs`는 `korea-meetup`, `community-meetup` 두 종류입니다. 연결된 행사에는 해당 `program`을 지정합니다. Hands-on은 소개만 유지하며 회차를 수집하거나 상세 기록을 만들지 않습니다.

| 행사 필드 | 규칙 |
|---|---|
| `event_date` | `'2026-10-14'`처럼 한국 기준 행사 날짜; Jekyll의 `date`를 사용하지 않음 |
| `starts_at`, `ends_at` | 확인된 경우만 `'2026-10-14T19:00:00+09:00'` 형식; 종료는 시작 이후 |
| `event_status` | `scheduled`, `held`, `cancelled`, `postponed` |
| `format` | `online`, `offline`, `hybrid` |
| `venue` | 확정된 장소; 미확정이면 생략 |
| `organizers`, `co_organizers`, `hosts`, `supporters` | 출처에서 확인한 역할별 이름 배열; 과거 회차에서 자동 복사하지 않음 |
| `registration` | URL·마감 시각 필수. 시작 시각·`status: closed` 또는 `not_open`·`selection: true` 선택 |
| `recording` | `not_recorded`(녹화 안 함) 또는 `limited`(음성 녹화 문제로 제공 제한); 다른 원인의 제한은 세션별 `video.note` 사용 |
| `related_links` | `title`, `url`, 선택적 `kind`; 공개 후기·재생목록·블로그 등 |

날짜가 지났다고 `held`로 바꾸지 않습니다. 개최 여부가 확인되면 변경합니다. 취소·연기한 행사는 삭제하지 않고 상태를 바꿉니다. 새 날짜가 확정되기 전에는 기존 날짜와 `postponed`를 유지하며, 행사 목록의 변경 안내에 표시됩니다.

다가오는 행사·지난 행사 구분은 빌드 시각 기준입니다. 현재 배포 워크플로는 매일 빌드합니다. 신청 버튼은 명시된 마감 이후 브라우저에서도 ‘참가 안내 보기’로 바뀌며, 실제 접수 여부는 모집 플랫폼을 따릅니다. 조기 마감은 `registration.status: closed`로 수정합니다. 발표자 모집은 일반 참가 신청과 혼동하지 않도록 본문 또는 이름이 분명한 관련 링크에 둡니다.

## 영상·발표자료를 추가하는 방법

하나의 `sessions` 항목에 같은 발표의 영상·자료·코드를 연결합니다. 영상과 자료 중 하나만 있어도 됩니다.

```yaml
sessions:
  - uid: example-talk
    title: 실제 발표 제목
    speakers:
      - name: 발표자 이름
        affiliation: 공개된 소속
    video:
      state: public
      youtube_id: aV2lNdf1UHc  # 형식 예시: 실제 영상 ID로 교체
      start_seconds: 0
    slides:
      state: public
      url: https://example.org/event/talk/slides-v1.pdf
      format: pdf
      size_bytes: 10485760
    code_url: https://github.com/example/project
```

`video.state`는 `public`, `pending`, `private`, `not_recorded`, `limited`, `unknown`을 지원합니다. `slides.state`는 `public`, `pending`, `private`, `unknown`을 지원합니다. `limited` 영상에는 `note`로 제한 사유를 설명할 수 있습니다. 미입력 또는 `unknown`은 버튼을 생략합니다. `pending`은 공개 예정이라는 확인을 받은 경우에만 사용합니다. 비공개 상태에는 URL·영상 ID 자체를 넣지 않습니다.

공개 영상은 YouTube 링크와 접을 수 있는 플레이어를 함께 제공합니다. 플레이어를 펼칠 때만 `youtube-nocookie.com` iframe을 만들며, 닫으면 재생을 멈춥니다. 자동 재생하지 않습니다. YouTube 자체의 지역·로그인·임베드 제한은 직접 링크로 확인할 수 있습니다. 재생목록은 `related_links`에 추가합니다. YouTube API 키는 필요하지 않습니다.

제5회 기술 세미나는 **음성이 녹음되지 않는 문제가 있어 영상 제공이 제한됨**을 `recording: limited`로 안내합니다. 녹화를 하지 않은 행사로 바꾸지 않습니다.

2026년 5월·8월 vLLM Community Meetup의 원본 사진·작업용 앨범을 공개 갤러리에 연결하지 않습니다. 해당 후기의 검토가 완료된 공개용 개별 일러스트만 사용할 수 있습니다. 이번 페이지에는 확정되지 않은 이미지·후기 주소를 넣지 않았습니다.

## 대용량 자료 저장

발표자료는 홈페이지 Git 이력에 넣거나 submodule로 받아 `_site`에 복사하지 않습니다. 초기 권장 방식은 조직 소유의 별도 자료 저장소에 **GitHub Release 첨부파일**로 공개본을 올리고, 확정된 HTTPS 주소만 `slides.url`에 기록하는 것입니다. 새 저장소·Release 생성이나 업로드는 이 홈페이지 구현에 포함되지 않습니다.

- 홈페이지: 메타데이터·본문·작은 최적화 이미지.
- GitHub Releases: 행사/세션별 공개 PDF·PPTX·ZIP. Git 트리에는 관리 규칙을 두고 바이너리는 첨부파일로 관리.
- YouTube: 공개 발표 영상. 원본 영상은 사이트에 저장하지 않음.
- 조직 보관 공간: 원본·비공개 자료·동의 기록. 공개 홈페이지에는 작업용 링크를 연결하지 않음.

PDF 자체 도메인 열람, 장기 보관 및 대규모 배포가 필요해지면 R2와 자료 전용 도메인을 검토합니다. 홈페이지 템플릿은 바꾸지 않고 URL을 교체할 수 있습니다. submodule은 사이트와 함께 빌드할 **소스**에 적합하며 대용량 발표자료 배포의 기본안으로 쓰지 않습니다. Git LFS 역시 별도 클라이언트·CI·비용 관리가 필요하므로 기본안에서 제외합니다. 서비스 제한과 비용 비교의 출처는 [개편 계획 9절](homepage-renewal-plan-2026-10-02.md#9-대용량-발표자료-보관-방안)에 있습니다.

자료 공개 시 발표자의 동의·라이선스·배포 범위를 확인합니다. 홈페이지 소스의 라이선스를 외부 발표자료에 자동 적용하지 않습니다. 공개본은 `slides-v1.pdf`, 수정본은 `slides-v2.pdf`처럼 별도 버전으로 보존합니다. 게시 후 자료의 공개 접근 여부·파일 크기·영상과의 짝을 확인합니다.

## 로컬 검증과 미리보기

처음 설치할 때 저장소의 Ruby·Node 버전에 맞춰 의존성을 설치합니다. 기존 서브모듈은 고정된 커밋으로 초기화하며 단순 콘텐츠 작업 중 `--remote`로 갱신하지 않습니다.

```sh
npm ci --ignore-scripts
BUNDLE_PATH=vendor/bundler bundle install
git submodule update --init --recursive
make include-yarn-deps
```

콘텐츠를 수정할 때마다 다음을 실행합니다.

```sh
ruby scripts/check_content.rb
ruby scripts/test_content.rb
BUNDLE_PATH=vendor/bundler bundle exec ruby scripts/test_content_render.rb
BUNDLE_PATH=vendor/bundler bundle exec jekyll build
python3 scripts/check_content_links.py
python3 -m http.server 4000 --bind 127.0.0.1 --directory _site
```

[로컬 미리보기](http://127.0.0.1:4000/)에서 변경 페이지와 상위 목록을 확인합니다. 데스크톱·390px 모바일에서 긴 제목, 메뉴, 신청 상태, 자료가 없는 발표, 영상 열기/닫기를 봅니다. `Tab`·`Escape`로 메뉴를 조작할 수 있어야 합니다. 외부 출처의 내용·영상 재생·공개 동의는 자동 검사만으로 보증되지 않습니다.

PR Preview와 운영 배포의 빌드 단계에서 동일한 콘텐츠·템플릿 검증을 실행합니다. 운영 빌드나 미리보기 배포는 기존 워크플로 조건을 따릅니다. 콘텐츠 추가를 위해 프레임워크를 교체하거나 별도 CMS를 설치하지 않습니다.

## AI Agent에 요청할 때

예: “공식 행사 URL과 공개된 후기 URL을 확인해 `_events/`에 회차를 추가해 주세요. 모임은 `vllm-kr`, 프로그램은 `community-meetup`입니다. 기존 회차와 중복되지 않게 하고, 자료 공개 상태를 추정하지 말고 확인된 링크만 사용하세요. 작성 가이드의 검증과 로컬 빌드를 실행해 주세요.”

Agent는 `AGENTS.md`에서 이 가이드·템플릿을 찾아 같은 파일을 수정합니다. 별도 스킬에 동일 규칙을 복제하지 않습니다. 여러 원문을 수집·정규화하는 작업이 반복될 때에만 이 가이드를 참조하는 얇은 스킬을 추가하면 됩니다.
