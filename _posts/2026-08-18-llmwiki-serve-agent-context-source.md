---
layout: blog_detail
title: "llmwiki-serve: Markdown/Obsidian 문서를 코딩 에이전트용 MCP context source로 쓰는 로컬 서버"
author: "Knowledge Bridge Labs"
category: ["MCP", "community", "opensource"]
date: 2026-08-18 12:00:00
---

안녕하세요. 코딩 에이전트가 내 Markdown 문서를 직접 찾아 읽게 해주는 `llmwiki-serve`를 preview로 공개했습니다. 현재 PyPI에 공개된 기준은 `llmwiki-serve==0.2.10`입니다. 이번 0.2.10에는 opt-in SQLite GraphStore가 들어갔습니다.

쓰기 어렵지 않게 만드는 것을 먼저 목표로 잡았습니다. Codex나 Claude Code를 쓰고 있다면 `llmwiki-bridge` marketplace plugin을 설치한 뒤 "이 wiki 폴더를 연결해줘"라고 시킬 수 있습니다. 플러그인이 `llmwiki-serve` 설치, 로컬 서버 실행, MCP 등록 과정을 안내합니다. 한 번 등록해두면 이후 세션에서도 같은 문서 폴더를 도구처럼 조회할 수 있습니다.

요즘 프로젝트를 하다 보면 README, ADR, spec, 회의 노트, Obsidian 문서가 계속 쌓입니다. 문제는 Codex나 Claude Code 같은 에이전트가 그 문서를 항상 잘 읽고 작업하지는 않는다는 점이었습니다.

매번 "이 파일도 봐줘", "저 폴더도 참고해줘"라고 붙여넣는 건 번거롭고, 그렇다고 이것 때문에 별도 RAG 앱을 만들거나 채팅 UI를 새로 붙이는 것도 과하다고 느꼈습니다.

`llmwiki-serve`는 이 사이를 작게 해결하려고 만든 도구입니다.

로컬 Markdown 폴더를 하나 지정하면, 그 폴더를 읽기 전용 context server로 열어줍니다. 에이전트는 여기에 붙어서 관련 문서를 찾고, 필요한 페이지를 읽고, 어떤 문서를 근거로 봤는지 같이 가져갈 수 있습니다.

예를 들면 이런 식입니다.

- "이번 릴리스에서 빠진 체크가 뭐야?"
- "이 설계 변경과 관련된 ADR을 찾아줘."
- "이 기능을 고치기 전에 봐야 할 spec이 있어?"
- "내 Obsidian vault에서 이 주제와 연결된 노트를 찾아줘."

이때 원본 Markdown 파일은 수정하지 않습니다. 서버는 문서를 옮기거나 변환해서 관리하지 않고, 현재 폴더를 읽어서 에이전트가 쓰기 좋은 조회 인터페이스만 제공합니다.

지원하는 것은 대략 이렇습니다.

- Markdown 폴더, Obsidian 스타일 wikilink, YAML front matter, heading, tag를 읽습니다.
- CLI, HTTP, MCP로 붙일 수 있습니다.
- 검색 결과와 함께 source ref, 관련 문서, graph context를 돌려줍니다.
- draft/private/confidential 문서는 기본 응답에서 제외합니다.
- 네트워크 응답에서는 로컬 루트 경로를 숨기고, 기본 CORS도 localhost 계열만 허용합니다.

코딩 에이전트에 붙이는 방식도 단순하게 잡았습니다.

가장 쉬운 경로는 marketplace plugin입니다. 이 플러그인은 별도 runtime이 아니라, Codex/Claude Code 안에서 `llmwiki-serve` 연결을 도와주는 setup/status/doctor skill 묶음입니다.

Codex에서는 이렇게 설치할 수 있습니다.

```bash
codex plugin marketplace add knowledge-bridge-labs/llmwiki-plugins --ref main
codex plugin add llmwiki-bridge@knowledge-bridge-labs
```

Claude Code에서는 이렇게 설치할 수 있습니다.

```bash
claude plugin marketplace add --scope user knowledge-bridge-labs/llmwiki-plugins
claude plugin install --scope user llmwiki-bridge@knowledge-bridge-labs
```

설치 후에는 새 세션을 열고 "Use the `llmwiki-bridge:setup` skill to connect ./wiki."처럼 요청하면 됩니다.

플러그인 없이 직접 붙인다면 기본 흐름은 설치, 서버 실행, MCP 등록입니다.

```bash
uv tool install llmwiki-serve
llmwiki-serve serve ./wiki --host 127.0.0.1 --port 8765
```

그다음 코딩 에이전트에는 `http://127.0.0.1:8765/mcp/stream`을 MCP server로 등록합니다. 이렇게 해두면 새 세션에서도 같은 문서 폴더를 `llmwiki_context`, `llmwiki_search`, `llmwiki_read`, `llmwiki_graph`, `llmwiki_graph_neighbors`, `llmwiki_source_refs`, `llmwiki_source_bundle` 도구로 조회할 수 있습니다.

예를 들면 Codex에서는 이렇게 등록할 수 있습니다.

```bash
codex mcp add llmwiki --url http://127.0.0.1:8765/mcp/stream
```

Claude Code에서는 사용자 설정으로 이렇게 등록할 수 있습니다.

```bash
claude mcp add -s user --transport http llmwiki http://127.0.0.1:8765/mcp/stream
```

MCP 등록은 남아 있지만, 실제 조회를 하려면 `llmwiki-serve serve ...` 프로세스는 켜져 있어야 합니다.

터미널에서 먼저 만져볼 때는 샘플 wiki로 바로 확인할 수 있습니다.

```bash
git clone https://github.com/knowledge-bridge-labs/llmwiki-serve.git
llmwiki-serve query ./llmwiki-serve/examples/sample-wiki "release readiness"
llmwiki-serve serve ./llmwiki-serve/examples/sample-wiki --host 127.0.0.1 --port 8765
```

기본 설치는 가볍게 동작합니다.

모델을 호출하지 않고, 답변을 합성하지 않고, 임베딩 인덱스도 만들지 않습니다. 기본 검색은 로컬 Markdown projection 위의 lexical ranking이고, 필요하면 exact substring 확인용 `literal` 모드를 선택할 수 있습니다.

의미 검색이 필요하면 별도 옵션으로 켤 수 있습니다.

```bash
uv tool install "llmwiki-serve[vector]"
llmwiki-serve serve ./wiki \
  --vector-provider fastembed \
  --vector-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --vector-model-download allow
```

이 경우 FastEmbed 기반 `vector`/`hybrid` 검색을 사용합니다. 별도 PyTorch serving을 띄우는 방식은 아니고, FastEmbed가 ONNX Runtime으로 같은 서버 프로세스 안에서 로컬 임베딩을 만듭니다. 기본은 이미 캐시된 모델만 쓰는 local-files-only 모드이고, 처음 실행에서 모델 다운로드가 필요하면 위 예시처럼 `--vector-model-download allow`를 명시합니다.

벡터 캐시는 서비스 대상 wiki root 밖의 로컬 sidecar에 저장됩니다. 설치만으로 자동 활성화되지는 않고, 서버를 띄울 때 운영자가 provider를 켜야 합니다.

오래 떠 있는 서버나 멀티 워커 환경에서는 Redis/Valkey projection cache도 붙일 수 있습니다.

```bash
uv tool install "llmwiki-serve[redis]"
llmwiki-serve serve ./wiki \
  --projection-store redis \
  --redis-url redis://127.0.0.1:6379/0 \
  --cache-namespace local \
  --source-id my-wiki
```

`0.2.10`에 추가된 SQLite GraphStore도 있습니다. 기본 설치에 포함되는 표준 `sqlite3` 기반 기능이라 별도 extra package는 필요 없고, 기본값은 꺼져 있습니다. 사용할 때는 `llmwiki-serve serve ./wiki --graph-store sqlite --graph-store-path ../.llmwiki-cache/wiki-graph.sqlite`처럼 명시하고, 경로는 반드시 source root 밖에 둬야 합니다. 이 파일은 `/graph`, `/graph/neighborhood`, MCP graph tool인 `llmwiki_graph`, `llmwiki_graph_neighbors`가 쓰는 derived graph snapshot cache이지 원본 문서나 source of truth가 아닙니다. 지워도 현재 projection에서 다시 만들 수 있지만, graph node label, path, relation, tag/source-ref metadata가 들어갈 수 있는 민감한 local derived state라 공개 repo나 동기화 폴더에 넣지 않는 쪽이 맞습니다.

벤치마크 숫자도 공개 repo에 같이 올려두었습니다.

`0.2.8` 기준으로 BEIR SciFact `test` split을 Markdown으로 projection해서 `LlmWikiService.search(query, limit=100)`로 돌린 리포트가 있습니다. 데이터 크기는 `5,183` docs, `300` queries, `339` qrels이고, opt-in `english` analyzer에서 nDCG@10 `0.6905`, Recall@100 `0.9287`이 나왔습니다. 같은 quality metric이 Windows와 Ubuntu 리포트에서 일치했고, warm fixed-index 검색 latency는 해당 환경 기준 Windows p50 `272.5ms` / p95 `535.2ms`, Ubuntu p50 `58.1ms` / p95 `118.2ms`였습니다.

이 수치는 공식 BEIR leaderboard 주장이 아니라, Markdown projection 위에서 같은 데이터를 재현한 공개 aggregate report입니다.

GitHub:
https://github.com/knowledge-bridge-labs/llmwiki-serve

Docs:
https://knowledge-bridge-labs.github.io/llmwiki-docs/

궁금한 피드백은 세 가지입니다.

- 실제 Markdown/Obsidian 폴더에서 잘 안 읽히는 패턴이 있는지
- MCP로 붙였을 때 tool 이름과 인자 형태가 쓰기 편한지
- "에이전트용 read-only 문서 source layer"라는 경계가 이해되는지

작은 프로젝트 문서 폴더나 개인 vault에 붙여보고 이상한 부분이 있으면 알려주세요. 필요한 기능, 깨지는 Markdown 패턴, 연결이 안 되는 코딩 에이전트가 있으면 GitHub Issues로 남겨주셔도 좋습니다.

Issues:
https://github.com/knowledge-bridge-labs/llmwiki-serve/issues
