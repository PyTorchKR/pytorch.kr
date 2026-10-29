#!/usr/bin/env ruby
# Exercise actual Liquid templates in a tiny isolated Jekyll fixture site.
require 'jekyll'
require 'tmpdir'
require 'fileutils'
require 'yaml'

root = File.expand_path('..', __dir__)
Dir.mktmpdir('pytorchkr-render-') do |dir|
  %w[_layouts _includes _events _groups _projects _data events projects groups].each { |name| FileUtils.mkdir_p(File.join(dir, name)) }
  %w[event activity].each { |name| FileUtils.cp(File.join(root, "_layouts/#{name}.html"), File.join(dir, '_layouts')) }
  %w[session content_links event_list event_status activity_card section_hero main_menu mobile_menu dropdown_links open_graph_and_meta activity_directory activity_relations event_navigation material_list].each { |name| FileUtils.cp(File.join(root, "_includes/#{name}.html"), File.join(dir, '_includes')) }
  File.write(File.join(dir, '_layouts/general.html'), '{% if page.hero_image %}{% include section_hero.html %}{% endif %}{{ content }}')
  File.write(File.join(dir, '_layouts/default.html'), '{{ content }}{% include main_menu.html %}{% include mobile_menu.html %}')
  File.write(File.join(dir, '_includes/quick_start_module.html'), '')
  FileUtils.cp(File.join(root, 'index.html'), dir)
  File.write(File.join(dir, 'all-events.html'), "---\n---\n{% include event_list.html limit=20 %}")
  File.write(File.join(dir, 'metadata.html'), "---\n---\n{% include open_graph_and_meta.html %}")
  FileUtils.cp(File.join(root, '_data/navigation.yml'), File.join(dir, '_data'))
  %w[projects groups].each do |name|
    FileUtils.cp_r(Dir.glob(File.join(root, name, '*')), File.join(dir, name))
    4.times do |i|
      data = { 'uid' => "#{name}-#{i}", 'title' => "#{name} #{i}", 'summary' => 'Fixture', 'order' => i, 'featured' => i < 3, 'group_type' => (i == 3 ? 'series' : 'community') }
      data['group_ids'] = ['groups-0'] if name == 'projects' && i == 0
      File.write(File.join(dir, "_#{name}/#{name}-#{i}.md"), data.to_yaml + "---\n")
    end
    File.write(File.join(dir, "_#{name}/hidden.md"), { 'title' => 'Hidden activity', 'published' => false }.to_yaml + "---\n")
  end
  FileUtils.cp_r(Dir.glob(File.join(root, 'events/*')), File.join(dir, 'events'))
  config = { 'source' => dir, 'destination' => File.join(dir, '_site'), 'baseurl' => '/preview', 'timezone' => 'Asia/Seoul',
    'collections' => %w[events projects groups features].to_h { |name| [name, { 'output' => true, 'permalink' => "/#{name}/:path/" }] },
    'defaults' => YAML.safe_load(File.read(File.join(root, '_config.yml')))['defaults'] }
  session = { 'uid' => 'talk', 'title' => 'Public talk', 'video' => { 'state' => 'public', 'youtube_id' => 'aV2lNdf1UHc', 'start_seconds' => 90 }, 'slides' => { 'state' => 'public', 'url' => 'https://example.org/slides.pdf', 'size_bytes' => 10485760, 'format' => 'pdf' } }
  write = lambda do |uid, extra|
    data = { 'uid' => uid, 'title' => uid, 'summary' => 'Fixture', 'event_date' => '2099-10-14', 'event_status' => 'scheduled', 'last_verified_at' => '2026-10-02' }.merge(extra)
    File.write(File.join(dir, "_events/#{uid}.md"), data.to_yaml + "---\nFixture body\n")
  end
  write.call('future', { 'group_ids' => ['groups-0'], 'project_ids' => ['projects-0'], 'sessions' => [session], 'registration' => { 'url' => 'https://example.org/register', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('cancelled', { 'event_status' => 'cancelled', 'registration' => { 'url' => 'https://example.org/register', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('postponed', { 'event_status' => 'postponed' })
  write.call('past-unconfirmed', { 'event_date' => '2000-01-01', 'registration' => { 'url' => 'https://example.org/register', 'closes_at' => '2000-01-01T00:00:00+09:00' } })
  write.call('not-open', { 'registration' => { 'url' => 'https://example.org/register', 'opens_at' => '2099-10-01T00:00:00+09:00', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('early-closed', { 'registration' => { 'url' => 'https://example.org/register', 'status' => 'closed', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('draft-secret', { 'published' => false })
  write.call('limited', { 'event_date' => '2000-01-01', 'event_status' => 'held', 'recording' => 'limited' })
  write.call('no-media', { 'event_date' => '2099-11-01', 'sessions' => [{ 'uid' => 'talk', 'title' => 'No media' }] })
  write.call('slides-only', { 'event_date' => '2000-01-01', 'sessions' => [{ 'uid' => 'talk', 'title' => 'Slides only', 'slides' => { 'state' => 'public', 'url' => 'https://example.org/slides-only.pdf' } }] })
  write.call('private', { 'sessions' => [{ 'uid' => 'talk', 'title' => 'Private talk', 'video' => { 'state' => 'private' }, 'slides' => { 'state' => 'private' } }] })
  Jekyll::Site.new(Jekyll.configuration(config)).process
  read = ->(uid) { File.read(File.join(dir, "_site/events/#{uid}/index.html")) }
  assert = ->(test, label) { raise label unless test }
  future = read.call('future')
  assert.call(future.include?('참가 신청하기') && future.include?('data-registration-closes'), 'future registration absent')
  assert.call(future.include?('watch?v=aV2lNdf1UHc&amp;t=90s') && future.include?('slides.pdf') && future.include?('10.0 MB'), 'media pairing missing')
  assert.call(future.include?('<iframe src="https://www.youtube-nocookie.com/embed/aV2lNdf1UHc?start=90"') && future.include?('loading="lazy"'), 'inline lazy player or start time missing')
  assert.call(!future.include?('<details') && !future.include?('autoplay='), 'player must be visible without autoplay')
  assert.call(future.include?('title="발표 영상: Public talk"'), 'player accessible title missing')
  %w[no-media private limited].each { |uid| assert.call(!read.call(uid).include?('<iframe'), "#{uid}: unavailable video embedded") }
  %w[events/future projects/projects-0 groups/groups-0].each do |name|
    detail = File.read(File.join(dir, "_site/#{name}/index.html"))
    assert.call(detail.scan('<h1>').size == 1 && detail.include?('class="section-hero-image" src="/preview/assets/'), "#{name}: detail hero missing or heading duplicated")
  end
  assert.call(future.include?('href="/preview/events/"'), 'baseurl not honored')
  %w[groups/groups-0 projects/projects-0].each do |name|
    related = File.read(File.join(dir, "_site/#{name}/index.html"))
    assert.call(related.include?('href="/preview/events/future/"'), "#{name}: a linked event did not appear automatically")
    assert.call(future.include?("href=\"/preview/#{name}/\""), "future: related #{name} link missing")
  end
  %w[cancelled past-unconfirmed not-open early-closed].each { |uid| assert.call(!read.call(uid).include?('참가 신청하기'), "#{uid}: invalid signup CTA") }
  assert.call(read.call('past-unconfirmed').include?('지난 일정'), 'past date inferred as held')
  assert.call(read.call('limited').include?('음성이 녹음되지'), 'audio failure notice missing')
  %w[no-media private].each { |uid| assert.call(!read.call(uid).include?('YouTube에서 보기') && !read.call(uid).include?('slides.pdf'), "#{uid}: media button leaked") }
  assert.call(read.call('private').include?('비공개'), 'private state notice missing')
  index = File.read(File.join(dir, '_site/events/index.html'))
  archive = File.read(File.join(dir, '_site/events/archive/index.html'))
  materials = File.read(File.join(dir, '_site/events/materials/index.html'))
  assert.call(archive.include?('/events/past-unconfirmed/') && !index.include?('/events/past-unconfirmed/'), 'past events mixed into upcoming list')
  %w[future slides-only].each do |uid|
    assert.call(read.call(uid).include?("href=\"/preview/events/materials/#material-#{uid}--talk\""), "#{uid}: session return link missing")
    assert.call(materials.include?("id=\"material-#{uid}--talk\" tabindex=\"-1\""), "#{uid}: focusable material destination missing")
  end
  %w[private no-media].each { |uid| assert.call(!read.call(uid).include?('발표 목록으로 돌아가기'), "#{uid}: return link has no public destination") }
  %w[future past-unconfirmed cancelled postponed no-media].each do |uid|
    tabs = read.call(uid).match(/<nav class="content-tabs".*?<\/nav>/m)[0]
    %w[/events/ /events/materials/ /events/archive/].each { |path| assert.call(tabs.include?("href=\"/preview#{path}\""), "#{uid}: event navigation missing") }
    assert.call(!tabs.include?('aria-current="page"'), "#{uid}: detail falsely marks a listing as current")
  end
  assert.call(materials.include?('/preview/events/future/#session-talk') && !materials.include?('Private talk') && !materials.include?('No media'), 'public materials missing or unavailable materials leaked')
  assert.call(index.include?('/events/future/') && index.include?('취소·일정 변경 안내') && index.include?('/events/cancelled/') && index.include?('/events/postponed/'), 'future or changed event disappeared')
  series = index.match(/<h2>행사 시리즈<\/h2>(.*?)<\/section>/m)[1]
  assert.call(series.include?('/groups/groups-3/') && !series.include?('/groups/groups-0/'), 'event series mixed with community groups')
  assert.call(!index.include?('draft-secret') && !File.exist?(File.join(dir, '_site/events/draft-secret/index.html')), 'unpublished event leaked')
  home = File.read(File.join(dir, '_site/index.html'))
  project_section = home.match(/id="home-projects">(.*?)id="home-events"/m)[1]
  assert.call(project_section.scan('class="activity-card"').size == 3, 'homepage must show three public projects')
  assert.call(project_section.scan(/<h3><a href="([^"]+)"/).uniq.size == 3, 'homepage projects must be distinct')
  event_section = home.match(/id="home-events">(.*?)key-features-module/m)[1]
  assert.call(event_section.scan('class="event-row"').size == 3, 'homepage must show three events')
  all_events = File.read(File.join(dir, '_site/all-events.html'))
  dates = all_events.scan(/class="event-date" datetime="([^"]+)"/).flatten
  assert.call(dates == dates.sort.reverse && dates.include?('2000-01-01') && dates.include?('2099-11-01'), 'combined list must include past and future events in descending date order')
  assert.call(!event_section.include?('/events/cancelled/') && !event_section.include?('/events/postponed/') && !event_section.include?('/events/past-unconfirmed/'), 'homepage mixed past or changed events into upcoming events')
  upcoming_dates = event_section.scan(/class="event-date" datetime="([^"]+)"/).flatten
  assert.call(upcoming_dates == upcoming_dates.sort, 'homepage upcoming events are not chronological')
  navigation = home.scan(/<nav class="main-menu".*?<\/nav>/m)
  assert.call(navigation.size == 2, 'desktop or mobile navigation missing')
  navigation.each do |menu|
    %w[projects groups events].each { |name| assert.call(menu.include?("href=\"/preview/#{name}/\""), "#{name} direct navigation missing") }
    assert.call(!menu.include?('/projects/projects-') && !menu.include?('/groups/groups-'), 'detail links mixed into global navigation')
  end
  %w[projects groups].each do |collection|
    listing = File.read(File.join(dir, "_site/#{collection}/index.html"))
    all = File.read(File.join(dir, "_site/#{collection}/all/index.html"))
    assert.call(listing.include?("/#{collection}/#{collection}-0/") && !listing.include?("/#{collection}/#{collection}-3/"), 'featured selection ignored')
    assert.call(all.include?("/#{collection}/#{collection}-3/") && !all.include?('Hidden activity'), 'archive dropped a public record or leaked a draft')
    assert.call(!home.include?("menu-#{collection}"), 'individual activities remain in global dropdowns')
  end
  %w[projects groups].each do |collection|
    [0, 3].each do |i|
      detail = File.read(File.join(dir, "_site/#{collection}/#{collection}-#{i}/index.html"))
      breadcrumb = detail.match(/<nav class="content-breadcrumb".*?<\/nav>/m)[0]
      assert.call(breadcrumb.include?("href=\"/preview/#{collection}/all/#activity-#{collection}--#{collection}-#{i}\""), "#{collection}-#{i}: full directory return path missing")
    end
  end
  group = File.read(File.join(dir, '_site/groups/groups-0/index.html'))
  project = File.read(File.join(dir, '_site/projects/projects-0/index.html'))
  assert.call(group.include?('/preview/projects/projects-0/') && project.include?('/preview/groups/groups-0/'), 'bidirectional project-group relationship missing')
  assert.call(group.include?('id="related-events"') && project.include?('id="related-events"'), 'event relationship anchor missing')
  %w[projects groups events].each do |name|
    listing = File.read(File.join(dir, "_site/#{name}/index.html"))
    assert.call(listing.scan('<h1>').size == 1 && listing.include?('class="section-hero-image" src="/preview/assets/'), "#{name} photo hero missing")
  end

  original_environment = ENV['JEKYLL_ENV']
  begin
    %w[production preview].each do |environment|
      ENV['JEKYLL_ENV'] = environment
      Jekyll::Site.new(Jekyll.configuration(config)).process
      metadata = File.read(File.join(dir, '_site/metadata.html'))
      expected = environment == 'preview' ? 'noindex, nofollow' : 'index, follow, max-image-preview:large'
      assert.call(metadata.include?("<meta name=\"robots\" content=\"#{expected}\" />"), "#{environment}: incorrect indexing policy")
      assert.call(metadata.scan('name="robots"').size == 1, "#{environment}: conflicting indexing directives")
    end
  ensure
    ENV['JEKYLL_ENV'] = original_environment
  end

end
puts 'Content rendering: future, changed, draft, registration, video/slides and missing-media, photo headers, activity directories, relationships, public materials, direct navigation and upcoming homepage limits passed.'
