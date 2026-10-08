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
  %w[session content_links event_list event_status activity_card section_hero main_menu mobile_menu dropdown_links].each { |name| FileUtils.cp(File.join(root, "_includes/#{name}.html"), File.join(dir, '_includes')) }
  File.write(File.join(dir, '_layouts/general.html'), '{% if page.hero_image %}{% include section_hero.html %}{% endif %}{{ content }}')
  File.write(File.join(dir, '_layouts/default.html'), '{{ content }}{% include main_menu.html %}{% include mobile_menu.html %}')
  File.write(File.join(dir, '_includes/quick_start_module.html'), '')
  FileUtils.cp(File.join(root, 'index.html'), dir)
  File.write(File.join(dir, 'all-events.html'), "---\n---\n{% include event_list.html limit=20 %}")
  FileUtils.cp(File.join(root, '_data/navigation.yml'), File.join(dir, '_data'))
  %w[projects groups].each do |name|
    FileUtils.cp(File.join(root, "#{name}/index.html"), File.join(dir, name))
    4.times do |i|
      data = { 'uid' => "#{name}-#{i}", 'title' => "#{name} #{i}", 'summary' => 'Fixture', 'order' => i }
      File.write(File.join(dir, "_#{name}/#{name}-#{i}.md"), data.to_yaml + "---\n")
    end
    File.write(File.join(dir, "_#{name}/hidden.md"), { 'title' => 'Hidden activity', 'published' => false }.to_yaml + "---\n")
  end
  FileUtils.cp(File.join(root, 'events/index.html'), File.join(dir, 'events'))
  config = { 'source' => dir, 'destination' => File.join(dir, '_site'), 'baseurl' => '/preview', 'timezone' => 'Asia/Seoul',
    'collections' => %w[events projects groups features].to_h { |name| [name, { 'output' => true, 'permalink' => "/#{name}/:path/" }] },
    'defaults' => YAML.safe_load(File.read(File.join(root, '_config.yml')))['defaults'] }
  session = { 'uid' => 'talk', 'title' => 'Public talk', 'video' => { 'state' => 'public', 'youtube_id' => 'aV2lNdf1UHc', 'start_seconds' => 90 }, 'slides' => { 'state' => 'public', 'url' => 'https://example.org/slides.pdf', 'size_bytes' => 10485760, 'format' => 'pdf' } }
  write = lambda do |uid, extra|
    data = { 'uid' => uid, 'title' => uid, 'summary' => 'Fixture', 'event_date' => '2099-10-14', 'event_status' => 'scheduled', 'last_verified_at' => '2026-10-02' }.merge(extra)
    File.write(File.join(dir, "_events/#{uid}.md"), data.to_yaml + "---\nFixture body\n")
  end
  write.call('future', { 'sessions' => [session], 'registration' => { 'url' => 'https://example.org/register', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('cancelled', { 'event_status' => 'cancelled', 'registration' => { 'url' => 'https://example.org/register', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('postponed', { 'event_status' => 'postponed' })
  write.call('past-unconfirmed', { 'event_date' => '2000-01-01', 'registration' => { 'url' => 'https://example.org/register', 'closes_at' => '2000-01-01T00:00:00+09:00' } })
  write.call('not-open', { 'registration' => { 'url' => 'https://example.org/register', 'opens_at' => '2099-10-01T00:00:00+09:00', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('early-closed', { 'registration' => { 'url' => 'https://example.org/register', 'status' => 'closed', 'closes_at' => '2099-10-13T23:59:00+09:00' } })
  write.call('draft-secret', { 'published' => false })
  write.call('limited', { 'event_date' => '2000-01-01', 'event_status' => 'held', 'recording' => 'limited' })
  write.call('no-media', { 'event_date' => '2099-11-01', 'sessions' => [{ 'uid' => 'talk', 'title' => 'No media' }] })
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
  %w[cancelled past-unconfirmed not-open early-closed].each { |uid| assert.call(!read.call(uid).include?('참가 신청하기'), "#{uid}: invalid signup CTA") }
  assert.call(read.call('past-unconfirmed').include?('지난 일정'), 'past date inferred as held')
  assert.call(read.call('limited').include?('음성이 녹음되지'), 'audio failure notice missing')
  %w[no-media private].each { |uid| assert.call(!read.call(uid).include?('YouTube에서 보기') && !read.call(uid).include?('slides.pdf'), "#{uid}: media button leaked") }
  assert.call(read.call('private').include?('비공개'), 'private state notice missing')
  index = File.read(File.join(dir, '_site/events/index.html'))
  assert.call(index.include?('/events/future/') && index.include?('취소·일정 변경 안내') && index.include?('/events/cancelled/') && index.include?('/events/postponed/'), 'future or changed event disappeared')
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
  assert.call(event_section.scan(/class="event-date" datetime="([^"]+)"/).flatten == dates.first(3), 'homepage must show the latest three events')
  %w[desktop mobile].each do |mode|
    %w[projects groups].each do |collection|
      panel = home.match(/id="#{mode}-menu-#{collection}"[^>]*>(.*?)<\/(?:div|ul)>/m)[1]
      assert.call(panel.include?("href=\"/preview/#{collection}/\""), "#{mode} collection index missing")
      4.times { |i| assert.call(panel.include?("href=\"/preview/#{collection}/#{collection}-#{i}/\""), "#{mode} collection item missing") }
      assert.call(!panel.include?('hidden'), 'unpublished navigation entry leaked')
    end
  end
  %w[projects groups events].each do |name|
    listing = File.read(File.join(dir, "_site/#{name}/index.html"))
    assert.call(listing.scan('<h1>').size == 1 && listing.include?('class="section-hero-image" src="/preview/assets/'), "#{name} photo hero missing")
  end

end
puts 'Content rendering: future, changed, draft, registration, video/slides and missing-media, photo headers, collection navigation and homepage limits passed.'
