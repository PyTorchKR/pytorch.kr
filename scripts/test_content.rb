#!/usr/bin/env ruby
require 'tmpdir'
require 'fileutils'
require_relative 'check_content'

root = File.expand_path('..', __dir__)
checks = 0
calendar = ContentCheck.new(root)
calendar.date('fixture.md', '2026-02-30T19:00:00+09:00', 'registration.closes_at', timestamp: true)
raise 'invalid timestamp date was silently normalized' if calendar.errors.empty?
checks += 1
Dir.mktmpdir('pytorchkr-content-') do |dir|
  %w[projects groups events].each { |name| FileUtils.cp_r(File.join(root, "_#{name}"), dir) }
  path = File.join(dir, '_events/physical-ai-seminar-2026.md')
  original = File.read(path)
  assert = lambda do |label, change, expected|
    parts = original.split(/^---\s*$\n?/, 3)
    data = YAML.safe_load(parts[1])
    change.call(data)
    File.write(path, data.to_yaml + "---\n" + parts[2])
    errors = ContentCheck.new(dir).run
    raise "#{label}: expected #{expected.inspect}, got #{errors.inspect}" unless errors.any? { |e| e.include?(expected) }
    checks += 1
  end
  assert.call('broken relationship', ->(d) { d['group_ids'] = ['missing-group'] }, 'unknown groups uid')
  assert.call('bad date', ->(d) { d['event_date'] = '2026-02-30' }, 'quote')
  assert.call('timezone required', ->(d) { d['starts_at'] = '2026-10-14T19:00:00' }, 'explicit UTC offset')
  assert.call('backwards time', ->(d) { d['ends_at'] = '2026-10-14T18:00:00+09:00' }, 'must be later')
  assert.call('wrong program', ->(d) { d['program'] = 'hands-on' }, 'program must belong')
  assert.call('unsafe URL', ->(d) { d['website_url'] = 'javascript:alert(1)' }, 'invalid URL')
  assert.call('private source', ->(d) { d['sources'] = ['file:///Users/example/private.pdf'] }, 'invalid URL')
  assert.call('early close', ->(d) { d['registration']['opens_at'] = '2026-10-07T00:00:00+09:00' }, 'closes before')
  assert.call('unknown status', ->(d) { d['event_status'] = 'done' }, 'invalid event_status')
  assert.call('unsafe draft', ->(d) { d['summary'] = 'upload://REPLACE-WITH-image' }, 'unpublished placeholders')
  assert.call('missing media', ->(d) { d['sessions'] = [{ 'uid' => 'talk', 'title' => 'Talk', 'video' => { 'state' => 'public' } }] }, '11-character')
  assert.call('private URL leak', ->(d) { d['sessions'] = [{ 'uid' => 'talk', 'title' => 'Talk', 'slides' => { 'state' => 'private', 'url' => 'https://example.org/private.pdf' } }] }, 'must not contain')
  assert.call('duplicate sessions', ->(d) { d['sessions'] = Array.new(2) { { 'uid' => 'talk', 'title' => 'Talk' } } }, 'unique slugs')
  File.write(path, original)
  raise ContentCheck.new(dir).run.join("\n") unless ContentCheck.new(dir).run.empty?
end
puts "Content validator: #{checks} rejection cases and restored valid content passed."
