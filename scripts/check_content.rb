#!/usr/bin/env ruby
# Validate the small content contract shared by people, agents and Liquid.
require 'yaml'
require 'date'
require 'time'
require 'uri'

class ContentCheck
  SLUG = /\A[a-z0-9]+(?:-[a-z0-9]+)*\z/
  STATES = %w[public pending private unknown not_recorded limited].freeze
  attr_reader :errors, :records

  def initialize(root)
    @root, @errors, @records = root, [], {}
  end

  def error(path, message)
    @errors << "#{path}: #{message}"
  end

  def require_text(path, data, keys)
    keys.each { |key| error(path, "#{key} must be a nonempty string") unless data[key].is_a?(String) && !data[key].strip.empty? }
  end

  def date(path, value, key, timestamp: false)
    pattern = timestamp ? /\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\z/ : /\A\d{4}-\d{2}-\d{2}\z/
    raise ArgumentError unless value.is_a?(String) && pattern.match?(value)
    Date.iso8601(value[0, 10]) # Time.iso8601 alone normalizes invalid calendar days.
    timestamp ? Time.iso8601(value) : Date.iso8601(value)
  rescue ArgumentError
    error(path, "#{key} must be a quoted ISO date#{timestamp ? ' with an explicit UTC offset' : ''}")
    nil
  end

  def url(path, value, internal: true)
    uri = URI.parse(value.to_s)
    valid = uri.scheme == 'https' && uri.host && !uri.userinfo
    valid ||= internal && value.is_a?(String) && value.start_with?('/') && !value.start_with?('//') && !value.include?('..')
    error(path, "invalid URL: #{value}") unless valid
  rescue URI::InvalidURIError
    error(path, "invalid URL: #{value}")
  end

  def links(path, data)
    case data
    when Hash
      data.each do |key, value|
        if key == 'url' || key.end_with?('_url')
          url(path, value)
        else
          links(path, value)
        end
      end
    when Array
      data.each { |value| links(path, value) }
    end
  end

  def media(path, media, type)
    return if media.nil?
    unless media.is_a?(Hash)
      error(path, "#{type} must be a mapping")
      return
    end
    allowed = type == 'video' ? STATES : %w[public pending private unknown]
    error(path, "invalid #{type}.state") unless allowed.include?(media['state'])
    if media['state'] == 'public'
      if type == 'video'
        error(path, 'public video requires an 11-character youtube_id') unless /\A[A-Za-z0-9_-]{11}\z/.match?(media['youtube_id'].to_s)
      else
        require_text(path, media, ['url'])
      end
    elsif media.key?('url') || media.key?('youtube_id')
      error(path, "non-public #{type} must not contain a URL or video ID")
    end
    if media.key?('start_seconds') && !(media['start_seconds'].is_a?(Integer) && media['start_seconds'] >= 0)
      error(path, 'start_seconds must be a nonnegative integer')
    end
    if media.key?('size_bytes') && !(media['size_bytes'].is_a?(Integer) && media['size_bytes'] > 0)
      error(path, 'size_bytes must be a positive integer')
    end
  end

  def run
    %w[projects groups events].each do |collection|
      @records[collection] = {}
      Dir.glob(File.join(@root, "_#{collection}", '*.md')).sort.each do |file|
        path = file.delete_prefix(@root + '/')
        text = File.read(file)
        match = text.match(/\A---\s*\n(.*?)\n---\s*\n/m)
        unless match
          error(path, 'missing YAML front matter')
          next
        end
        begin
          data = YAML.safe_load(match[1])
        rescue Psych::Exception => e
          error(path, "invalid YAML (quote dates): #{e.message.lines.first.strip}")
          next
        end
        unless data.is_a?(Hash)
          error(path, 'front matter must be a mapping')
          next
        end
        require_text(path, data, %w[uid title summary])
        uid = data['uid']
        error(path, 'uid must match the lowercase, hyphenated filename') unless SLUG.match?(uid.to_s) && File.basename(file, '.md') == uid
        error(path, "duplicate uid #{uid}") if @records[collection].key?(uid)
        @records[collection][uid] = data
        date(path, data['last_verified_at'], 'last_verified_at')
        if !data['sources'].is_a?(Array) || data['sources'].empty?
          error(path, 'sources must contain at least one public source URL')
        else
          data['sources'].each { |source| url(path, source, internal: false) }
        end
        if data.key?('published') && ![true, false].include?(data['published'])
          error(path, 'published must be a boolean')
        end
        error(path, 'use event_date, not Jekyll date') if collection == 'events' && data.key?('date')
        error(path, 'projects require a website_url or repository_url') if collection == 'projects' && !data['website_url'] && !data['repository_url']
        links(path, data)
        if text.match?(%r{(?:/Users/|file://|upload://|REPLACE-WITH|PLACEHOLDER)})
          error(path, 'remove local paths and unpublished placeholders')
        end
      end
    end
    @records.each do |collection, items|
      items.each do |uid, data|
        path = "_#{collection}/#{uid}.md"
        { 'group_ids' => 'groups', 'project_ids' => 'projects' }.each do |key, target|
          next unless data.key?(key)
          unless data[key].is_a?(Array) && data[key].all? { |id| id.is_a?(String) }
            error(path, "#{key} must be a list of uids")
            next
          end
          data[key].each { |id| error(path, "unknown #{target} uid: #{id}") unless @records[target].key?(id) }
        end
        if collection == 'groups'
          programs = data.fetch('programs', [])
          unless programs.is_a?(Array) && programs.all? { |p| p.is_a?(Hash) && SLUG.match?(p['uid'].to_s) && p['title'].is_a?(String) }
            error(path, 'programs require uid and title')
          else
            error(path, 'duplicate program uid') unless programs.map { |p| p['uid'] }.uniq.size == programs.size
          end
        end
        next unless collection == 'events'
        date(path, data['event_date'], 'event_date')
        error(path, 'invalid event_status') unless %w[scheduled held cancelled postponed].include?(data['event_status'])
        error(path, 'invalid format') unless %w[online offline hybrid].include?(data['format'])
        error(path, 'invalid recording') if data['recording'] && !%w[limited not_recorded].include?(data['recording'])
        start = date(path, data['starts_at'], 'starts_at', timestamp: true) if data['starts_at']
        finish = date(path, data['ends_at'], 'ends_at', timestamp: true) if data['ends_at']
        error(path, 'ends_at requires starts_at and must be later') if finish && (!start || finish <= start)
        error(path, 'starts_at must match event_date in Korea time') if start && start.getlocal('+09:00').strftime('%F') != data['event_date']
        groups = Array(data['group_ids']).filter_map { |id| @records['groups'][id] }
        programs = groups.flat_map { |g| Array(g['programs']) }.select { |p| p.is_a?(Hash) }.map { |p| p['uid'] }
        error(path, 'program must belong to a referenced group') if data['program'] && !programs.include?(data['program'])
        error(path, 'vLLM.KR events require a tracked Meetup program') if Array(data['group_ids']).include?('vllm-kr') && !%w[korea-meetup community-meetup].include?(data['program'])
        if data['registration']
          registration = data['registration']
          if registration.is_a?(Hash)
            require_text(path, registration, ['url'])
            close = date(path, registration['closes_at'], 'registration.closes_at', timestamp: true)
            open = date(path, registration['opens_at'], 'registration.opens_at', timestamp: true) if registration['opens_at']
            error(path, 'registration closes before it opens') if open && close && close <= open
            error(path, 'invalid registration.status') if registration['status'] && !%w[open closed not_open].include?(registration['status'])
          else
            error(path, 'registration must be a mapping')
          end
        end
        sessions = data.fetch('sessions', [])
        unless sessions.is_a?(Array) && sessions.all? { |s| s.is_a?(Hash) }
          error(path, 'sessions must be a list of mappings')
          next
        end
        ids = sessions.map { |s| s['uid'] }
        error(path, 'session uids must be unique slugs') unless ids.uniq == ids && ids.all? { |id| SLUG.match?(id.to_s) }
        sessions.each do |session|
          require_text(path, session, %w[uid title])
          media(path, session['video'], 'video')
          media(path, session['slides'], 'slides')
          speakers = session.fetch('speakers', [])
          error(path, 'speakers require names') unless speakers.is_a?(Array) && speakers.all? { |s| s.is_a?(Hash) && s['name'].is_a?(String) && !s['name'].empty? }
        end
      end
    end
    @errors
  end
end

if $PROGRAM_NAME == __FILE__
  check = ContentCheck.new(File.expand_path('..', __dir__))
  errors = check.run
  abort(errors.join("\n")) unless errors.empty?
  puts "Content valid: #{check.records.transform_values(&:size)}"
end
