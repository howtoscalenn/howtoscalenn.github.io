require "cgi"

module AutoToc
  module_function

  def strip_tags(html)
    html.to_s.gsub(/<[^>]*>/, "")
  end

  def normalize_text(text)
    CGI.unescapeHTML(text.to_s).strip.gsub(/\s+/, " ")
  end

  def extract_id(attrs)
    # Match id attribute with or without leading space, handling both quote styles
    attrs.to_s[/\bid=(["'])([^"']+)\1/i, 2] ||
      attrs.to_s[/\bid=([^\s>"']+)/i, 1]
  end

  def default_slugify(text)
    normalize_text(text).downcase
                        .gsub(/[^a-z0-9\s-]/, "")
                        .tr("_", " ")
                        .gsub(/\s+/, "-")
                        .gsub(/-+/, "-")
                        .gsub(/\A-|-+\z/, "")
  end

  def ensure_heading_ids(html, slugify: nil)
    slugify_fn = slugify || method(:default_slugify)
    seen = Hash.new(0)

    html.to_s.gsub(/<(h2|h3)([^>]*)>(.*?)<\/\1>/im) do
      tag = Regexp.last_match(1)
      attrs = Regexp.last_match(2)
      inner = Regexp.last_match(3)

      existing_id = extract_id(attrs)
      next "<#{tag}#{attrs}>#{inner}</#{tag}>" if existing_id && !existing_id.empty?

      name = normalize_text(strip_tags(inner))
      slug = slugify_fn.call(name)
      slug = "section" if slug.empty?

      seen[slug] += 1
      unique_slug = seen[slug] == 1 ? slug : "#{slug}-#{seen[slug]}"

      if attrs.to_s.strip.empty?
        "<#{tag} id=\"#{unique_slug}\">#{inner}</#{tag}>"
      else
        "<#{tag}#{attrs} id=\"#{unique_slug}\">#{inner}</#{tag}>"
      end
    end
  end

  # Extract a TOC data structure compatible with `_layouts/distill.html`:
  # [
  #   { "name" => "...", "id" => "...", "subsections" => [{ "name" => "...", "id" => "..." }, ...] },
  #   ...
  # ]
  #
  # We intentionally only pick up h2/h3 to match the existing layout UI.
  def extract(html)
    sections = []
    current = nil

    html.to_s.scan(/<(h2|h3)([^>]*)>(.*?)<\/\1>/im) do |tag, attrs, inner|
      name = normalize_text(strip_tags(inner))
      next if name.empty?

      id = extract_id(attrs)
      entry = { "name" => name }
      entry["id"] = id if id && !id.empty?

      if tag.downcase == "h2"
        current = entry
        sections << current
      else
        next unless current
        current["subsections"] ||= []
        current["subsections"] << entry
      end
    end

    sections
  end

  # Extract TOC from markdown content (for Pages where content isn't converted yet)
  # Matches: ## Heading {#id} or ## Heading
  def extract_from_markdown(content, slugify: nil)
    slugify_fn = slugify || method(:default_slugify)
    sections = []
    current = nil

    content.to_s.each_line do |line|
      # Match ## or ### headings (use \# to avoid interpolation)
      if line =~ /^(\#{2,3})\s+(.+?)\s*(?:\{\#([^}]+)\})?\s*$/
        level = Regexp.last_match(1).length
        raw_name = Regexp.last_match(2)
        explicit_id = Regexp.last_match(3)

        # Strip HTML tags like <mark>...</mark>
        name = normalize_text(strip_tags(raw_name))
        next if name.empty?

        # Use explicit id or generate from name using provided slugify
        id = explicit_id && !explicit_id.empty? ? explicit_id : slugify_fn.call(name)
        entry = { "name" => name, "id" => id }

        if level == 2
          current = entry
          sections << current
        elsif level == 3 && current
          current["subsections"] ||= []
          current["subsections"] << entry
        end
      end
    end

    sections
  end
end

if defined?(Jekyll)
  # Use :pre_render hook with payload to directly set page.toc for templates
  Jekyll::Hooks.register [:pages, :documents], :pre_render do |doc, payload|
    # Respect explicit toc in front matter.
    next if doc.data.key?("toc")

    # Default behavior:
    # - opt-in via `auto_toc: true`
    # - or automatically for distill layout docs
    auto_enabled = doc.data["auto_toc"] == true || doc.data["layout"].to_s == "distill"
    next unless auto_enabled

    content = doc.content.to_s
    slugify_fn = ->(t) { Jekyll::Utils.slugify(t.to_s, mode: "default") }

    # Try HTML extraction first (for Documents where content is already converted)
    toc = AutoToc.extract(content)

    # If no HTML headings found, try markdown extraction (for Pages)
    toc = AutoToc.extract_from_markdown(content, slugify: slugify_fn) if toc.empty?

    unless toc.empty?
      doc.data["toc"] = toc
      payload["page"]["toc"] = toc if payload && payload["page"]
    end
  end
end


