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
    attrs.to_s[/\sid=(["'])([^"']+)\1/i, 2]
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
end

if defined?(Jekyll)
  Jekyll::Hooks.register [:pages, :documents], :post_convert do |doc|
    # Respect explicit toc in front matter.
    next if doc.data.key?("toc")

    # Default behavior:
    # - opt-in via `auto_toc: true`
    # - or automatically for distill layout docs
    auto_enabled = doc.data["auto_toc"] == true || doc.data["layout"].to_s == "distill"
    next unless auto_enabled

    toc = AutoToc.extract(doc.content)
    doc.data["toc"] = toc unless toc.empty?
  end
end


