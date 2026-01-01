require "minitest/autorun"
require_relative "../_plugins/auto_toc"

class AutoTocTest < Minitest::Test
  def test_extract_h2_h3_with_ids_and_nested_subsections
    html = <<~HTML
      <h2 id="motivation"><mark> Motivation </mark></h2>
      <p>hi</p>
      <h3 id="why">Why?</h3>
      <h3>Plain Sub</h3>
      <h2>Second Section</h2>
      <h3 id="s2-sub">S2 Sub</h3>
    HTML

    toc = AutoToc.extract(html)

    assert_equal 2, toc.size
    assert_equal "Motivation", toc[0]["name"]
    assert_equal "motivation", toc[0]["id"]
    assert_equal 2, toc[0]["subsections"].size
    assert_equal({ "name" => "Why?", "id" => "why" }, toc[0]["subsections"][0])
    assert_equal({ "name" => "Plain Sub" }, toc[0]["subsections"][1])

    assert_equal "Second Section", toc[1]["name"]
    assert_equal 1, toc[1]["subsections"].size
    assert_equal({ "name" => "S2 Sub", "id" => "s2-sub" }, toc[1]["subsections"][0])
  end

  def test_ignores_orphan_h3_before_any_h2
    html = <<~HTML
      <h3 id="orphan">Orphan</h3>
      <h2 id="top">Top</h2>
      <h3>Child</h3>
    HTML

    toc = AutoToc.extract(html)

    assert_equal 1, toc.size
    assert_equal "Top", toc[0]["name"]
    assert_equal 1, toc[0]["subsections"].size
    assert_equal "Child", toc[0]["subsections"][0]["name"]
  end
end


