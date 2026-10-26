from sam_engine.sam_text import parse_output_text


def test_parse_example_line():
    # Example grammar from protocol/sam3.md (payload illustrative only).
    line = ("<0f>0<|box;x1=10;y1=20;x2=14;y2=24;w=100;h=80;c=0.75|>"
            "<|mask;x=0;y=0;c=0.75;data=5,5,!!!!!(QO(0lu8?|>")
    records, diagnostics = parse_output_text(line)
    assert diagnostics == []
    assert len(records) == 1
    rec = records[0]
    assert rec.object_id == "0"
    # Inclusive x2/y2 -> half-open w/h: 14-10+1=5, 24-20+1=5.
    assert (rec.x, rec.y, rec.w, rec.h) == (10, 20, 5, 5)
    assert rec.confidence == 0.75
    assert (rec.mask_width, rec.mask_height) == (5, 5)
    assert rec.mask_encoding == "one_bit"
    assert rec.mask_payload.startswith("!")


def test_parse_multi_object_line():
    line = ("<0f>0<|box;x1=0;y1=0;x2=9;y2=9;w=100;h=100|>"
            "<|mask;x=0;y=0;data=10,10,!!!!!!!!!!|>,"
            "1<|box;x1=50;y1=50;x2=59;y2=59;w=100;h=100|>"
            "<|mask;x=0;y=0;data=10,10,!!!!!!!!!!|>")
    records, _ = parse_output_text(line)
    assert [r.object_id for r in records] == ["0", "1"]
    assert records[1].x == 50 and records[1].w == 10


def test_malformed_lines_skipped_with_diagnostic():
    records, diagnostics = parse_output_text(
        "not a frame line\n"
        "<0f>0<|box;x1=0;y1=0;x2=9;y2=9;w=100;h=100|>"
        "<|mask;x=0;y=0;data=10,10,!!!!!!!!!!|>\n"
        "<3f>0<|box;x1=0;y1=0;x2=9;y2=9;w=100;h=100|>"
        "<|mask;x=0;y=0;data=10,10,!!!!!!!!!!|>"
    )
    assert len(records) == 1  # only the frame-0 line survives
    assert len(diagnostics) == 2


def test_lossless_marker():
    line = ("<0f>2<|box;x1=0;y1=0;x2=4;y2=4;w=50;h=50|>"
            "<|mask;x=0;y=0;data=5,5,~~~~~abc|>")
    records, _ = parse_output_text(line)
    assert records[0].mask_encoding == "lossless"
    assert records[0].object_id == "2"
