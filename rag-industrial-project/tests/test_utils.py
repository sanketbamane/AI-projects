from app.utils import simple_chunk_text

def test_chunk_small():
    text = 'A. ' + ('hello ' * 100)
    chunks = simple_chunk_text(text, chunk_size_chars=200, overlap_chars=20)
    assert len(chunks) >= 1
