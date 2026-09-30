from claudius.packing import count_tokens, pack


def test_pack_produces_fixed_length_blocks():
    docs = [list(range(1, 11))] * 10
    seqs = list(pack(docs, eos_id=0, seq_len=8))
    assert all(len(s) == 8 for s in seqs)
    assert len(seqs) == (10 * 11) // 8


def test_eos_separates_documents():
    seqs = list(pack([[1, 2], [3, 4]], eos_id=0, seq_len=3))
    assert seqs == [[1, 2, 0], [3, 4, 0]]


def test_tail_is_dropped_not_padded():
    seqs = list(pack([[1, 2, 3]], eos_id=0, seq_len=8))
    assert seqs == []


def test_count_tokens():
    assert count_tokens([[1, 2], [3]]) == 3
