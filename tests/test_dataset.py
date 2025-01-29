import pytest

from lrrnn.dataset import (
    CorpusType,
    parse_syn,
    load_corpus,
    format_morpheme,
    format_sentence,
    save_sentences_to_file,
)


SAMPLE_KNP_CONTENT = (
    "# S-ID:950101001-001\n"
    "* 0 2D\n"
    "太郎 たろう * 名詞 人名 * * \n"
    "は は * 助詞 副助詞 * *\n"
    "* 1 2D\n"
    "東京 とうきょう * 名詞 固有名詞 * *\n"
    "大学 だいがく * 名詞 普通名詞 * *\n"
    "に に * 助詞 格助詞 * *\n"
    "* 2 -1D\n"
    "行った いった 行く 動詞 * 子音動詞カ行促音便形 基本形\n"
    "EOS\n"
)

SAMPLE_KNP_PARSED = [
    [
        {
            "surface": "太郎",
            "yomi": "たろう",
            "base_form": "*",
            "pos1": "名詞",
            "pos2": "人名",
            "group": "*",
            "conjugation": "*",
        },
        {
            "surface": "は",
            "yomi": "は",
            "base_form": "*",
            "pos1": "助詞",
            "pos2": "副助詞",
            "group": "*",
            "conjugation": "*",
        },
        {
            "surface": "東京",
            "yomi": "とうきょう",
            "base_form": "*",
            "pos1": "名詞",
            "pos2": "固有名詞",
            "group": "*",
            "conjugation": "*",
        },
        {
            "surface": "大学",
            "yomi": "だいがく",
            "base_form": "*",
            "pos1": "名詞",
            "pos2": "普通名詞",
            "group": "*",
            "conjugation": "*",
        },
        {
            "surface": "に",
            "yomi": "に",
            "base_form": "*",
            "pos1": "助詞",
            "pos2": "格助詞",
            "group": "*",
            "conjugation": "*",
        },
        {
            "surface": "行った",
            "yomi": "いった",
            "base_form": "行く",
            "pos1": "動詞",
            "pos2": "*",
            "group": "子音動詞カ行促音便形",
            "conjugation": "基本形",
        },
    ]
]


def test_parse_syn(tmp_path):
    knp_file = tmp_path / "sample.KNP"
    knp_file.write_text(
        SAMPLE_KNP_CONTENT,
        encoding="euc-jp",
    )

    sentences = parse_syn(knp_file)
    assert sentences == SAMPLE_KNP_PARSED


def test_load_corpus(tmp_path):
    corpus_dir = tmp_path / "KyotoCorpus4.0"
    corpus_dir.mkdir()

    knp_files = ["sample0.KNP", "sample1.KNP"]
    DAT = "dat"
    SYN = "syn"
    (corpus_dir / DAT).mkdir()
    (corpus_dir / DAT / SYN).mkdir()
    sample0 = corpus_dir / DAT / SYN / knp_files[0]
    sample0.write_text(SAMPLE_KNP_CONTENT, encoding="euc-jp")
    sample1 = corpus_dir / DAT / SYN / knp_files[1]
    sample1.write_text(SAMPLE_KNP_CONTENT, encoding="euc-jp")

    sentences = load_corpus(knp_files, corpus_dir)
    assert sentences == [SAMPLE_KNP_PARSED[0], SAMPLE_KNP_PARSED[0]]


def test_format_morpheme():
    morpheme = {
        "surface": "太郎",
        "yomi": "たろう",
        "base_form": "*",
        "pos1": "名詞",
        "pos2": "人名",
        "group": "*",
        "conjugation": "*",
    }
    assert (
        format_morpheme(morpheme, CorpusType.KANA) == "たろう/名詞/人名/*/*/太郎/たろう"
    )
    assert (
        format_morpheme(morpheme, CorpusType.KANJI) == "太郎/名詞/人名/*/*/太郎/たろう"
    )
    assert format_morpheme(morpheme, CorpusType.UNSEGMENTED_KANA) == "たろう"
    assert format_morpheme(morpheme, CorpusType.UNSEGMENTED_KANJI) == "太郎"
    with pytest.raises(ValueError):
        format_morpheme(morpheme, 0)


def test_format_sentence():
    sentence = SAMPLE_KNP_PARSED[0]
    assert format_sentence(sentence, CorpusType.KANA) == (
        "たろう/名詞/人名/*/*/太郎/たろう "
        "は/助詞/副助詞/*/*/は/は "
        "とうきょう/名詞/固有名詞/*/*/東京/とうきょう "
        "だいがく/名詞/普通名詞/*/*/大学/だいがく "
        "に/助詞/格助詞/*/*/に/に "
        "いった/動詞/*/子音動詞カ行促音便形/基本形/行った/いった"
    )
    assert format_sentence(sentence, CorpusType.KANJI) == (
        "太郎/名詞/人名/*/*/太郎/たろう "
        "は/助詞/副助詞/*/*/は/は "
        "東京/名詞/固有名詞/*/*/東京/とうきょう "
        "大学/名詞/普通名詞/*/*/大学/だいがく "
        "に/助詞/格助詞/*/*/に/に "
        "行った/動詞/*/子音動詞カ行促音便形/基本形/行った/いった"
    )
    assert (
        format_sentence(sentence, CorpusType.UNSEGMENTED_KANA)
        == "たろうはとうきょうだいがくにいった"
    )
    assert (
        format_sentence(sentence, CorpusType.UNSEGMENTED_KANJI)
        == "太郎は東京大学に行った"
    )
    with pytest.raises(ValueError):
        format_sentence(sentence, 0)


def test_save_sentences_to_file(tmp_path):
    sentences = SAMPLE_KNP_PARSED
    file_path = tmp_path / "sample.txt"
    corpus_type = CorpusType.KANA
    save_sentences_to_file(sentences, file_path, corpus_type)

    assert file_path.read_text() == (
        "たろう/名詞/人名/*/*/太郎/たろう "
        "は/助詞/副助詞/*/*/は/は "
        "とうきょう/名詞/固有名詞/*/*/東京/とうきょう "
        "だいがく/名詞/普通名詞/*/*/大学/だいがく "
        "に/助詞/格助詞/*/*/に/に "
        "いった/動詞/*/子音動詞カ行促音便形/基本形/行った/いった\n"
    )
