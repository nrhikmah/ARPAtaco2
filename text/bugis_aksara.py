lontara_to_latin = str.maketrans({
    "ᨀ":"ka",
    "ᨁ":"ga",
    "ᨂ":"nga",
    "ᨃ":"ngka",
    "ᨄ":"pa",
    "ᨅ":"ba",
    "ᨆ":"ma",
    "ᨇ":"mpa",
    "ᨈ":"ta",
    "ᨉ":"da",
    "ᨊ":"na",
    "ᨋ":"nra",
    "ᨌ":"ca",
    "ᨍ":"ja",
    "ᨎ":"nya",
    "ᨏ":"nca",
    "ᨐ":"ya",
    "ᨑ":"ra",
    "ᨒ":"la",
    "ᨓ":"wa",
    "ᨔ":"sa",
    "ᨕ":"a",
    "ᨖ":"ha","ᨗ":"i","ᨘ":"u","ᨙ":"é","ᨚ":"o","ᨛ":"e","᨞":",","᨟":"."
})

def remove_one_char_before_vowel(text):
    result = []
    skip_next = False

    for i, char in enumerate(text):
        if skip_next:
            skip_next = False
            continue

        if char == 'i':
            if i > 0:
                result.pop()
        if char == 'u':
            if i > 0:
                result.pop()
        if char == 'é':
            if i > 0:
                result.pop()
        if char == 'o':
            if i > 0:
                result.pop()
        if char == 'e':
            if i > 0:
                result.pop()
                result.append('e')
        else:
            result.append(char)

    return ''.join(result)