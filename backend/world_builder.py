current_word = ""

def add_letter(letter):

    global current_word

    current_word += letter

    return current_word


def clear_word():

    global current_word

    current_word = ""

    return current_word