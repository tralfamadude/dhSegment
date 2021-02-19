import numpy as np
from string import punctuation

class TextUtil:
    def __init__(self):
        # prep chars to remove except single quote and comma
        self.charsToRemove = punctuation.replace("'", "").replace(",", "").replace("-", "").replace(".", "")
        #  and add some other chars to remove
        self.charsToRemove += "®“”"
        self.charsToRemoveMap = np.full((65536), False)
        for i in range(len(self.charsToRemove)):
            c = self.charsToRemove[i]
            self.charsToRemoveMap[ord(c)] = True

    def removeIt(self, c):
        """
        :param c:  char to test
        :return: True of char should be removed.
        """
        return self.charsToRemoveMap[ord(c)]

    def cleanAuthors(self, authors):
        """
        Clean up authors string which will contain mis-recognized superscripts, but keep single quote
        char for names like O'Reilly.
        :param authors: ocr chars from authors block
        :return: cleaned string
        """
        result = ""
        offset = 0
        n = len(authors)
        try:
            while offset < n:
                c = authors[offset]
                offset += 1
                if c.isalpha() or c == ' ' or c == '-' or c == '.':
                    result += c
                    continue
                if c == ',':
                    result += c  #  keep comma
                    result += ' '  # space after comma
                    offset += 1
                    if offset >= n:
                        break  #  unlikely to see comma at end
                    c = authors[offset]
                    while self.removeIt(c):  #  skip chars
                        offset += 1 # skip
                        if offset >= n:  #  safety
                            break
                        c = authors[offset]
                    # now we are probably have c==' '
                    if c == ' ':
                        continue
                    # now we are looking to remove non-alpha chars until we see an alpha
                    while not c.isalpha() and c == "'" and offset < n:
                        offset += 1
                        continue
        except Exception:  # just in case
            print(f"Exception occurred cleaning:  {authors}")
        result = result.replace("\n", " ")  # convert EOL chars to space
        result = " ".join(result.split())   # remove consecutive spaces
        return result

    def one_line(self, s):
        s = s.replace("\n", " ")  # convert EOL chars to space
        s = " ".join(s.split())   # remove consecutive spaces
        return s

if __name__ == '__main__':
    test_example = 'A Conde-Agudelo,* AT Papageorghiou,"* SH Kennedy,” J Villar®“'
    test_result = 'A Conde-Agudelo, AT Papageorghiou, SH Kennedy, J Villar'
    cleaner = TextUtil()
    print(f"{cleaner.charsToRemoveMap}")
    r = cleaner.cleanAuthors(test_example)
    good = test_result == r
    print(f" {good}\n  {test_example}\n  {test_result}\n  {r}")
