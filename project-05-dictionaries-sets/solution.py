"""Project 05: Dictionaries and Sets - SOLUTION"""
def count_words(text):
    """Count word frequencies."""
    word_count = {}
    words = text.split()
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

def find_unique_elements(items):
    """Find unique elements."""
    return list(set(items))

def set_union(set1, set2):
    """Return union."""
    return set1 | set2

def set_intersection(set1, set2):
    """Return intersection."""
    return set1 & set2

def set_difference(set1, set2):
    """Return difference."""
    return set1 - set2

