from collections import Counter
import re
from functools import reduce

with open('sample.txt', 'r') as file:
    text = file.read().lower()
    words = re.findall(r'\w+', text)
    word_counts = Counter(words)

top_words = word_counts.most_common(10)

total_count = reduce(lambda x, y: x + y[1], top_words, 0)

word_percentages = {word: count/total_count*100 for word, count in top_words}

print("Top 10 words and their frequencies:")
for word, percentage in sorted(word_percentages.items(), key=lambda x: x[1], reverse=True):
    print(f"{word:<10} : {percentage:.2f}%")

longest_word = max(words, key=len)
print(f"\nLongest word: {longest_word}")