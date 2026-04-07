import random

with open("quotes.txt", "r") as f:
    lines = f.readlines()
    quote = random.choice(lines).strip()
    length = max(len(line) for line in quote.split("\\n"))
    formatted_quote = quote.replace("\\n", "\n")

    print("\n" + "="*length)
    print(formatted_quote)
    print("="*length + "\n")