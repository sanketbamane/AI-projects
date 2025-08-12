import json, os
kb = [
    {"id":1, "text":"Our refund policy allows returns within 30 days."},
    {"id":2, "text":"We offer 24/7 customer support via chat."},
    {"id":3, "text":"Premium plan includes advanced analytics and priority support."}
]
MEMFILE = 'src/memory.json'
if os.path.exists(MEMFILE):
    memory = json.load(open(MEMFILE))
else:
    memory = []
def retrieve(query):
    # naive keyword match
    q = query.lower()
    for doc in kb:
        if any(w in doc['text'].lower() for w in q.split()):
            return doc['text']
    return "I don't know, please check our docs."

while True:
    q = input("You: ")
    if q.strip().lower() in ('exit','quit'):
        break
    resp = retrieve(q)
    print("Bot:", resp)
    memory.append({'query':q, 'response':resp})
    json.dump(memory, open(MEMFILE,'w'), indent=2)
