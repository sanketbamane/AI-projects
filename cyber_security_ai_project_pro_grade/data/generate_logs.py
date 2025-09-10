import argparse, random, pandas as pd
from datetime import datetime, timedelta
SAMPLE_BENIGN = [ "Accepted password for user from 10.0.0.1 port 22", "Service started: apache2", "User logged in" ]
SAMPLE_MALICIOUS = [ "Failed password for invalid user admin from 192.168.5.5 port 22", "SQL injection attempt: ' OR '1'='1", "Multiple failed login attempts for root" ]
SOURCES = ['ssh','web','app','db','sys']

def gen(n=1000, mal_prob=0.15, out='data/logs.csv'):
    rows = []
    for _ in range(n):
        mal = random.random() < mal_prob
        rows.append({
            'timestamp': (datetime.utcnow() - timedelta(seconds=random.randint(0,86400))).isoformat()+'Z',
            'source': random.choice(SOURCES),
            'message': random.choice(SAMPLE_MALICIOUS if mal else SAMPLE_BENIGN),
            'label': 1 if mal else 0
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--out', default='data/logs.csv')
    args = p.parse_args()
    gen(args.n, out=args.out)
