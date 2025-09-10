import os, requests
from api.config import settings

OIDC_INTROSPECT = settings.OIDC_INTROSPECT
CLIENT_ID = settings.OIDC_CLIENT_ID
CLIENT_SECRET = settings.OIDC_CLIENT_SECRET

def verify_oidc_token(token: str):
    if not OIDC_INTROSPECT or not CLIENT_ID or not CLIENT_SECRET:
        return None
    try:
        resp = requests.post(OIDC_INTROSPECT, data={'token': token, 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}, timeout=5)
        if resp.status_code == 200:
            d = resp.json()
            if d.get('active'):
                return d.get('username') or d.get('sub')
    except Exception as e:
        print('introspect error', e)
    return None
