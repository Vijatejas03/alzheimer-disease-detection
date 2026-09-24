import requests

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
})

r = s.get("https://alzheimer-xai-vijay.streamlit.app", allow_redirects=False)
print("First request status:", r.status_code)
print("Location:", r.headers.get("Location"))
print("Set-Cookie:", r.headers.get("Set-Cookie"))

r_full = s.get("https://alzheimer-xai-vijay.streamlit.app")
print("\nFollowed redirects:")
print("Final status:", r_full.status_code)
print("Final URL:", r_full.url)
print("History length:", len(r_full.history))
for i, h in enumerate(r_full.history):
    print(f"  Step {i}: {h.status_code} -> {h.headers.get('Location')}")
    
print("\nFinal response length:", len(r_full.text))
print("Final response snippet:\n", r_full.text[:500])
