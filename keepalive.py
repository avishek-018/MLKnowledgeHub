# keepalive.py
import os
from neo4j import GraphDatabase
from playwright.sync_api import sync_playwright

# 1. Neo4j: tiny write so it counts as activity
with GraphDatabase.driver(os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])) as d:
    d.execute_query("MERGE (h:Heartbeat {id: 1}) SET h.ts = datetime()")

# 2. Streamlit: real browser so the websocket connects
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page()
    page.goto("https://mlknowledge.streamlit.app/")
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        wake.click()
    page.wait_for_timeout(30000)
    b.close()
