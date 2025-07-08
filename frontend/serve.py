#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur web simple pour l'interface PMR
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 3000
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Interface PMR disponible sur: http://localhost:{PORT}")
        print(f"📁 Dossier servi: {DIRECTORY}")
        print("🔄 Appuyez sur Ctrl+C pour arrêter")
        
        # Ouvrir automatiquement le navigateur
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Serveur arrêté")
