# ESTADO_ARCHIVO: EN_MODIFICACION
"""Ejecutar desde la raíz: .Vector_IA_wsl/bin/python app/run.py."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.client import ApiClient, ErrorAPI


def libre(port):
    with socket.socket() as connection:
        return connection.connect_ex(('127.0.0.1', port)) != 0


def main():
    children = []
    api = ApiClient('http://127.0.0.1:8000')
    env = {**os.environ, 'VECTORIA_API_URL': api.url, 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        if not libre(8501):
            raise RuntimeError('El puerto 8501 ya está ocupado. Cierra la instancia anterior.')
        if libre(8000):
            children.append(subprocess.Popen([sys.executable, '-B', '-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', '8000'], cwd=ROOT, env=env))
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if children[0].poll() is not None:
                    raise RuntimeError('La API no pudo iniciarse.')
                try:
                    api.salud()
                    break
                except ErrorAPI:
                    time.sleep(.4)
            else:
                raise RuntimeError('La API no estuvo lista en 180 segundos.')
        else:
            api.salud()
        children.append(subprocess.Popen([sys.executable, '-B', '-m', 'streamlit', 'run', 'app/Home.py',
                        '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true',
                        '--server.runOnSave', 'true', '--server.fileWatcherType', 'poll',
                        '--browser.gatherUsageStats', 'false', '--theme.base', 'light',
                        '--theme.backgroundColor', '#f7f6f2', '--theme.secondaryBackgroundColor', '#eeeee7',
                        '--theme.textColor', '#242727', '--theme.primaryColor', '#087F72'], cwd=ROOT, env=env))
        print('VectorIA: http://127.0.0.1:8501 · API: http://127.0.0.1:8000/docs', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
    except KeyboardInterrupt:
        print('Cerrando VectorIA…')
    except (RuntimeError, ErrorAPI) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
