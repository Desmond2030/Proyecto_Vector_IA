# ESTADO_ARCHIVO: EN_MODIFICACION
"""Único punto de comunicación con FastAPI; la app no carga modelos."""
import os
import requests


class ErrorAPI(ValueError):
    pass


class ApiClient:
    def __init__(self, url=None):
        self.url = (url or os.environ.get('VECTORIA_API_URL', 'http://127.0.0.1:8000')).rstrip('/')

    def request(self, method, path, payload=None, *, binary=None, params=None, timeout=180):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.request(method, self.url + path, json=payload, data=binary,
                                           params=params, timeout=(5, timeout))
            data = response.json()
        except requests.RequestException as exc:
            raise ErrorAPI('No hay conexión con VectorIA. Inicia los servicios con python app/run.py.') from exc
        except ValueError as exc:
            raise ErrorAPI('El servidor no devolvió JSON válido.') from exc
        if not response.ok:
            detail = data.get('detail', 'Error del servidor')
            if isinstance(detail, list):
                detail = '; '.join(f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg', '')}" for e in detail)
            raise ErrorAPI(str(detail))
        return data

    def salud(self):
        value = self.request('GET', '/health', timeout=5)
        if value.get('servicio') != 'vectoria':
            raise ErrorAPI('El puerto está ocupado por otro servicio. Se esperaba VectorIA.')
        return value

    def subir(self, archivo, tipo='datos'):
        return self.request('POST', f'/files/{tipo}', binary=archivo.getvalue(), params={'nombre': archivo.name})
