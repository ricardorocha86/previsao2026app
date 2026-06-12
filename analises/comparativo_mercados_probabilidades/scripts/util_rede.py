# -*- coding: utf-8 -*-
"""
Utilidades de rede compartilhadas pelos scripts de extração.

Inclui o contorno de bloqueio de DNS do provedor (no Brasil, domínios de
mercados de previsão como Kalshi e Polymarket costumam ser bloqueados no
DNS do provedor). O contorno resolve o domínio via DNS-over-HTTPS do
Google (https://8.8.8.8/resolve) e conecta direto no IP — a conexão TLS
continua usando o hostname correto (SNI), então o certificado valida.
"""
import socket

import requests
import urllib3.util.connection as _uc

UA = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36'
}

_cache_dns = {}
_patch_aplicado = False


def _resolver_via_doh(host):
    """Resolve um hostname via DNS-over-HTTPS do Google."""
    if host not in _cache_dns:
        r = requests.get('https://8.8.8.8/resolve',
                         params={'name': host, 'type': 'A'}, timeout=15)
        respostas = [a['data'] for a in r.json().get('Answer', [])
                     if a.get('type') == 1]
        if not respostas:
            raise RuntimeError(f'DNS-over-HTTPS não resolveu {host}')
        _cache_dns[host] = respostas[0]
    return _cache_dns[host]


def aplicar_contorno_dns():
    """
    Ativa o fallback de DNS: se o DNS local falhar (bloqueio do provedor),
    resolve via DNS-over-HTTPS. Idempotente.
    """
    global _patch_aplicado
    if _patch_aplicado:
        return
    _original = _uc.create_connection

    def _com_fallback(address, *args, **kwargs):
        host, port = address
        try:
            socket.getaddrinfo(host, port)
            return _original(address, *args, **kwargs)
        except socket.gaierror:
            ip = _resolver_via_doh(host)
            return _original((ip, port), *args, **kwargs)

    _uc.create_connection = _com_fallback
    _patch_aplicado = True
