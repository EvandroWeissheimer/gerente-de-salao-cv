"""Utilidades da janela do OpenCV (maximizar / tela cheia)."""
from __future__ import annotations

import sys

import cv2


def maximizar(nome: str) -> None:
    """Maximiza a janela (continua com barra de título e barra de tarefas).

    Chame depois do primeiro cv2.imshow, quando a janela já existe.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, nome)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # 3 = SW_MAXIMIZE
                return
        except Exception:
            pass
    # Outros sistemas: redimensiona para o tamanho da tela
    try:
        import tkinter
        raiz = tkinter.Tk()
        raiz.withdraw()
        largura, altura = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
        raiz.destroy()
        cv2.resizeWindow(nome, largura, altura - 80)
    except Exception:
        pass


def tela_cheia(nome: str, ativar: bool) -> None:
    if ativar:
        cv2.setWindowProperty(nome, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(nome, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        maximizar(nome)
