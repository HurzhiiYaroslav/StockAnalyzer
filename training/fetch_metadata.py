import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.services import metadata_service

def main():
    """
    Точка входу для запуску процесу повного оновлення кешу метаданих.
    """
    metadata_service.populate_full_cache()

if __name__ == "__main__":
    main()