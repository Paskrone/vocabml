import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool

_DSN = os.getenv("PG_DSN", "postgresql://pascalgrcic@localhost:5432/vocabml")

_POOL = SimpleConnectionPool(
    minconn=int(os.getenv("PG_MINCONN", "1")),
    maxconn=int(os.getenv("PG_MAXCONN", "5")),
    dsn=_DSN,
)

def get_conn():
    return _POOL.getconn()

def put_conn(conn):
    _POOL.putconn(conn)
