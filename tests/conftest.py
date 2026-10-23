"""Integration tests run in disposable schemas of a dedicated test database."""
import os
import uuid
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from paperkb.config import settings
from paperkb.db import connect, migrate

@pytest.fixture
def conn():
    options = conninfo_to_dict(os.environ.get('PAPERKB_TEST_DATABASE_URL') or settings()['database_url'])
    options['dbname'] = 'paperkb_test'
    connection = connect(make_conninfo(**options))
    schema = 'test_' + uuid.uuid4().hex
    connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    connection.execute(sql.SQL('SET search_path TO {},public').format(sql.Identifier(schema)))
    migrate(connection)
    connection.commit()
    try:
        yield connection
    finally:
        connection.rollback()
        connection.execute('SET search_path TO public')
        connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        connection.commit()
        connection.close()
