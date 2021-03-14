"""Submodule internal_utils.py includes the following functions: <br>
- postgresql_data_extraction():
- postgresql_table_names_list():
"""
import os

import pandas as pd
import psycopg2


def postgresql_data_extraction(
    table_name: str = "suggested_energy_intake",
    database_name: str = "tracker",
    user: str = "tracker",
):
    r"""
    Load data from a specified postgresql database.

    Parameters
    ----------
    table_name: str
        The name of the table to extract from the postgresql database.
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.

    Returns
    -------
    pd.DataFrame

    Examples
    ---------
    >>> from jmspack.internal_utils import postgresql_data_extraction
    >>> df = postgresql_data_extraction()
    """
    df = pd.DataFrame()
    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        df = pd.read_sql_query(f"SELECT * from {table_name}", conn)
        _ = conn.close()

    except:
        print("I am unable to connect to the database")

    return df


def postgresql_table_names_list(
    database_name: str = "tracker",
    user="tracker",
):
    r"""
    Extract the table names from a specified postgresql database.

    Parameters
    ----------
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.

    Returns
    -------
    list

    Examples
    ---------
    >>> from jmspack.internal_utils import postgresql_table_names_list
    >>> table_names = postgresql_table_names_list()
    """
    table_list = False
    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        cursor = conn.cursor()
        cursor.execute(
            "select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';"
        )
        table_list = cursor.fetchall()
        cursor.close()
        _ = conn.close()
    except psycopg2.Error as e:
        print(e)
    return table_list
